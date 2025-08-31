# scripts/prepare_dataset.py

import argparse
import gc
import sys
from pathlib import Path
import traceback

import yaml
from tqdm import tqdm

from etude.decode.tokenizer import TinyREMITokenizer
from etude.decode.vocab import Vocab, PAD_TOKEN
from etude.data.dataset import EtudeDataset


def _tokenize_and_build_vocab(base_dir: Path, vocab_path: Path) -> tuple:
    """
    Scans subdirectories, tokenizes MIDI data into Event sequences,
    and builds a vocabulary from all events.
    """
    print(f"[STEP 1/3] Tokenizing source files and building vocabulary...")
    all_cond_events, all_tgt_events = [], []
    processed_dirs = []

    if not base_dir.exists():
        print(f"[ERROR] Base data directory not found: {base_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    original_subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    print(f"    > Found {len(original_subdirs)} potential subdirectories.")
    
    for original_dir in tqdm(original_subdirs, desc="    > Tokenizing"):
        tempo_file = original_dir / "tempo.json"
        cond_file = original_dir / "extract.json"
        tgt_file = original_dir / "cover.json"
        
        if not (tempo_file.exists() and cond_file.exists() and tgt_file.exists()):
            continue
        
        try:
            cond_tokenizer = TinyREMITokenizer(str(tempo_file))
            cond_events = cond_tokenizer.encode(str(cond_file), with_grace_note=False)
            tgt_tokenizer = TinyREMITokenizer(str(tempo_file))
            tgt_events = tgt_tokenizer.encode(str(tgt_file), with_grace_note=True)
            
            if cond_events and tgt_events:
                all_cond_events.append(cond_events)
                all_tgt_events.append(tgt_events)
                processed_dirs.append(original_dir.name)
        except Exception as e:
            print(f"[ERROR] Error processing {original_dir.name}: {e}", file=sys.stderr)

    if not processed_dirs:
        print("[ERROR] No data was successfully tokenized. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    vocab = Vocab(special_tokens=[PAD_TOKEN])
    vocab.build_from_events(all_cond_events + all_tgt_events)
    vocab.save(vocab_path)
    
    print(f"[INFO] Step 1 completed. Processed {len(processed_dirs)} directories and found {len(vocab)} unique tokens.")
    
    return all_cond_events, all_tgt_events, processed_dirs

def _encode_and_save_sequences(
    vocab: Vocab,
    cond_events_list: list,
    tgt_events_list: list,
    processed_dir_names: list,
    output_dir: Path,
    save_format: str
) -> int:
    """
    Encodes all event sequences using the vocabulary and saves them
    to the output directory.
    """
    print(f"\n[STEP 2/3] Encoding {len(processed_dir_names)} sequences and saving to disk.")
    num_saved = 0
    for i, dir_name in enumerate(tqdm(processed_dir_names, desc="    > Encoding")):
        basename = f"{i+1:04d}"
        output_subdir = output_dir / basename
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        cond_output_path = output_subdir / f"{basename}_cond.{save_format}"
        tgt_output_path = output_subdir / f"{basename}_tgt.{save_format}"
        
        try:
            vocab.encode_and_save_sequence(cond_events_list[i], cond_output_path, format=save_format)
            vocab.encode_and_save_sequence(tgt_events_list[i], tgt_output_path, format=save_format)
            num_saved += 1
        except Exception as e:
            print(f"[ERROR] Error encoding/saving for {dir_name}: {e}", file=sys.stderr)
            
    print(f"[INFO] Step 2 completed. Saved {num_saved} tokenized pairs.")
    return num_saved

def _validate_dataset_and_print_stats(output_dir: Path, vocab: Vocab, config: dict):
    """
    Initializes the EtudeDataset on the newly created data
    to validate it and print detailed statistics.
    """
    print(f"\n[STEP 3/3] Validating final dataset and calculating statistics...")
    try:
        save_format = config['save_format']
        dataset = EtudeDataset(
            dataset_dir=output_dir,
            vocab=vocab,
            max_seq_len=config['max_seq_len_for_stats'],
            cond_suffix=f'_cond.{save_format}',
            tgt_suffix=f'_tgt.{save_format}',
            data_format=save_format,
            verbose_stats=True  # Ensure the detailed stats are printed
        )
        
        if len(dataset) == 0:
            print("\n[WARN] Dataset validation finished, but the dataset contains zero valid samples.")

    except Exception as e:
        print(f"[ERROR] Error during dataset validation: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    
    print("[INFO] Step 3 completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the dataset for the 'decode' stage. This script tokenizes, builds a vocabulary, encodes sequences, and validates the final dataset."
    )
    parser.add_argument(
        "--config", type=str, default="configs/dataset_config.yaml",
        help="Path to the dataset configuration YAML file."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    base_data_dir = Path(config['base_data_dir'])
    output_dir = Path(config['output_dir'])
    vocab_path = output_dir / config['vocab_filename']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Commencing dataset preparation pipeline.")
    print(f"    - Input data directory:  {base_data_dir.resolve()}")
    print(f"    - Output data directory: {output_dir.resolve()}")
    print(f"    - Vocabulary path:       {vocab_path.resolve()}\n")

    # --- Execute Pipeline ---
    cond_events, tgt_events, dir_names = _tokenize_and_build_vocab(base_data_dir, vocab_path)
    
    gc.collect() 
    
    try:
        vocab = Vocab.load(vocab_path)
    except Exception as e:
        print(f"  [ERROR] Could not reload vocab for encoding: {e}.", file=sys.stderr)
        sys.exit(1)
        
    num_saved = _encode_and_save_sequences(
        vocab, cond_events, tgt_events, dir_names, output_dir, config['save_format']
    )
    
    del cond_events, tgt_events, dir_names
    gc.collect()

    if num_saved > 0:
        _validate_dataset_and_print_stats(output_dir, vocab, config)
    else:
        print("\n[WARN] No sequences were saved. Skipping dataset validation.")
    
    print("\n[INFO] Dataset preparation finished.")
    print(f"    - Final dataset is ready in: {output_dir.resolve()}.")

if __name__ == "__main__":
    main()