import sys, pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from pathlib import Path
import traceback
from tqdm import tqdm
import gc

from corpus.tokenizer_v2 import MidiTokenizer, Vocab
from corpus.dataset import EtudeDataset

# --- Configuration ---
DATASET_DIR = Path("./dataset/")
BASE_DATA_DIR = Path("./dataset/synced/")
PREPROCESSED_DIR = Path("./dataset/tokenized/")
VOCAB_PATH = PREPROCESSED_DIR / "vocab.json"
DATA_SAVE_FORMAT = 'npy'
MAX_SEQ_LEN_DATASET = 1024
PAD_TOKEN = "<PAD>"
# BATCH_SIZE = 8 # Batch size for DataLoader demo, not strictly needed for stats
# NUM_WORKERS_DATALOADER = 0

# --- Step 0: Ensure Output Directory Exists ---
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory set to: {PREPROCESSED_DIR.resolve()}")
print(f"Vocabulary will be saved to: {VOCAB_PATH.resolve()}")

# --- Phase 1 & 2: Tokenize, Collect Events, Build Vocab ---
# (這部分保持和您提供的腳本類似，但確保 Vocab 初始化正確)
print("\n--- Phase 1 & 2: Tokenizing and Building Vocabulary ---")
# ... (假設這部分成功執行並創建了 VOCAB_PATH) ...
# --- Start: Code from your Phase 1 & 2 ---
all_cond_event_sequences = []
all_tgt_event_sequences = []
processed_original_dirs = []

if not BASE_DATA_DIR.exists():
    print(f"Error: Base data directory not found: {BASE_DATA_DIR.resolve()}")
    sys.exit(1)
original_subdirs = sorted([d for d in BASE_DATA_DIR.iterdir() if d.is_dir()])
print(f"Found {len(original_subdirs)} potential subdirectories in {BASE_DATA_DIR}.")
for original_dir in tqdm(original_subdirs, desc="Scanning & Tokenizing"):
    tempo_file = original_dir / "tempo.json"
    cond_file = original_dir / "extract.json"
    tgt_file = original_dir / "cover.json"
    if not (tempo_file.exists() and cond_file.exists() and tgt_file.exists()): continue
    try:
        cond_tokenizer = MidiTokenizer(str(tempo_file))
        cond_events = cond_tokenizer.encode(str(cond_file), with_grace_note=False)
        tgt_tokenizer = MidiTokenizer(str(tempo_file))
        tgt_events = tgt_tokenizer.encode(str(tgt_file), with_grace_note=True)
        if cond_events and tgt_events:
            all_cond_event_sequences.append(cond_events)
            all_tgt_event_sequences.append(tgt_events)
            processed_original_dirs.append(original_dir)
    except Exception as e: print(f"Error processing {original_dir.name}: {e}")
print(f"Successfully tokenized {len(processed_original_dirs)} directories.")
if not processed_original_dirs: print("Error: No data tokenized."); sys.exit(1)

vocab = Vocab(special_tokens=[PAD_TOKEN])
vocab.build_from_events(all_cond_event_sequences + all_tgt_event_sequences)
vocab.save(VOCAB_PATH)
print(f"Vocabulary saved with {len(vocab)} tokens.")
# --- End: Code from your Phase 1 & 2 ---


# --- Phase 3: Encode Sequences with Final Vocab and Save ---
# (這部分保持和您提供的腳本類似)
# ... (假設這部分成功執行) ...
# --- Start: Code from your Phase 3 ---
try:
    vocab = Vocab.load(VOCAB_PATH)
    print(f"Successfully reloaded vocabulary from {VOCAB_PATH}.")
except Exception as e: print(f"Error reloading vocab: {e}"); sys.exit(1)
num_saved = 0
for i in tqdm(range(len(processed_original_dirs)), desc="Encoding & Saving"):
    cond_events = all_cond_event_sequences[i]
    tgt_events = all_tgt_event_sequences[i]
    basename = f"{i+1:04d}"
    output_subdir = PREPROCESSED_DIR / basename
    output_subdir.mkdir(parents=True, exist_ok=True)
    cond_output_path = output_subdir / f"{basename}_cond.{DATA_SAVE_FORMAT}"
    tgt_output_path = output_subdir / f"{basename}_tgt.{DATA_SAVE_FORMAT}"
    try:
        vocab.encode_and_save_sequence(cond_events, cond_output_path, format=DATA_SAVE_FORMAT)
        vocab.encode_and_save_sequence(tgt_events, tgt_output_path, format=DATA_SAVE_FORMAT)
        num_saved += 1
    except Exception as e: print(f"Error encoding/saving for {processed_original_dirs[i].name}: {e}")
print(f"Successfully encoded and saved {num_saved} pairs to {PREPROCESSED_DIR}.")
del all_cond_event_sequences, all_tgt_event_sequences, processed_original_dirs
gc.collect()
# --- End: Code from your Phase 3 ---


# --- Phase 4 & 5: Initialize Dataset and Calculate Statistics ---
print("\n--- Phase 4 & 5: Initializing Dataset and Calculating Statistics ---")
if num_saved == 0:
     print("No data was saved. Cannot initialize dataset or calculate statistics.")
else:
    print(f"Initializing EtudeDataset with data from: {PREPROCESSED_DIR.resolve()} for statistics...")
    dataset_instance = None
    try:
        dataset_instance = EtudeDataset(
            dataset_dir=PREPROCESSED_DIR,
            vocab=vocab,
            max_seq_len=MAX_SEQ_LEN_DATASET,
            cond_suffix=f'_cond.{DATA_SAVE_FORMAT}',
            tgt_suffix=f'_tgt.{DATA_SAVE_FORMAT}',
            data_format=DATA_SAVE_FORMAT
        )
        print(f"\nEtudeDataset initialization complete. Number of processed samples (bar-pairs): {len(dataset_instance)}")

        if len(dataset_instance) == 0:
            print("Dataset is empty after initialization and filtering. Check logs.")

    except Exception as e:
        print(f"Error during EtudeDataset initialization or statistics display: {e}")
        print(traceback.format_exc())

print("\n--- Script Finished ---")