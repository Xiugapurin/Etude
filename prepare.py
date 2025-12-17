# prepare.py

import argparse
import sys
from pathlib import Path
import subprocess

import pandas as pd
import yaml
import json
from tqdm import tqdm

from etude.utils.download import download_audio_from_url
from etude.utils.logger import logger
from etude.models.hft_transformer import HFT_Transformer
from etude.data.beat_analyzer import BeatAnalyzer
from etude.data.aligner import AudioAligner
from etude.utils.preprocess import (
    compute_wp_std,
    create_time_map_from_downbeats,
    weakly_align
)
from etude.data.extractor import AMTAPC_Extractor
from etude.data.tokenizer import TinyREMITokenizer
from etude.data.vocab import Vocab, PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN

def run_stage_1_download(config: dict, verbose: bool = False):
    """
    Handles Stage 1: Downloading all raw audio files from the source CSV.
    """
    logger.stage(1, "Downloading Raw Audio")

    stage_config = config['download']
    csv_path = Path(stage_config['csv_path'])
    output_dir = Path(stage_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error(f"Input CSV file not found at: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        if verbose:
            logger.info(f"Loaded '{csv_path.name}'. Found {len(df)} song pairs to process.")
    except Exception as e:
        logger.error(f"Failed to read or parse CSV file: {e}")
        sys.exit(1)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Downloading"):
        song_index = index + 1
        piano_id, pop_id = row['piano_ids'], row['pop_ids']

        song_dir = output_dir / f"{song_index:04d}"
        song_dir.mkdir(exist_ok=True)

        cover_output_path = song_dir / "cover.wav"
        origin_output_path = song_dir / "origin.wav"

        if not cover_output_path.exists():
            piano_url = f"https://www.youtube.com/watch?v={piano_id}"
            download_audio_from_url(piano_url, cover_output_path)

        if not origin_output_path.exists():
            pop_url = f"https://www.youtube.com/watch?v={pop_id}"
            download_audio_from_url(pop_url, origin_output_path)

    logger.info("Stage 1: Download complete.")


def run_stage_2_preprocess(config: dict, verbose: bool = False):
    """
    Handles Stage 2: Generates all intermediate analysis files
    (beat_pred.json, tempo.json, transcription.json).
    """
    logger.stage(2, "Preprocessing")

    # TODO: Skip already preprocessed directories

    raw_dir = Path(config['download']['output_dir'])
    processed_dir = Path(config['preprocess']['output_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/project_config.yaml", 'r') as f:
        project_config = yaml.safe_load(f)

    with open(config['preprocess']['hft_transformer']['feature_config_path'], 'r') as f:
        hft_feature_config = json.load(f)

    transcriber = HFT_Transformer(
        config=hft_feature_config,
        model_path=config['preprocess']['hft_transformer']['model_path'],
        verbose=verbose
    )

    song_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    for song_dir in tqdm(song_dirs, desc="[Stage 2] Processing Songs"):
        song_name = song_dir.name
        output_song_dir = processed_dir / song_name
        output_song_dir.mkdir(exist_ok=True)

        # --- Transcription (for cover.wav) ---
        cover_wav = song_dir / "cover.wav"
        transcription_json = output_song_dir / "transcription.json"

        if transcription_json.exists() and verbose:
            logger.skip(f"Transcription for {song_name}: transcription.json already exists.")
        elif not cover_wav.exists():
            logger.warn(f"Skipping transcription for {song_name}: cover.wav not found.")
        else:
            if verbose:
                logger.substep(f"Transcribing cover for {song_name}...")
            transcriber.transcribe(
                input_wav_path=cover_wav,
                output_json_path=transcription_json,
                inference_params=config['preprocess']['hft_transformer']['inference_params']
            )

        # --- Beat Detection (for origin.wav) ---
        origin_wav = song_dir / "origin.wav"
        sep_npy_path = output_song_dir / "sep.npy"
        beat_pred_path = output_song_dir / "beat_pred.json"
        tempo_path = output_song_dir / "tempo.json"

        if tempo_path.exists() and verbose:
            logger.skip(f"Beat/Tempo for {song_name}: tempo.json already exists.")
        elif not origin_wav.exists():
            logger.warn(f"Skipping beat/tempo for {song_name}: origin.wav not found.")
        else:
            if verbose:
                logger.substep(f"Processing beats for {song_name}...")

            separation_backend = project_config['env'].get('separation_backend', 'spleeter')

            if separation_backend == 'demucs':
                # Demucs runs in main environment
                separation_cmd = [
                    sys.executable, "scripts/run_separation.py",
                    "--input", str(origin_wav),
                    "--output", str(sep_npy_path),
                    "--backend", "demucs"
                ]
            else:
                # Spleeter requires separate conda environment
                separation_cmd = [
                    "conda", "run", "-n", project_config['env']['spleeter_env_name'],
                    "python", "scripts/run_separation.py",
                    "--input", str(origin_wav),
                    "--output", str(sep_npy_path),
                    "--backend", "spleeter"
                ]

            subprocess.run(separation_cmd, check=True, capture_output=True)

            # Beat detection now runs in main environment (madmom is installed)
            beat_detection_cmd = [
                sys.executable, "scripts/run_beat_detection.py",
                "--input_npy", str(sep_npy_path),
                "--output_json", str(beat_pred_path),
                "--model_path", config['preprocess']['beat_model_path'],
                "--config_path", config['preprocess']['config_path']
            ]
            subprocess.run(beat_detection_cmd, check=True, capture_output=True)

            beat_analyzer = BeatAnalyzer(verbose=verbose)
            tempo_data = beat_analyzer.analyze(beat_pred_path)
            beat_analyzer.save_tempo_data(tempo_data, tempo_path)

    logger.info("Stage 2: Transcription & Beat Detection complete.")


def run_stage_3_align_and_filter(config: dict, verbose: bool = False):
    """
    Handles Stage 3: Aligns transcriptions, filters based on wp-std,
    and prepares the final synced data.
    """
    logger.stage(3, "Align & Filter")

    raw_dir = Path(config['download']['output_dir'])
    processed_dir = Path(config['preprocess']['output_dir'])
    synced_dir = Path(config['align_and_filter']['output_dir'])
    synced_dir.mkdir(parents=True, exist_ok=True)

    wp_std_threshold = config['align_and_filter']['wp_std_threshold']

    aligner = AudioAligner(verbose=verbose)

    song_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])

    final_metadata = []

    for song_dir in tqdm(song_dirs, desc="[Stage 3] Aligning & Filtering"):
        song_name = song_dir.name

        origin_wav = raw_dir / song_name / "origin.wav"
        cover_wav = raw_dir / song_name / "cover.wav"
        beat_pred_path = song_dir / "beat_pred.json"
        transcription_path = song_dir / "transcription.json"

        final_cover_json = synced_dir / song_name / "cover.json"
        if final_cover_json.exists():
            if verbose:
                logger.skip(f"{song_name}: Already processed and filtered.")
            final_metadata.append({"dir_name": song_name, "status": "kept"})
            continue

        if not all(p.exists() for p in [origin_wav, cover_wav, beat_pred_path, transcription_path]):
            logger.warn(f"Skipping {song_name}: Missing one or more required input files.")
            continue

        if verbose:
            logger.substep(f"Processing {song_name}...")

        align_result = aligner.align(origin_wav, cover_wav, song_dir)
        if not align_result:
            logger.warn("Alignment failed. Skipping.")
            continue

        with open(beat_pred_path, 'r') as f:
            downbeats = json.load(f)['downbeat_pred']
        time_map = create_time_map_from_downbeats(downbeats, align_result)

        wp_std = compute_wp_std(time_map)
        if verbose:
            logger.substep(f"Calculated WP-Std: {wp_std:.4f}")

        if wp_std > wp_std_threshold:
            if verbose:
                logger.substep(f"Filtering out: WP-Std ({wp_std:.4f}) > Threshold ({wp_std_threshold})")
            continue

        if verbose:
            logger.substep("Keeping: WP-Std is within tolerance.")

        with open(transcription_path, 'r') as f:
            transcription_notes = json.load(f)
        aligned_notes = weakly_align(transcription_notes, time_map)

        output_song_dir = synced_dir / song_name
        output_song_dir.mkdir(exist_ok=True)

        with open(final_cover_json, 'w') as f:
            json.dump(aligned_notes, f, indent=4)

        final_metadata.append({"dir_name": song_name, "status": "kept", "wp_std": wp_std})

    metadata_path = synced_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(final_metadata, f, indent=4)

    logger.info(f"Stage 3: Align & Filter complete. Final metadata saved to {metadata_path}")


def run_stage_4_extract(config: dict, verbose: bool = False):
    """
    Handles Stage 4: Extracts notes from the ORIGINAL song (origin.wav) to be used
    as the condition for the decoder model.
    """
    logger.stage(4, "Extracting Condition Notes")

    stage_config = config['extract']
    raw_dir = Path(config['download']['output_dir'])
    output_base_dir = Path(config['align_and_filter']['output_dir'])

    metadata_path = output_base_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}. Cannot proceed. Please run Stage 3 first.")
        sys.exit(1)
    with open(metadata_path, 'r') as f:
        songs_to_process = json.load(f)

    with open(stage_config['config_path'], 'r') as f:
        amt_config = yaml.safe_load(f)
    extractor = AMTAPC_Extractor(config=amt_config, model_path=stage_config['model_path'])

    for song_info in tqdm(songs_to_process, desc="[Stage 4] Extracting"):
        if song_info.get("status") != "kept":
            continue

        song_name = song_info["dir_name"]
        origin_wav_path = raw_dir / song_name / "origin.wav"
        output_json_path = output_base_dir / song_name / "extract.json"

        if output_json_path.exists():
            if verbose:
                logger.skip(f"{song_name}: extract.json already exists.")
            continue

        if not origin_wav_path.exists():
            if verbose:
                logger.warn(f"Skipping {song_name}: origin.wav not found.")
            continue

        if verbose:
            logger.substep(f"Extracting notes from origin.wav for {song_name}...")

        extractor.extract(
            audio_path=str(origin_wav_path),
            output_json_path=str(output_json_path)
        )

    logger.info("Stage 4: Condition note extraction complete.")


def run_stage_5_tokenize(config: dict, verbose: bool = False):
    """
    Handles Stage 5: Tokenizes the filtered data, builds a vocabulary if needed,
    and saves the final sequences for training.
    """
    logger.stage(5, "Tokenizing Final Dataset")

    source_dir = Path(config['align_and_filter']['output_dir'])
    tokenized_dir = Path(config['tokenize']['output_dir'])
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(config['tokenize']['vocab_path'])
    save_format = config['tokenize']['save_format']

    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}. Cannot proceed with tokenization.")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        songs_to_process = json.load(f)

    if vocab_path.exists():
        if verbose:
            logger.info(f"Existing vocabulary found at {vocab_path}. Loading...")
        vocab = Vocab.load(vocab_path)
        needs_vocab_build = False
    else:
        if verbose:
            logger.info("No vocabulary found. It will be built from the dataset.")
        needs_vocab_build = True

    all_src_events, all_tgt_events = [], []
    processed_song_dirs = []

    for song_info in tqdm(songs_to_process, desc="[Stage 5] Tokenizing"):
        if song_info.get("status") != "kept":
            continue

        song_name = song_info["dir_name"]
        current_song_dir = source_dir / song_name

        tempo_path = Path(config['preprocess']['output_dir']) / song_name / "tempo.json"
        src_path = current_song_dir / "extract.json"
        tgt_path = current_song_dir / "cover.json"

        if not all(p.exists() for p in [tempo_path, src_path, tgt_path]):
            if verbose:
                logger.warn(f"Skipping {song_name}: Missing required json files for tokenization.")
            continue

        src_tokenizer = TinyREMITokenizer(tempo_path)
        src_events = src_tokenizer.encode(str(src_path), with_grace_note=True)

        tgt_tokenizer = TinyREMITokenizer(tempo_path)
        tgt_events = tgt_tokenizer.encode(str(tgt_path), with_grace_note=True)

        if src_events and tgt_events:
            all_src_events.append(src_events)
            all_tgt_events.append(tgt_events)
            processed_song_dirs.append(song_name)

    if not processed_song_dirs:
        logger.error("No valid song pairs found to tokenize. Exiting.")
        sys.exit(1)

    if needs_vocab_build:
        if verbose:
            logger.info(f"Building vocabulary from {len(processed_song_dirs)} processed song pairs...")
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        vocab = Vocab(special_tokens=special_tokens)
        vocab.build_from_events(all_src_events + all_tgt_events)
        vocab.save(vocab_path)
        if verbose:
            logger.info(f"Vocabulary with {len(vocab)} tokens saved to {vocab_path}")

    if verbose:
        logger.info(f"Encoding {len(processed_song_dirs)} pairs into integer sequences...")

    for i, song_name in enumerate(tqdm(processed_song_dirs, desc="[Stage 5] Encoding")):
        output_subdir = tokenized_dir / f"{i+1:04d}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        src_output_path = output_subdir / f"{i+1:04d}_src.{save_format}"
        tgt_output_path = output_subdir / f"{i+1:04d}_tgt.{save_format}"

        vocab.encode_and_save_sequence(all_src_events[i], src_output_path, format=save_format)
        vocab.encode_and_save_sequence(all_tgt_events[i], tgt_output_path, format=save_format)

    logger.info(f"Stage 5: Tokenization complete. Final dataset saved to {tokenized_dir.resolve()}")


def main():
    """Main function to orchestrate the data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-end data preparation pipeline for the Etude project."
    )
    parser.add_argument(
        "--config", type=str, default="configs/prepare_config.yaml",
        help="Path to the main data preparation configuration file."
    )
    parser.add_argument(
        "--start-from", type=str, choices=['download', 'preprocess', 'align', 'extract', 'tokenize'],
        default='download', help="The stage to start the pipeline from."
    )
    parser.add_argument(
        "--run-only", type=str, choices=['download', 'preprocess', 'align', 'extract', 'tokenize'],
        help="Run only a single specified stage."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging output for all stages.")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Execute Pipeline Stages ---
    pipeline_stages = ['download', 'preprocess', 'align', 'extract', 'tokenize']
    start_index = pipeline_stages.index(args.start_from)

    for i, stage in enumerate(pipeline_stages):
        if i < start_index:
            continue
        
        if args.run_only and args.run_only != stage:
            continue

        if stage == 'download':
            run_stage_1_download(config, verbose=args.verbose)
        elif stage == 'preprocess':
            run_stage_2_preprocess(config, verbose=args.verbose)
        elif stage == 'align':
            run_stage_3_align_and_filter(config, verbose=args.verbose)
        elif stage == 'extract':
            run_stage_4_extract(config, verbose=args.verbose)
        elif stage == 'tokenize':
            run_stage_5_tokenize(config, verbose=args.verbose)

    if args.verbose:
        logger.success("Data preparation script finished.")

if __name__ == "__main__":
    main()