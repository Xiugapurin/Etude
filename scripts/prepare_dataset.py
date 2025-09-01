# scripts/prepare_dataset.py

import argparse
import sys
from pathlib import Path
import subprocess

import pandas as pd
import yaml
import json
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import all necessary utilities from our library
from etude.utils.download import download_audio_from_url
from etude.transcription.hft_transformer import HFT_Transcriber
from etude.structuralize.beat_analyzer import BeatAnalyzer
# from etude.structuralize.aligner import AudioAligner # For Stage 2
# from etude.structuralize.tokenizer import MidiTokenizer # For Stage 4

def run_stage_1_download(config: dict):
    """
    Handles Stage 1: Downloading all raw audio files from the source CSV.
    """
    print("\n" + "="*25 + " Stage 1: Downloading Raw Audio " + "="*25)
    
    stage_config = config['download']
    csv_path = Path(stage_config['csv_path'])
    output_dir = Path(stage_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[ERROR] Input CSV file not found at: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded '{csv_path.name}'. Found {len(df)} song pairs to process.")
    except Exception as e:
        print(f"[ERROR] Failed to read or parse CSV file: {e}")
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
    
    print("✅ Stage 1: Download complete.")


def run_stage_2_transcript_and_beat_detect(config: dict):
    """
    Handles Stage 2: Processes raw audio to generate beat/tempo information (for origin)
    and transcription notes (for cover).
    """
    print("\n" + "="*25 + " Stage 2: Transcription & Beat Detection " + "="*25)
    
    raw_dir = Path(config['download']['output_dir'])
    processed_dir = Path(config['processing']['output_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/structuralize_config.yaml", 'r') as f:
        struct_config = yaml.safe_load(f)

    with open("configs/transcription_config.yaml", 'r') as f:
        hft_config = yaml.safe_load(f)
    
    with open(hft_config['feature_config_path'], 'r') as f:
        hft_feature_config = json.load(f)
    
    transcriber = HFT_Transcriber(
        config=hft_feature_config,
        model_path=hft_config['model_path']
    )

    song_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    for song_dir in tqdm(song_dirs, desc="[Stage 2] Processing Songs"):
        song_name = song_dir.name
        output_song_dir = processed_dir / song_name
        output_song_dir.mkdir(exist_ok=True)
        
        # --- Transcription (for cover.wav) ---
        cover_wav = song_dir / "cover.wav"
        transcription_json = output_song_dir / "transcription.json"

        if transcription_json.exists():
            tqdm.write(f"[SKIP] Transcription for {song_name}: transcription.json already exists.")
        elif not cover_wav.exists():
            tqdm.write(f"[WARN] Skipping transcription for {song_name}: cover.wav not found.")
        else:
            tqdm.write(f"\n  > Transcribing cover for {song_name}...")
            transcriber.transcribe(
                input_wav_path=cover_wav,
                output_json_path=transcription_json,
                inference_params=hft_config['inference_params']
            )

        # --- Beat Detection (for origin.wav) ---
        origin_wav = song_dir / "origin.wav"
        sep_npy_path = output_song_dir / "sep.npy"
        beat_pred_path = output_song_dir / "beat_pred.json"
        tempo_path = output_song_dir / "tempo.json"

        if tempo_path.exists():
            tqdm.write(f"[SKIP] Beat/Tempo for {song_name}: tempo.json already exists.")
        elif not origin_wav.exists():
            tqdm.write(f"[WARN] Skipping beat/tempo for {song_name}: origin.wav not found.")
        else:
            tqdm.write(f"\n    > Processing beats for {song_name}...")
            
            spleeter_cmd = [
                "conda", "run", "-n", struct_config['spleeter_env_name'],
                "python", "scripts/run_separation.py",
                "--input", str(origin_wav), "--output", str(sep_npy_path)
            ]
            subprocess.run(spleeter_cmd, check=True, capture_output=True)

            beat_detection_cmd = [
                "conda", "run", "-n", struct_config['madmom_env_name'],
                "python", "scripts/run_beat_detection.py",
                "--input_npy", str(sep_npy_path),
                "--output_json", str(beat_pred_path),
                "--model_path", struct_config['beat_model_path'],
                "--config_path", "configs/structuralize_config.yaml"
            ]
            subprocess.run(beat_detection_cmd, check=True, capture_output=True)
            
            beat_analyzer = BeatAnalyzer()
            tempo_data = beat_analyzer.analyze(beat_pred_path)
            BeatAnalyzer.save_tempo_data(tempo_data, tempo_path)
    
    print("✅ Stage 2: Transcription & Beat Detection complete.")


def run_stage_3_filter(config: dict):
    """
    Handles Stage 3: Filters the processed dataset based on quality metrics.
    """
    print("\n" + "="*25 + " Stage 3: Filtering Dataset " + "="*25)
    
    # TODO: Implement the logic for this stage.
    # This function will iterate through `dataset/synced`.
    # It will read metadata (e.g., from 'wp.json') and apply filtering rules
    # (e.g., keep only songs with WPD score below a threshold).
    # The output could be a new `metadata_filtered.json` file.

    print("🚧 Stage 3: Not yet implemented.")
    pass


def run_stage_4_tokenize(config: dict):
    """
    Handles Stage 4: Tokenizes the filtered data and builds the final vocabulary.
    """
    print("\n" + "="*25 + " Stage 4: Tokenizing Final Dataset " + "="*25)
    
    # TODO: Implement the logic for this stage.
    # This function is essentially the refactored version of the old 'prepare_dataset.py' script.
    # It will:
    # 1. Scan the 'filtered' or 'synced' directory.
    # 2. Use MidiTokenizer to encode 'extract.json' and 'cover.json' for each song.
    # 3. Build a global vocabulary from all events.
    # 4. Save the vocabulary and the final encoded .npy files to `dataset/tokenized`.

    print("🚧 Stage 4: Not yet implemented.")
    pass


def main():
    """Main function to orchestrate the data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-end data preparation pipeline for the Etude project."
    )
    parser.add_argument(
        "--config", type=str, default="configs/data_prepare_config.yaml",
        help="Path to the main data preparation configuration file."
    )
    # Add arguments to control which stages to run
    parser.add_argument(
        "--start-from", type=str, choices=['download', 'process', 'filter', 'tokenize'],
        default='download', help="The stage to start the pipeline from."
    )
    parser.add_argument(
        "--run-only", type=str, choices=['download', 'process', 'filter', 'tokenize'],
        help="Run only a single specified stage."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Execute Pipeline Stages ---
    pipeline_stages = ['download', 'process', 'filter', 'tokenize']
    start_index = pipeline_stages.index(args.start_from)

    for i, stage in enumerate(pipeline_stages):
        if i < start_index:
            continue
        
        if args.run_only and args.run_only != stage:
            continue

        if stage == 'download':
            run_stage_1_download(config)
        elif stage == 'process':
            run_stage_2_transcript_and_beat_detect(config)
        elif stage == 'filter':
            run_stage_3_filter(config)
        elif stage == 'tokenize':
            run_stage_4_tokenize(config)

    print("\n[INFO] Data preparation script finished.")

if __name__ == "__main__":
    main()