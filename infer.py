# infer.py

import sys
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import torch

from etude.data.extractor import AMTAPC_Extractor
from etude.data.beat_detector import BeatDetector
from etude.data.beat_analyzer import BeatAnalyzer
from etude.data.tokenizer import TinyREMITokenizer
from etude.data.vocab import Vocab
from etude.utils.preprocess import analyze_volume, save_volume_map
from etude.utils.model_loader import load_etude_decoder
from etude.utils.download import download_audio_from_url
from etude.utils.logger import logger


class InferencePipeline:
    """Orchestrates the entire inference process from audio to final MIDI."""
    def __init__(self, config: dict):
        self.config = config
        with open("configs/project_config.yaml", 'r') as f:
            self.project_config = yaml.safe_load(f)

        self.device = config['general']['device']
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.output_dir = Path(config['general']['output_dir'])
        self.work_dir = self.output_dir / "temp"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir.resolve()}")
        logger.info(f"Working directory: {self.work_dir.resolve()}")

    def _run_command(self, command: list):
        """Helper to run a command and check for errors."""
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing command: {' '.join(command)}")
            logger.substep(e.stderr.strip())
            sys.exit(1)

    def _prepare_audio(self, source: str) -> Path:
        """Ensures the source audio is available locally in the working directory."""
        logger.step("Preparing source audio")
        local_audio_path = self.work_dir / "origin.wav"

        if urlparse(source).scheme in ('http', 'https'):
            logger.substep("Downloading from URL...")
            success = download_audio_from_url(source, local_audio_path)
            if not success:
                logger.error("Audio download failed.")
                sys.exit(1)
        elif Path(source).is_file():
            logger.substep("Copying local file...")
            shutil.copy(source, local_audio_path)
        else:
            logger.error(f"Input source '{source}' is not a valid URL or local file.")
            sys.exit(1)

        logger.info(f"Audio prepared at: {local_audio_path}")
        return local_audio_path

    def _run_stage1_extract(self, audio_path: Path):
        """Runs the AMT extraction process."""
        logger.stage(1, "Extracting feature notes")
        extract_config = self.config['extract']
        with open(extract_config['config_path'], 'r') as f:
            amtapc_config = yaml.safe_load(f)

        logger.step("Running AMT feature extraction")
        extractor = AMTAPC_Extractor(config=amtapc_config, model_path=extract_config['model_path'], device=self.device)
        logger.substep("Extracting notes from audio...")
        extractor.extract(
            audio_path=str(audio_path),
            output_json_path=str(self.work_dir / "extract.json")
        )
        logger.info("Feature extraction complete.")

        logger.step("Analyzing volume contour")
        volume_map_output_path = self.work_dir / "volume.json"
        logger.substep("Computing volume map...")
        volume_map = analyze_volume(audio_path)
        save_volume_map(volume_map, volume_map_output_path)
        logger.info(f"Volume map saved to: {volume_map_output_path}")

    def _run_stage2_structuralize(self, audio_path: Path):
        """Runs the beat detection and tempo analysis process."""
        logger.stage(2, "Structuralizing tempo information")

        separation_backend = self.project_config['env'].get('separation_backend', 'spleeter')

        logger.step("Running source separation")
        if separation_backend == 'demucs':
            logger.substep("Using Demucs backend...")
            separation_cmd = [
                sys.executable, "scripts/run_separation.py",
                "--input", str(audio_path),
                "--output", str(self.work_dir / "sep.npy"),
                "--backend", "demucs"
            ]
        else:
            logger.substep("Using Spleeter backend...")
            separation_cmd = [
                "conda", "run", "-n", self.project_config['env']['spleeter_env_name'],
                "python", "scripts/run_separation.py",
                "--input", str(audio_path),
                "--output", str(self.work_dir / "sep.npy"),
                "--backend", "spleeter"
            ]
        self._run_command(separation_cmd)
        logger.info("Source separation complete.")

        logger.step("Running beat detection")
        logger.substep("Detecting beats and downbeats...")
        with open(self.config['structuralize']['config_path'], 'r') as f:
            beat_config = yaml.safe_load(f)['beat_detection']
        beat_detector = BeatDetector(
            config=beat_config,
            model_path=self.config['structuralize']['beat_model_path'],
            device=self.device
        )
        beat_detector.detect(
            input_npy_path=self.work_dir / "sep.npy",
            output_json_path=self.work_dir / "beat_pred.json",
            cleanup_input=True
        )
        logger.info("Beat detection complete.")

        logger.step("Generating tempo structure")
        logger.substep("Analyzing beat patterns...")
        beat_analyzer = BeatAnalyzer()
        tempo_data = beat_analyzer.analyze(self.work_dir / "beat_pred.json")
        beat_analyzer.save_tempo_data(tempo_data, self.work_dir / "tempo.json")
        logger.info("Tempo structure generated.")

    def _run_stage3_decode(self, target_attributes: dict, final_filename: str):
        """Runs the final music generation based on the intermediate files."""
        logger.stage(3, "Decoding with target attributes")

        decode_config = self.config['decoder']
        model = load_etude_decoder(decode_config['config_path'], decode_config['model_path'], self.device)
        
        vocab = Vocab.load(decode_config['vocab_path'])
        logger.info("Model and vocabulary loaded.")

        logger.step("Preparing input sequence")
        logger.substep("Tokenizing condition events...")
        tokenizer = TinyREMITokenizer(tempo_path=self.work_dir / "tempo.json")
        condition_events = tokenizer.encode(str(self.work_dir / "extract.json"))
        condition_ids = vocab.encode_sequence(condition_events)
        all_x_bars = tokenizer.split_sequence_into_bars(condition_ids, vocab.get_bar_bos_id(), vocab.get_bar_eos_id())
        num_bars = len(all_x_bars)
        target_attributes_per_bar = [target_attributes] * num_bars
        logger.info(f"Prepared {num_bars} bars for generation.")

        logger.step("Generating piano cover")
        logger.substep(f"Generating {num_bars} bars...")
        generated_events = model.generate(
            vocab=vocab,
            all_x_bars=all_x_bars,
            target_attributes_per_bar=target_attributes_per_bar,
            temperature=decode_config['temperature'],
            top_p=decode_config['top_p']
        )

        if generated_events:
            logger.step("Exporting to MIDI")
            logger.substep("Decoding events to notes...")
            final_notes = tokenizer.decode_to_notes(
                events=generated_events,
                volume_map_path=self.work_dir / "volume.json"
            )
            final_midi_path = self.output_dir / f"{final_filename}.mid"
            tokenizer.note_to_midi(final_notes, final_midi_path)
            logger.info(f"Final MIDI saved to: {final_midi_path.resolve()}")
        else:
            logger.warn("Model generated an empty sequence.")

    def run(self, audio_source: str, target_attributes: dict, final_filename: str, decode_only: bool = False):
        """Executes the inference pipeline, conditionally skipping preprocessing."""
        if not decode_only:
            # --- Full pipeline ---
            audio_path = self._prepare_audio(audio_source)
            self._run_stage1_extract(audio_path)
            self._run_stage2_structuralize(audio_path)
        else:
            # --- Decode-only shortcut ---
            logger.skip("Skipping Stages 1 & 2 (decode-only mode).")
            logger.step("Verifying intermediate files")
            required_files = ["extract.json", "tempo.json", "volume.json"]
            for f_name in required_files:
                if not (self.work_dir / f_name).exists():
                    logger.error(f"Missing required file '{f_name}' in {self.work_dir}")
                    logger.substep("Run the full pipeline once to generate these files.")
                    sys.exit(1)
            logger.info("All required intermediate files found.")

        # The decode stage runs in both cases
        self._run_stage3_decode(target_attributes, final_filename)
        logger.success("Inference pipeline finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="End-to-end piano cover generation pipeline.")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml", help="Path to the main inference config file.")
    parser.add_argument("--output_name", type=str, default="output", help="Base name for the final output MIDI file (without extension).")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=str, help="Path or URL to the source audio file for a full run.")
    source_group.add_argument("--decode-only", action="store_true", help="Skip preprocessing and run decode stage only, using files in 'outputs/inference/temp'.")
    
    attr_group = parser.add_argument_group('Target Attribute Controls')
    attr_group.add_argument("--polyphony", type=int, default=1, choices=[0, 1, 2], help="Target bin for Relative Polyphony.")
    attr_group.add_argument("--rhythm", type=int, default=1, choices=[0, 1, 2], help="Target bin for Relative Rhythmic Intensity.")
    attr_group.add_argument("--sustain", type=int, default=1, choices=[0, 1, 2], help="Target bin for Relative Note Sustain.")
    attr_group.add_argument(
        "--overlap", 
        type=int, 
        default=2,
        choices=[0, 1, 2], 
        help="Target bin for Pitch Overlap Ratio. [WARNING: For best quality, this should be kept at 2.]"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    target_attributes = {
        "polyphony_bin": args.polyphony, 
        "rhythm_intensity_bin": args.rhythm,
        "sustain_bin": args.sustain, 
        "pitch_overlap_bin": args.overlap
    }
        
    pipeline = InferencePipeline(config)
    pipeline.run(
        audio_source=args.input, 
        target_attributes=target_attributes, 
        final_filename=args.output_name,
        decode_only=args.decode_only
    )

if __name__ == "__main__":
    main()