# infer.py

import sys
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import torch

from etude.extract.extractor import AMT_Extractor
from etude.structuralize.beat_analyzer import BeatAnalyzer
from etude.structuralize.audio_analyzer import analyze_volume, save_volume_map
from etude.decode.tokenizer import MidiTokenizer
from etude.decode.vocab import Vocab
from etude.utils.model_loader import load_etude_decoder
from etude.utils.download import download_audio_from_url


class InferencePipeline:
    """Orchestrates the entire inference process from audio to final MIDI."""
    def __init__(self, config: dict):
        self.config = config
        self.device = config['general']['device']
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.output_dir = Path(config['general']['output_dir'])
        self.work_dir = self.output_dir / "temp"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[PIPELINE] Device: {self.device}")
        print(f"[PIPELINE] Final outputs will be saved to: {self.output_dir.resolve()}")
        print(f"[PIPELINE] Intermediate files will be stored in: {self.work_dir.resolve()}")

    def _run_command(self, command: list):
        """Helper to run a command and check for errors."""
        try:
            # For clarity, let's capture and print stdout on success
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            if result.stdout:
                print(f"    > Subprocess output:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Error executing command: {' '.join(command)}", file=sys.stderr)
            print(f"  {e.stderr}", file=sys.stderr)
            sys.exit(1)

    def _prepare_audio(self, source: str) -> Path:
        """
        Ensures the source audio is available locally in the working directory.
        Handles both URLs and local file paths.
        """
        print("Downloading Source Audio...")
        local_audio_path = self.work_dir / "origin.wav"

        if urlparse(source).scheme in ('http', 'https'):
            success = download_audio_from_url(source, local_audio_path)
            if not success:
                print("[Fatal] Audio download failed. Exiting.", file=sys.stderr)
                sys.exit(1)
        elif Path(source).is_file():
            print(f"    > Copying local file '{source}' to working directory.")
            shutil.copy(source, local_audio_path)
        else:
            print(f"[Fatal] Input source '{source}' is not a valid URL or local file.", file=sys.stderr)
            sys.exit(1)
        
        return local_audio_path

    def run(self, audio_source: str, target_attributes: dict, final_filename: str):
        """
        Executes the full inference pipeline.
        
        Args:
            audio_source (str): Path or URL to the input audio.
            target_attributes (dict): Dictionary of target attribute bins for generation.
            final_filename (str): The base name for the final output file (without extension).
        """
        audio_path = self._prepare_audio(audio_source)

        print("\n[STAGE 1] Extracting feature notes")

        ext_cfg = self.config['extractor']
        with open(ext_cfg['config_path'], 'r') as f: 
            amt_config = yaml.safe_load(f)
        
        extractor = AMT_Extractor(config=amt_config, model_path=ext_cfg['model_path'], device=self.device)
        extractor.extract(
            audio_path=str(audio_path),
            output_json_path=str(self.work_dir / "extract.json")
        )

        print("    > Analyzing volume contour")
        volume_map = analyze_volume(audio_path)
        save_volume_map(volume_map, self.work_dir / "volume.json")


        print("\n[STAGE 2] Structuralizing tempo information")

        spleeter_cmd = [ "conda", "run", "-n", self.config['preprocessing']['spleeter_env_name'],
                         "python", "scripts/run_separation.py",
                         "--input", str(audio_path), "--output", str(self.work_dir / "sep.npy") ]
        print("    > Running source separation for beat detection")
        self._run_command(spleeter_cmd)

        beat_detection_cmd = [ "conda", "run", "-n", self.config['preprocessing']['madmom_env_name'],
                               "python", "scripts/run_beat_detection.py",
                               "--input_npy", str(self.work_dir / "sep.npy"),
                               "--output_json", str(self.work_dir / "beat_pred.json"),
                               "--model_path", self.config['structuralizer']['beat_model_path'],
                               "--config_path", self.config['structuralizer']['config_path'] ]
        print("    > Running beat detection")
        self._run_command(beat_detection_cmd)

        beat_analyzer = BeatAnalyzer()
        tempo_data = beat_analyzer.analyze(self.work_dir / "beat_pred.json")
        BeatAnalyzer.save_tempo_data(tempo_data, self.work_dir / "tempo.json")


        print("\n[STAGE 3] Decoding with target attributes to generate piano cover...")

        dec_cfg = self.config['decoder']
        model = load_etude_decoder(dec_cfg['config_path'], dec_cfg['model_path'], self.device)
        vocab = Vocab.load(dec_cfg['vocab_path'])
        tokenizer = MidiTokenizer(tempo_path=self.work_dir / "tempo.json")

        condition_events = tokenizer.encode(str(self.work_dir / "extract.json"))
        condition_ids = vocab.encode_sequence(condition_events)

        all_x_bars = tokenizer.split_sequence_into_bars(condition_ids, vocab.get_bar_bos_id(), vocab.get_bar_eos_id())
        num_bars = len(all_x_bars)
        target_attributes_per_bar = [target_attributes] * num_bars
        
        generated_events = model.generate(
            vocab=vocab,
            all_x_bars=all_x_bars,
            target_attributes_per_bar=target_attributes_per_bar,
            temperature=dec_cfg['temperature'], top_p=dec_cfg['top_p']
        )
        
        if generated_events:
            final_notes = tokenizer.decode_to_notes(
                events=generated_events, 
                volume_map_path=self.work_dir / "volume.json"
            )

            final_midi_path = self.output_dir / f"{final_filename}.mid"
            tokenizer.note_to_midi(final_notes, final_midi_path)

            print(f"    > Final MIDI saved to: {final_midi_path.resolve()}")
        else:
            print("    > [Warning] Model generated an empty sequence.")

        print("\n[SUCCESS] Inference pipeline finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="End-to-end piano cover generation pipeline.")
    parser.add_argument("audio_source", type=str, help="Path or URL to the source audio file.")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml", help="Path to the main inference config file.")
    parser.add_argument("--output_name", type=str, default="output", help="Base name for the final output MIDI file (without extension).")
    
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
    pipeline.run(args.audio_source, target_attributes, args.output_name)

if __name__ == "__main__":
    main()