# scripts/run_beat_detection.py

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from etude.models.beat_transformer import Demixed_DilatedTransformerModel
from etude.utils.logger import logger


class BeatDetector:
    """
    A class to handle beat and downbeat detection from pre-processed audio features.
    It encapsulates model loading, inference, and post-processing with madmom.
    """
    def __init__(self, config: dict, model_path: str, device: str = 'auto'):
        """
        Initializes the BeatDetector.

        Args:
            config (dict): A dictionary containing beat detection parameters.
            model_path (str): Path to the pre-trained model checkpoint.
            device (str): The device to run on ('cuda', 'cpu', or 'auto').
        """
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.config = config
        self.model = self._load_model(model_path)
        
        # The feature rate (FPS) is derived from the original audio's sample rate
        # and the STFT hop length used to create the spectrogram.
        fps = 44100 / self.config['fps_divisor']
        
        self.beat_tracker = DBNBeatTrackingProcessor(
            min_bpm=self.config['min_bpm'], max_bpm=self.config['max_bpm'], 
            fps=fps, threshold=self.config['threshold']
        )
        self.downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=self.config['beats_per_bar'],
            min_bpm=self.config['min_bpm'], max_bpm=self.config['max_bpm'],
            fps=fps, threshold=self.config['threshold']
        )
        logger.substep(f"BeatDetector initialized on device: {self.device}")

    def _load_model(self, model_path: str) -> nn.Module:
        """Loads the pre-trained beat detection model."""
        model_params = self.config['model']
        model = Demixed_DilatedTransformerModel(
            attn_len=model_params['attn_len'], instr=model_params['instr'], 
            ntoken=model_params['ntoken'], dmodel=model_params['dmodel'], 
            nhead=model_params['nhead'], d_hid=model_params['d_hid'], 
            nlayers=model_params['nlayers'], norm_first=model_params['norm_first']
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        # The state_dict might be nested inside the checkpoint file
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        model.to(self.device)
        model.eval()
        return model

    def detect(self, input_npy_path: str, output_json_path: str):
        """
        Performs beat and downbeat detection on a .npy file.

        Args:
            input_npy_path (str): Path to the input '.npy' file from source separation.
            output_json_path (str): Path to save the output JSON file.
        """
        input_file = Path(input_npy_path)
        output_file = Path(output_json_path)
        
        try:
            logger.substep(f"Loading features from: {input_file.name}")
            features = np.load(input_file)

            with torch.no_grad():
                model_input = torch.from_numpy(features).unsqueeze(0).float().to(self.device)
                activation, _ = self.model(model_input)

            # Get beat and downbeat activations from the model output
            beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
            downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()

            # Use madmom's DBN trackers to find beat times
            dbn_beat_pred = self.beat_tracker(beat_activation)
            
            # Prepare combined activations for downbeat tracking
            beat_minus_downbeat = np.maximum(beat_activation - downbeat_activation, 0)
            combined_act = np.stack([beat_minus_downbeat, downbeat_activation], axis=-1)
            
            dbn_downbeat_pred_raw = self.downbeat_tracker(combined_act)
            # Filter for events that are actual downbeats (class 1)
            dbn_downbeat_pred = dbn_downbeat_pred_raw[dbn_downbeat_pred_raw[:, 1] == 1][:, 0]

            results = {
                'beat_pred': dbn_beat_pred.tolist(),
                'downbeat_pred': dbn_downbeat_pred.tolist(),
            }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.substep(f"Beat detection successful. Results saved to: {output_file.name}")

            # Important side-effect: remove the input npy file after processing
            input_file.unlink()
            logger.substep(f"Removed temporary file: {input_file.name}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during beat detection: {e}")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Structuralize Stage: Perform beat and downbeat detection from audio features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_npy", required=True, help="Path to the input 'sep.npy' file.")
    parser.add_argument("--output_json", required=True, help="Path to save the output 'beat_pred.json' file.")
    parser.add_argument("--model_path", required=True, help="Path to the beat detection model checkpoint (.pt).")
    parser.add_argument("--config_path", default="configs/structuralize_config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)['beat_detection']

    detector = BeatDetector(config=config, model_path=args.model_path)
    detector.detect(input_npy_path=args.input_npy, output_json_path=args.output_json)

    sys.exit(0)


if __name__ == "__main__":
    main()