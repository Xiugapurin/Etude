# etude/data/beat_detector.py

"""
Beat and downbeat detection module.

This module provides the BeatDetector class for detecting beats and downbeats
from pre-processed audio features using a trained transformer model and
madmom's DBN post-processing.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from ..config.schema import BeatDetectorConfig
from ..models.beat_transformer import Demixed_DilatedTransformerModel
from ..utils.logger import logger


class BeatDetector:
    """
    A class to handle beat and downbeat detection from pre-processed audio features.
    It encapsulates model loading, inference, and post-processing with madmom.
    """

    def __init__(
        self,
        config: BeatDetectorConfig,
        model_path: Union[str, Path],
        device: str = "auto",
    ):
        """
        Initializes the BeatDetector.

        Args:
            config: BeatDetectorConfig containing beat detection parameters.
            model_path: Path to the pre-trained model checkpoint.
            device: The device to run on ('cuda', 'mps', 'cpu', or 'auto').
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.config = config
        self.model = self._load_model(model_path)

        # The feature rate (FPS) is derived from the original audio's sample rate
        # and the STFT hop length used to create the spectrogram.
        fps = 44100 / self.config.fps_divisor

        self.beat_tracker = DBNBeatTrackingProcessor(
            min_bpm=self.config.min_bpm,
            max_bpm=self.config.max_bpm,
            fps=fps,
            threshold=self.config.threshold,
        )
        self.downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=self.config.beats_per_bar,
            min_bpm=self.config.min_bpm,
            max_bpm=self.config.max_bpm,
            fps=fps,
            threshold=self.config.threshold,
        )
        logger.debug(f"BeatDetector initialized on device: {self.device}")

    def _load_model(self, model_path: Union[str, Path]) -> nn.Module:
        """Loads the pre-trained beat detection model."""
        model_cfg = self.config.model
        model = Demixed_DilatedTransformerModel(
            attn_len=model_cfg.attn_len,
            instr=model_cfg.instr,
            ntoken=model_cfg.ntoken,
            dmodel=model_cfg.dmodel,
            nhead=model_cfg.nhead,
            d_hid=model_cfg.d_hid,
            nlayers=model_cfg.nlayers,
            norm_first=model_cfg.norm_first,
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        # The state_dict might be nested inside the checkpoint file
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        model.to(self.device)
        model.eval()
        return model

    def detect(
        self,
        input_npy_path: Union[str, Path],
        output_json_path: Optional[Union[str, Path]] = None,
        cleanup_input: bool = True
    ) -> Dict:
        """
        Performs beat and downbeat detection on a .npy file.

        Args:
            input_npy_path: Path to the input '.npy' file from source separation.
            output_json_path: Optional path to save the output JSON file.
            cleanup_input: If True, removes the input .npy file after processing.

        Returns:
            A dictionary containing 'beat_pred' and 'downbeat_pred' lists.
        """
        input_file = Path(input_npy_path)

        logger.debug(f"Loading features from: {input_file.name}")
        features = np.load(input_file)

        with torch.no_grad():
            model_input = torch.from_numpy(features).unsqueeze(0).float().to(self.device)
            activation, _ = self.model(model_input)

        # Get beat and downbeat activations from the model output
        beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
        downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()

        # Use madmom's DBN trackers to find beat times
        # Suppress numpy RuntimeWarnings (e.g., divide by zero in log) unless LOG_LEVEL is DEBUG
        with warnings.catch_warnings():
            if not logger.is_debug():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

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

        # Save to file if path provided
        if output_json_path:
            output_file = Path(output_json_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.debug(f"Beat detection results saved to: {output_file.name}")

        # Clean up the temporary input file
        if cleanup_input and input_file.exists():
            input_file.unlink()
            logger.debug(f"Removed temporary file: {input_file.name}")

        return results
