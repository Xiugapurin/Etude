# src/etude/preprocessing/aligner.py

import json
from pathlib import Path
from contextlib import redirect_stdout
import os
from typing import Dict, Optional, Union

import numpy as np
import librosa
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning


class AudioAligner:
    """
    Performs audio alignment between a reference (origin) and a target (cover) audio file.
    
    This class encapsulates the logic for feature extraction and alignment using
    Multi-Resolution Multi-Scale Dynamic Time Warping (MRMSDTW). It provides a simple
    interface for obtaining the warping path and handles caching of results.
    """

    def __init__(self, fs: int = 22050, feature_rate: int = 50):
        """
        Initializes the AudioAligner with configuration parameters.

        Args:
            fs (int): The sampling rate for audio processing.
            feature_rate (int): The rate of feature extraction in Hz.
        """
        self.fs = fs
        self.feature_rate = feature_rate
        # --- Default parameters for MRMSDTW ---
        self.step_weights = np.array([1.5, 1.5, 2.0])
        self.threshold_rec = 10 ** 6
        self.win_len_smooth = np.array([101, 51, 21, 1])

    def align(self, origin_audio_path: Union[str, Path], cover_audio_path: Union[str, Path], cache_path: Optional[Union[str, Path]] = None) -> Optional[Dict]:
        """
        Aligns two audio files to find the optimal warping path and pitch shift.

        This is the main public method. It handles loading/saving from a cache file
        and orchestrates the alignment process.

        Args:
            origin_audio_path (Union[str, Path]): Path to the reference audio file.
            cover_audio_path (Union[str, Path]): Path to the audio file to be aligned.
            cache_path (Optional): Path to a .json file for loading/saving the alignment result.

        Returns:
            A dictionary containing the warping path and other alignment info, e.g.,
            {'wp': np.ndarray, 'pitch_shift': int, ...}, or None if alignment fails.
        """
        if cache_path and Path(cache_path).exists():
            cached_result = self._load_from_cache(cache_path)
            if cached_result:
                print(f"Alignment result successfully loaded from cache: {cache_path}")
                return cached_result
        
        try:
            origin_audio, _ = librosa.load(str(origin_audio_path), sr=self.fs)
            cover_audio, _ = librosa.load(str(cover_audio_path), sr=self.fs)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

        result = self._compute_warping_path(origin_audio, cover_audio)

        if cache_path:
            self._save_to_cache(result, cache_path)
            
        return result

    def _get_features(self, audio: np.ndarray, tuning_offset: float) -> tuple[np.ndarray, np.ndarray]:
        """Extracts chroma and DLNCO features from an audio signal."""
        # Suppress verbose output from the synctoolbox library
        with redirect_stdout(open(os.devnull, "w")):
            f_pitch = audio_to_pitch_features(f_audio=audio, Fs=self.fs, tuning_offset=tuning_offset, feature_rate=self.feature_rate)
            f_chroma = pitch_to_chroma(f_pitch=f_pitch)
            f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

            f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=self.fs, tuning_offset=tuning_offset)
            f_DLNCO = pitch_onset_features_to_DLNCO(
                f_peaks=f_pitch_onset, feature_rate=self.feature_rate,
                feature_sequence_length=f_chroma_quantized.shape[1]
            )

        return f_chroma_quantized, f_DLNCO

    def _compute_warping_path(self, origin_audio: np.ndarray, cover_audio: np.ndarray) -> Dict:
        """The core private method for feature extraction and MRMSDTW alignment."""
        # 1. Estimate tuning for both audio files
        tuning_offset_cover = estimate_tuning(cover_audio, self.fs)
        tuning_offset_origin = estimate_tuning(origin_audio, self.fs)
        
        # 2. Extract features
        f_chroma_cover, f_dlnco_cover = self._get_features(cover_audio, tuning_offset_cover)
        f_chroma_origin, f_dlnco_origin = self._get_features(origin_audio, tuning_offset_origin)

        # 3. Compute optimal chroma shift to handle key differences
        f_cens_cover = quantized_chroma_to_CENS(f_chroma_cover, 201, 50, self.feature_rate)[0]
        f_cens_origin = quantized_chroma_to_CENS(f_chroma_origin, 201, 50, self.feature_rate)[0]
        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_cover, f_cens_origin)
        
        f_chroma_origin_shifted = shift_chroma_vectors(f_chroma_origin, opt_chroma_shift)
        f_dlnco_origin_shifted = shift_chroma_vectors(f_dlnco_origin, opt_chroma_shift)
        
        # 4. Run the MRMSDTW algorithm
        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_cover, f_onset1=f_dlnco_cover,
            f_chroma2=f_chroma_origin_shifted, f_onset2=f_dlnco_origin_shifted,
            input_feature_rate=self.feature_rate,
            step_weights=self.step_weights,
            win_len_smooth=self.win_len_smooth,
            threshold_rec=self.threshold_rec,
            alpha=0.5,
        )
        wp = make_path_strictly_monotonic(wp)

        # Calculate pitch shift in semitones
        pitch_shift = -opt_chroma_shift % 12
        if pitch_shift > 6:
            pitch_shift -= 12

        return {
            "wp": wp.astype(int),
            "pitch_shift": int(pitch_shift),
            "num_frames_cover": f_chroma_cover.shape[1],
            "num_frames_origin": f_chroma_origin.shape[1]
        }

    def _load_from_cache(self, cache_path: Union[str, Path]) -> Optional[Dict]:
        """Loads alignment results from a JSON cache file."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert list back to numpy array
            data['wp'] = np.array(data['wp'], dtype=int)
            return data
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _save_to_cache(self, result_data: Dict, cache_path: Union[str, Path]):
        """Saves alignment results to a JSON cache file."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make a copy to avoid modifying the original dict
        data_to_save = result_data.copy()
        # Convert numpy array to list for JSON serialization
        data_to_save['wp'] = result_data['wp'].tolist()
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Alignment result saved to cache: {path}")