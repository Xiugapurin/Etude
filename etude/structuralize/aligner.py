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

    def __init__(self, fs: int = 22050, feature_rate: int = 50, verbose: bool = False):
        """
        Initializes the AudioAligner with configuration parameters.

        Args:
            fs (int): The sampling rate for audio processing.
            feature_rate (int): The rate of feature extraction in Hz.
        """
        self.fs = fs
        self.feature_rate = feature_rate
        self.verbose = verbose
        
        # --- Default parameters for MRMSDTW ---
        self.step_weights = np.array([1.5, 1.5, 2.0])
        self.threshold_rec = 10 ** 6
        self.win_len_smooth = np.array([101, 51, 21, 1])

    def align(
        self, 
        origin_audio_path: Union[str, Path], 
        cover_audio_path: Union[str, Path], 
        song_dir: Union[str, Path]
    ) -> Optional[Dict]:
        """
        Aligns two audio files using a "cache-first" strategy with a rich cache format.
        If a valid cache entry is found, it returns the cached data WITHOUT accessing audio files.
        """
        version_key = Path(cover_audio_path).stem
        
        # Step 1: Attempt to load the complete result from the rich cache.
        cached_result = self._load_from_cache(song_dir, version_key)
        if cached_result:
            return cached_result
        
        # Step 2: Cache miss. Fallback to full alignment, which requires .wav files.
        if self.verbose:
            print(f"    > [INFO] No valid cache for '{version_key}'. Attempting alignment from .wav files.")
        
        if not Path(origin_audio_path).exists() or not Path(cover_audio_path).exists():
            return None 

        try:
            origin_audio, _ = librosa.load(str(origin_audio_path), sr=self.fs)
            cover_audio, _ = librosa.load(str(cover_audio_path), sr=self.fs)
        except Exception as e:
            print(f"    > [ERROR] Failed to load audio files for alignment: {e}")
            return None

        # Compute the full result from scratch.
        result = self._compute_warping_path(origin_audio, cover_audio)
        self._save_to_cache(song_dir, version_key, result)
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
        """The core private method for feature extraction and MrMsDTW alignment."""
        tuning_offset_cover = estimate_tuning(cover_audio, self.fs)
        tuning_offset_origin = estimate_tuning(origin_audio, self.fs)
        
        f_chroma_cover, f_dlnco_cover = self._get_features(cover_audio, tuning_offset_cover)
        f_chroma_origin, f_dlnco_origin = self._get_features(origin_audio, tuning_offset_origin)

        f_cens_cover = quantized_chroma_to_CENS(f_chroma_cover, 201, 50, self.feature_rate)[0]
        f_cens_origin = quantized_chroma_to_CENS(f_chroma_origin, 201, 50, self.feature_rate)[0]
        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_cover, f_cens_origin)
        
        f_chroma_origin_shifted = shift_chroma_vectors(f_chroma_origin, opt_chroma_shift)
        f_dlnco_origin_shifted = shift_chroma_vectors(f_dlnco_origin, opt_chroma_shift)
        
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

        pitch_shift = -opt_chroma_shift % 12
        if pitch_shift > 6:
            pitch_shift -= 12

        return {
            "wp": wp.astype(int),
            "pitch_shift": int(pitch_shift),
            "num_frames_cover": f_chroma_cover.shape[1],
            "num_frames_origin": f_chroma_origin.shape[1]
        }

    def _load_from_cache(self, song_dir: Union[str, Path], version_key: str) -> Optional[Dict]:
        """Loads a rich alignment result from the shared 'wp.json' cache file."""
        cache_path = Path(song_dir) / "wp.json"
        if not cache_path.exists(): return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            version_data = all_data.get(version_key)
            # Check if the cached data has the rich format
            if isinstance(version_data, dict) and all(k in version_data for k in ["wp", "num_frames_cover", "num_frames_origin"]):
                version_data['wp'] = np.array(version_data['wp'], dtype=int)
                # Add pitch_shift if it's not in the simple cache
                version_data.setdefault('pitch_shift', 0)
                return version_data
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _save_to_cache(self, song_dir: Union[str, Path], version_key: str, result_data: Dict):
        """Saves a rich alignment result to the shared 'wp.json' cache file."""
        cache_path = Path(song_dir) / "wp.json"
        all_data = {}
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            except json.JSONDecodeError: pass
        
        # Make a copy to avoid modifying the original dict and convert wp to list
        data_to_save = result_data.copy()
        data_to_save['wp'] = result_data['wp'].tolist()
        
        all_data[version_key] = data_to_save
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)
        
        if self.verbose:
            print(f"    > Rich alignment data for '{version_key}' saved to cache: {cache_path}")