# src/etude/preprocessing/audio_analyzer.py

import json
from pathlib import Path
from typing import Union

import librosa
import numpy as np

def analyze_volume(
    audio_path: Union[str, Path],
    sr: int = 22050,
    resolution: int = 20
) -> np.ndarray:
    """
    Analyzes the volume contour of an audio file using Root-Mean-Square (RMS) energy.

    Args:
        audio_path (Union[str, Path]): Path to the input audio file.
        sr (int): The sample rate to use for loading the audio.
        resolution (int): The desired time resolution of the output in Hz (steps per second).

    Returns:
        np.ndarray: A 1D numpy array representing the normalized volume contour [0, 1].
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    y, effective_sr = librosa.load(str(audio_path), sr=sr)
    
    # Calculate hop_length to match the desired time resolution
    hop_length = effective_sr // resolution
    
    # Use a frame length that is twice the hop length for a good RMS calculation window
    frame_length = hop_length * 2
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize the RMS values to a [0, 1] range for consistent use
    if rms.max() > rms.min():
        normalized_rms = (rms - rms.min()) / (rms.max() - rms.min())
    else:
        # Handle the case of a silent audio file
        normalized_rms = np.zeros_like(rms)
        
    return normalized_rms

def save_volume_map(volume_map: np.ndarray, output_path: Union[str, Path]):
    """
    Saves a volume map to a JSON file.

    Args:
        volume_map (np.ndarray): The volume map array to save.
        output_path (Union[str, Path]): The path to the output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(volume_map.tolist(), f)
    print(f"Volume map saved successfully to: {output_path}")