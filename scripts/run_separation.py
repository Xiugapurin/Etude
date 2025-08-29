# scripts/preprocessing/run_separation.py

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

def separate_and_extract_features(input_path: str, output_path: str):
    """
    Performs 5-stem source separation using Spleeter, converts each stem into a
    dB-scaled Mel spectrogram, and saves the stacked features as a NumPy array.

    This script is designed to be called from a dedicated Conda environment
    where Spleeter, Librosa, and their dependencies are installed.

    Args:
        input_path (str): Path to the source audio file.
        output_path (str): Path to save the resulting feature array as a .npy file.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"[Error] Input audio file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # --- Spleeter & Librosa Core Logic ---
        # 1. Initialize the 5-stem separator.
        print("           > Initializing Spleeter separator (5stems)...")
        separator = Separator('spleeter:5stems')
        
        # Define Mel filter banks, matching the original script's parameters.
        mel_filter_bank = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
        
        # 2. Load audio at the required sample rate.
        print(f"           > Loading audio: {input_file.name}")
        audio_loader = AudioAdapter.default()
        sample_rate = 44100
        waveform, _ = audio_loader.load(str(input_file), sample_rate=sample_rate)

        # 3. Perform the 5-stem separation.
        print("           > Separating audio into 5 stems...")
        separated_stems = separator.separate(waveform)

        # 4. Process each stem into a dB Mel spectrogram.
        print("           > Converting each stem to a dB Mel Spectrogram...")
        processed_spectrograms = []
        for key in separated_stems:
            stem_waveform = separated_stems[key]
            
            # Use Spleeter's internal STFT function (as in the original script)
            stft_result = separator._stft(stem_waveform)
            
            # Average stereo channels if present and compute power spectrogram
            power_spec = np.abs(np.mean(stft_result, axis=-1))**2
            
            # Project onto Mel filter bank
            mel_spec = np.dot(power_spec, mel_filter_bank)
            
            processed_spectrograms.append(mel_spec)

        # Stack all spectrograms into a single array: (stems, mel_bins, frames)
        stacked_mel_specs = np.stack(processed_spectrograms)
        
        # Transpose to (stems, frames, mel_bins)
        stacked_mel_specs = np.transpose(stacked_mel_specs, (0, 2, 1))

        # 5. Convert power to dB scale.
        db_specs = np.stack([librosa.power_to_db(s, ref=np.max) for s in stacked_mel_specs])
        
        # Transpose again to final shape: (stems, mel_bins, frames)
        final_features = np.transpose(db_specs, (0, 2, 1))

        # 6. Save the final feature array.
        print(f"           > Saving final feature array to {output_file.name}...")
        np.save(output_file, final_features)
        
        print("           > Feature extraction complete.")

    except Exception as e:
        print(f"[Error] An unexpected error occurred during Spleeter processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Audio feature extraction via 5-stem separation and Mel spectrogram conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument("--output", required=True, help="Path for the output .npy feature file.")
    args = parser.parse_args()
    
    separate_and_extract_features(args.input, args.output)

if __name__ == "__main__":
    main()