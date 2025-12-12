# scripts/run_separation.py

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def separate_and_extract_features(input_path: str, output_path: str, device: str = "auto"):
    """
    Performs 5-stem source separation using Demucs, converts each stem into a
    dB-scaled Mel spectrogram, and saves the stacked features as a NumPy array.

    Args:
        input_path (str): Path to the source audio file.
        output_path (str): Path to save the resulting feature array as a .npy file.
        device (str): Device to run on ('cuda', 'cpu', 'mps', or 'auto').
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"[ERROR] Input audio file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"    > Using device: {device}")

    try:
        # Load Demucs model (htdemucs: 4 stems - drums, bass, other, vocals)
        # Note: Demucs htdemucs_6s provides 6 stems including piano
        print("    > Initializing Demucs separator (htdemucs_6s)...")
        model = get_model("htdemucs_6s")
        model.to(device)

        # Define Mel filter banks, matching the original script's parameters.
        sample_rate = 44100
        mel_filter_bank = librosa.filters.mel(
            sr=sample_rate, n_fft=4096, n_mels=128, fmin=30, fmax=11000
        ).T

        print(f"    > Loading audio: {input_file.name}")
        waveform, sr = torchaudio.load(str(input_file))

        # Resample if necessary
        if sr != sample_rate:
            print(f"    > Resampling from {sr}Hz to {sample_rate}Hz...")
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Convert mono to stereo if necessary
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(device)

        print("    > Separating audio into stems...")
        with torch.no_grad():
            sources = apply_model(model, waveform, device=device, progress=True)

        # sources shape: (batch, num_sources, channels, samples)
        # htdemucs_6s sources order: drums, bass, other, vocals, guitar, piano
        sources = sources.squeeze(0)  # Remove batch dimension

        # Select 5 stems to match original spleeter output format
        # Spleeter 5stems: vocals, drums, bass, piano, other
        # htdemucs_6s: drums(0), bass(1), other(2), vocals(3), guitar(4), piano(5)
        # We'll use: vocals(3), drums(0), bass(1), piano(5), other(2)
        stem_indices = [3, 0, 1, 5, 2]  # vocals, drums, bass, piano, other
        stem_names = ["vocals", "drums", "bass", "piano", "other"]

        print("    > Converting each stem to a dB Mel Spectrogram...")
        processed_spectrograms = []

        for idx, name in zip(stem_indices, stem_names):
            stem_waveform = sources[idx]  # (channels, samples)
            # Convert to mono by averaging channels
            stem_mono = stem_waveform.mean(dim=0).cpu().numpy()

            # Compute STFT
            stft_result = librosa.stft(stem_mono, n_fft=4096, hop_length=1024)
            power_spec = np.abs(stft_result) ** 2

            # Apply mel filterbank
            mel_spec = np.dot(power_spec.T, mel_filter_bank)
            processed_spectrograms.append(mel_spec)

        stacked_mel_specs = np.stack(processed_spectrograms)
        stacked_mel_specs = np.transpose(stacked_mel_specs, (0, 2, 1))

        db_specs = np.stack([librosa.power_to_db(s, ref=np.max) for s in stacked_mel_specs])
        final_features = np.transpose(db_specs, (0, 2, 1))

        print(f"    > Saving final feature array to {output_file.name}...")
        np.save(output_file, final_features)

        print("    > Feature extraction complete.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during Demucs processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Audio feature extraction via 6-stem separation (Demucs) and Mel spectrogram conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument("--output", required=True, help="Path for the output .npy feature file.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run inference on."
    )
    args = parser.parse_args()

    separate_and_extract_features(args.input, args.output, args.device)


if __name__ == "__main__":
    main()
