# scripts/run_separation.py

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa


def separate_with_spleeter(input_file: Path, mel_filter_bank: np.ndarray, sample_rate: int):
    """
    Performs 5-stem source separation using Spleeter.

    Args:
        input_file: Path to the input audio file.
        mel_filter_bank: Pre-computed mel filterbank matrix.
        sample_rate: Target sample rate (44100).

    Returns:
        List of mel spectrograms for each stem.
    """
    from spleeter.separator import Separator
    from spleeter.audio.adapter import AudioAdapter

    print("    > Initializing Spleeter separator (5stems)...")
    separator = Separator('spleeter:5stems')

    print(f"    > Loading audio: {input_file.name}")
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(str(input_file), sample_rate=sample_rate)

    print("    > Separating audio into 5 stems...")
    separated_stems = separator.separate(waveform)

    print("    > Converting each stem to a dB Mel Spectrogram...")
    processed_spectrograms = []
    for key in separated_stems:
        stem_waveform = separated_stems[key]
        stft_result = separator._stft(stem_waveform)
        power_spec = np.abs(np.mean(stft_result, axis=-1)) ** 2
        mel_spec = np.dot(power_spec, mel_filter_bank)
        processed_spectrograms.append(mel_spec)

    return processed_spectrograms


def separate_with_demucs(input_file: Path, mel_filter_bank: np.ndarray, sample_rate: int, device: str):
    """
    Performs 6-stem source separation using Demucs (htdemucs_6s).

    Args:
        input_file: Path to the input audio file.
        mel_filter_bank: Pre-computed mel filterbank matrix.
        sample_rate: Target sample rate (44100).
        device: Device to run inference on.

    Returns:
        List of mel spectrograms for each stem (5 stems to match Spleeter output).
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Determine device
    # Note: MPS has limitations with large convolutions (Output channels > 65536)
    # so we fall back to CPU for Demucs on Apple Silicon
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # MPS is not well supported for Demucs due to conv1d limitations
            device = "cpu"
    elif device == "mps":
        print("    > Warning: MPS has limitations with Demucs, falling back to CPU")
        device = "cpu"
    print(f"    > Using device: {device}")

    print("    > Initializing Demucs separator (htdemucs_6s)...")
    model = get_model("htdemucs_6s")
    model.to(device)

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
    # Note: Spleeter's "other" includes guitar, so we need to merge other(2) + guitar(4)
    stem_configs = [
        ("vocals", [3]),       # vocals
        ("drums", [0]),        # drums
        ("bass", [1]),         # bass
        ("piano", [5]),        # piano
        ("other", [2, 4]),     # other + guitar (to match Spleeter's "other")
    ]

    print("    > Converting each stem to a dB Mel Spectrogram...")
    processed_spectrograms = []

    for _, indices in stem_configs:
        # Merge multiple stems if needed (e.g., other + guitar)
        stem_waveform = sum(sources[idx] for idx in indices)  # (channels, samples)
        # Convert to mono by averaging channels
        stem_mono = stem_waveform.mean(dim=0).cpu().numpy()

        # Compute STFT
        stft_result = librosa.stft(stem_mono, n_fft=4096, hop_length=1024)
        power_spec = np.abs(stft_result) ** 2

        # Apply mel filterbank
        mel_spec = np.dot(power_spec.T, mel_filter_bank)
        processed_spectrograms.append(mel_spec)

    return processed_spectrograms


def separate_and_extract_features(input_path: str, output_path: str, backend: str = "spleeter", device: str = "auto"):
    """
    Performs source separation and converts each stem into a dB-scaled Mel spectrogram.

    Args:
        input_path (str): Path to the source audio file.
        output_path (str): Path to save the resulting feature array as a .npy file.
        backend (str): Separation backend to use ('spleeter' or 'demucs').
        device (str): Device to run on (for demucs: 'cuda', 'cpu', 'mps', or 'auto').
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"[ERROR] Input audio file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Define Mel filter banks, matching the original script's parameters.
        sample_rate = 44100
        mel_filter_bank = librosa.filters.mel(
            sr=sample_rate, n_fft=4096, n_mels=128, fmin=30, fmax=11000
        ).T

        # Run separation based on backend
        if backend == "demucs":
            processed_spectrograms = separate_with_demucs(input_file, mel_filter_bank, sample_rate, device)
        else:
            processed_spectrograms = separate_with_spleeter(input_file, mel_filter_bank, sample_rate)

        stacked_mel_specs = np.stack(processed_spectrograms)
        stacked_mel_specs = np.transpose(stacked_mel_specs, (0, 2, 1))

        db_specs = np.stack([librosa.power_to_db(s, ref=np.max) for s in stacked_mel_specs])
        final_features = np.transpose(db_specs, (0, 2, 1))

        print(f"    > Saving final feature array to {output_file.name}...")
        np.save(output_file, final_features)

        print("    > Feature extraction complete.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during {backend} processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Audio feature extraction via source separation and Mel spectrogram conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument("--output", required=True, help="Path for the output .npy feature file.")
    parser.add_argument(
        "--backend",
        default="spleeter",
        choices=["spleeter", "demucs"],
        help="Source separation backend to use."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run inference on (only used with demucs backend)."
    )
    args = parser.parse_args()

    separate_and_extract_features(args.input, args.output, args.backend, args.device)


if __name__ == "__main__":
    main()
