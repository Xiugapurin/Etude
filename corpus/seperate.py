import numpy as np
import librosa
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import argparse
import os
import sys
import soundfile as sf


def separate_and_process(input_audio_path, output_npy_path, output_combo_wav_path=None):
    """
    Performs audio separation using Spleeter, processes stems into dB Mel spectrograms,
    and saves the result as a NumPy array.
    Matches the processing logic and final output shape of the provided snippet.
    """
    separator = Separator('spleeter:5stems')
    mel_f = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
    audio_loader = AudioAdapter.default()

    waveform, _ = audio_loader.load(input_audio_path, sample_rate=44100)
    x = separator.separate(waveform)

    if output_combo_wav_path and 'vocals' in x and 'drums' in x:
        vocal_waveform = x['vocals']
        drum_waveform = x['drums']
        combo_waveform = vocal_waveform + drum_waveform
        sf.write(output_combo_wav_path, combo_waveform, 44100)

    x = np.stack([np.dot(np.abs(np.mean(separator._stft(x[key]), axis=-1))**2, mel_f) for key in x])
    x = np.transpose(x, (0, 2, 1))
    x = np.stack([librosa.power_to_db(x[i], ref=np.max) for i in range(len(x))])
    x = np.transpose(x, (0, 2, 1))

    np.save(output_npy_path, x)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_path = os.path.abspath(os.path.join(script_dir, '..', 'infer', 'src', 'origin.wav'))
    default_output_path = os.path.abspath(os.path.join(script_dir, '..', 'infer', 'src', 'sep.npy'))
    default_combo_wav_path = os.path.abspath(os.path.join(script_dir, '..', 'infer', 'src', 'combo.wav'))

    parser = argparse.ArgumentParser(
        description='Separate audio using Spleeter, process stems into dB Mel spectrograms, and save as NPY.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_audio', type=str, default=default_input_path,
                        help='Path to the input audio file.')
    parser.add_argument('--output_npy', type=str, default=default_output_path,
                        help='Path to save the output NumPy array.')
    parser.add_argument('--output_combo_wav', type=str, default=default_combo_wav_path,
                    help='Path to save the output combined (vocals + drums) stem as a WAV file.')

    args = parser.parse_args()

    print("--- Starting Audio Separation and Processing ---")
    print(f"Input Audio:  {args.input_audio}")
    print(f"Output NPY:   {args.output_npy}")
    print(f"Output WAV: {args.output_combo_wav}")
    print("-------------------------------------------------")

    separate_and_process(
        input_audio_path=args.input_audio,
        output_npy_path=args.output_npy,
        output_combo_wav_path=args.output_combo_wav
    )

    print("-------------------------------------------------")
    print("--- Script finished successfully ---")
    sys.exit(0)