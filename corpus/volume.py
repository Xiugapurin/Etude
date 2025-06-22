import librosa
import numpy as np
import json
from pathlib import Path
import argparse

def analyze_and_save_volume_map(input_wav: str, output_json: str, time_resolution: int = 20):
    """
    Analyzes the volume of a WAV file and saves it as a 1D map.

    Args:
        input_wav (str): Path to the input .wav file.
        output_json (str): Path to save the output .json file.
        time_resolution (int): Number of time steps per second for the analysis.
    """
    input_path = Path(input_wav)
    output_path = Path(output_json)

    if not input_path.exists():
        print(f"Error: Input WAV file not found at {input_path}")
        return

    print(f"Loading audio from: {input_path}")
    y, sr = librosa.load(str(input_path), sr=None)

    # 計算對應時間精度的 hop_length
    hop_length = sr // time_resolution
    
    # 使用 librosa 的 RMS (Root-Mean-Square) 功能來估算每個時間點的能量/音量
    # frame_length 是一個建議值，可以涵蓋足夠的波形來計算 RMS
    frame_length = hop_length * 2 
    
    print(f"Analyzing volume with a time resolution of {time_resolution} Hz...")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 標準化 RMS 值到 [0, 1] 區間，以利後續使用
    if rms.max() > rms.min():
        normalized_rms = (rms - rms.min()) / (rms.max() - rms.min())
    else:
        normalized_rms = np.zeros_like(rms) # 如果音訊為靜音

    volume_map = normalized_rms.tolist()

    # 儲存為 JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(volume_map, f)

    print(f"Successfully created volume map with {len(volume_map)} steps.")
    print(f"Saved to: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the volume contour of a WAV file and save it as a JSON map.")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to the input WAV file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output volume map JSON file.")
    parser.add_argument("--resolution", type=int, default=20, help="Time resolution in Hz (steps per second).")
    
    args = parser.parse_args()
    
    analyze_and_save_volume_map(args.input_wav, args.output_json, args.resolution)
