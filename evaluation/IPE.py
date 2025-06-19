import numpy as np
import pretty_midi
from sklearn.cluster import KMeans
from collections import Counter
import warnings
import json
import os

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')

class IPECalculator:
    """
    一個用於計算 IOI 模式熵 (IPE) 的工具類別。
    現已支援 .mid 和 .json 兩種輸入格式。
    """
    def __init__(self, n_gram: int = 8, n_clusters: int = 16, 
                 mu_entropy: float = 4.5, sigma_entropy: float = 0.5):
        self.n_gram = n_gram
        self.n_clusters = n_clusters
        self.mu_entropy = mu_entropy
        self.sigma_entropy = sigma_entropy

    def get_ioi_from_json(self, json_path: str) -> np.ndarray:
        """
        從給定的 JSON 檔案中提取並計算 IOI 序列。

        Args:
            json_path (str): JSON 檔案的路徑。

        Returns:
            np.ndarray: IOI 時間間隔序列 (單位：秒)。
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                notes = json.load(f)
            
            if not notes:
                return np.array([])
            
            # 提取所有音符的起始時間
            onsets = [note['onset'] for note in notes]
            
            # 後續邏輯與 get_ioi_from_midi 完全相同
            unique_onsets = np.unique(onsets)
            unique_onsets.sort()

            if len(unique_onsets) < 2:
                return np.array([])
            
            ioi_sequence = np.diff(unique_onsets)
            return ioi_sequence

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing JSON file {json_path}: Invalid format. {e}")
            return np.array([])
        except Exception as e:
            print(f"An unexpected error occurred with file {json_path}: {e}")
            return np.array([])

    def get_ioi_sequence(self, midi_path: str) -> np.ndarray:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            onsets = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    onsets.extend([note.start for note in instrument.notes])
            if not onsets: return np.array([])
            unique_onsets = np.unique(onsets)
            unique_onsets.sort()
            if len(unique_onsets) < 2: return np.array([])
            return np.diff(unique_onsets)
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return np.array([])


    def calculate_ipe(self, file_path: str) -> dict:
        """
        執行完整的 IPE 計算流程。會根據副檔名自動選擇解析方法。

        Args:
            file_path (str): 要分析的 MIDI 或 JSON 檔案路徑。

        Returns:
            dict: 一個包含計算結果的字典。
        """
        # 步驟 1: 根據副檔名獲取 IOI 序列
        if file_path.lower().endswith('.mid'):
            ioi_sequence = self.get_ioi_sequence(file_path)
        elif file_path.lower().endswith('.json'):
            ioi_sequence = self.get_ioi_from_json(file_path)
        else:
            return {"error": f"Unsupported file format: {os.path.basename(file_path)}"}

        if ioi_sequence.size == 0:
            return {"error": "Could not extract a valid IOI sequence."}

        symbol_sequence = self.quantize_ioi_to_symbols(ioi_sequence)
        if symbol_sequence.size == 0:
            return {"error": "Could not quantize IOI sequence."}
        ngrams = self.get_ngrams_from_sequence(symbol_sequence, self.n_gram)
        entropy = self.get_shannon_entropy(ngrams)
        numerator = (entropy - self.mu_entropy)**2
        denominator = 2 * self.sigma_entropy**2
        ipe_score = np.exp(-numerator / denominator)

        return {
            "ipe_score": ipe_score,
            "shannon_entropy": entropy,
            "n_gram_count": len(ngrams),
            "unique_n_gram_count": len(set(ngrams)),
            "ioi_sequence_length": len(ioi_sequence),
            "symbol_sequence_length": len(symbol_sequence)
        }
    
    def quantize_ioi_to_symbols(self, ioi_sequence: np.ndarray) -> np.ndarray:
        valid_ioi = ioi_sequence[ioi_sequence > 1e-6]
        if len(valid_ioi) < self.n_clusters: return np.array([])
        log_ioi = np.log(valid_ioi).reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(log_ioi)
        return kmeans.labels_

    def get_ngrams_from_sequence(self, sequence: np.ndarray, n: int) -> list:
        if len(sequence) < n: return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

    def get_shannon_entropy(self, items: list) -> float:
        if not items: return 0.0
        counts = Counter(items)
        total_items = len(items)
        entropy = -np.sum([(c / total_items) * np.log2(c / total_items) for c in counts.values()])
        return entropy