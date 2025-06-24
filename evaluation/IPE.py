import numpy as np
import pretty_midi
from sklearn.cluster import KMeans
from collections import Counter
import warnings
import json
import os

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')
# 我們將透過程式碼邏輯來避免 ConvergenceWarning，但也可以保留
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')


class IPECalculator:
    """
    一個用於計算 IOI 模式熵 (IPE) 的工具類別。
    【v2.0 - 穩健版】
    - 新增了對 IOI 序列的過濾和限制規則。
    - 新增了對 k-means 聚類的動態調整以避免 ConvergenceWarning。
    """
    def __init__(self, n_gram: int = 8, n_clusters: int = 16, 
                 mu_entropy: float = 8.2157, sigma_entropy: float = 0.9775,
                 min_ioi: float = 0.001, max_ioi: float = 4.0):
        """
        初始化 IPE 計算器。

        Args:
            n_gram (int): N-gram 長度。
            n_clusters (int): k-means 聚類的目標簇數量。
            mu_entropy (float): 理想熵值。
            sigma_entropy (float): 熵值容忍度。
            min_ioi (float): 被視為同時發生的最小 IOI 間隔 (秒)。
            max_ioi (float): IOI 間隔的最大值上限 (秒)。
        """
        self.n_gram = n_gram
        self.n_clusters = n_clusters
        self.mu_entropy = mu_entropy
        self.sigma_entropy = sigma_entropy
        self.min_ioi = min_ioi
        self.max_ioi = max_ioi
        print(f"IPE Calculator initialized with: n_gram={n_gram}, n_clusters={n_clusters}, μ_Hn={mu_entropy}, σ_c={sigma_entropy}")
        print("NEW!")

    def _process_raw_ioi(self, ioi_sequence: np.ndarray) -> np.ndarray:
        """
        【新】對原始 IOI 序列應用過濾和限制規則。
        """
        # 規則 1 & 2: 將小於 min_ioi 的間隔視為 0，然後過濾掉所有為 0 的間隔
        processed_ioi = ioi_sequence[ioi_sequence >= self.min_ioi]
        
        # 規則 3: 將大於 max_ioi 的間隔值設為 max_ioi
        processed_ioi[processed_ioi > self.max_ioi] = self.max_ioi
        
        return processed_ioi

    def get_ioi_from_json(self, json_path: str) -> np.ndarray:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                notes = json.load(f)
            if not notes: return np.array([])
            onsets = [note['onset'] for note in notes]
            unique_onsets = np.unique(onsets)
            unique_onsets.sort()
            if len(unique_onsets) < 2: return np.array([])
            
            raw_ioi = np.diff(unique_onsets)
            # 【修改】呼叫新的處理函式
            return self._process_raw_ioi(raw_ioi)
        except Exception as e:
            print(f"Error processing JSON file {json_path}: {e}")
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
            
            raw_ioi = np.diff(unique_onsets)
            # 【修改】呼叫新的處理函式
            return self._process_raw_ioi(raw_ioi)
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return np.array([])

    def quantize_ioi_to_symbols(self, ioi_sequence: np.ndarray) -> np.ndarray:
        """
        【已修改】動態調整 k-means 的 n_clusters 以避免 ConvergenceWarning。
        """
        if ioi_sequence.size == 0: return np.array([])

        log_ioi = np.log(ioi_sequence).reshape(-1, 1)
        
        # --- 【關鍵修改】動態調整 n_clusters ---
        n_unique_points = len(np.unique(log_ioi))
        
        # 如果獨特的點少於目標簇數，則使用較小的值
        actual_n_clusters = min(self.n_clusters, n_unique_points)
        
        # 如果聚類數小於2，則無法進行聚類
        if actual_n_clusters < 2:
            # print("Warning: Not enough unique IOI values to perform clustering.")
            return np.array([])
        # ---

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        kmeans.fit(log_ioi)
        
        return kmeans.labels_
    
    # calculate_ipe 和其他方法保持不變，因為它們的輸入已經被上游方法處理好了
    def calculate_ipe(self, file_path: str) -> dict:
        # ... 此方法邏輯不變 ...
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

    def get_ngrams_from_sequence(self, sequence: np.ndarray, n: int) -> list:
        # ... 此方法邏輯不變 ...
        if len(sequence) < n: return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

    def get_shannon_entropy(self, items: list) -> float:
        # ... 此方法邏輯不變 ...
        if not items: return 0.0
        counts = Counter(items)
        total_items = len(items)
        entropy = -np.sum([(c / total_items) * np.log2(c / total_items) for c in counts.values()])
        return entropy