import numpy as np
import pretty_midi
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
import warnings
import json

class IPECalculator:
    def __init__(
            self, 
            n_gram: int = 8, 
            n_clusters: int = 8, 
            min_ioi: float = 0.0625, 
            max_ioi: float = 4.0
        ):
        """
        初始化節奏熵計算器。

        Args:
            n_gram (int): N-gram 長度。
            n_clusters (int): k-means 聚類的目標簇數量。
            min_ioi (float): 最小 IOI 間隔。
            max_ioi (float): 最大 IOI 間隔。
        """
        self.n_gram = n_gram
        self.n_clusters = n_clusters
        self.min_ioi = min_ioi
        self.max_ioi = max_ioi

    def _process_raw_ioi(self, ioi_sequence: np.ndarray) -> np.ndarray:
        # processed_ioi = ioi_sequence[ioi_sequence > 1e-9]
        processed_ioi = np.clip(ioi_sequence, self.min_ioi, self.max_ioi)
        
        return processed_ioi

    def _get_ioi_from_file(self, file_path: str) -> np.ndarray:
        """從單一檔案路徑提取並處理 IOI。"""
        onsets = []
        try:
            if file_path.lower().endswith('.mid'):
                midi_data = pretty_midi.PrettyMIDI(file_path)
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        onsets.extend([note.start for note in instrument.notes])
            elif file_path.lower().endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    notes = json.load(f)
                if notes:
                    onsets = [note['onset'] for note in notes]
            
            if len(onsets) < 2: return np.array([])
            
            unique_onsets = np.unique(onsets)
            if len(unique_onsets) < 2: return np.array([])
            
            raw_ioi = np.diff(unique_onsets)
            return self._process_raw_ioi(raw_ioi)
        except Exception:
            return np.array([])

    def _quantize_ioi_to_symbols(self, ioi_sequence: np.ndarray) -> np.ndarray:
        if ioi_sequence.size == 0: return np.array([])
        log_ioi = np.log(ioi_sequence).reshape(-1, 1)
        n_unique_points = len(np.unique(log_ioi))
        actual_n_clusters = min(self.n_clusters, n_unique_points)
        if actual_n_clusters < 2: return np.array([])

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*", category=ConvergenceWarning)
            kmeans.fit(log_ioi)
        return kmeans.labels_
    
    def _get_ngrams_from_sequence(self, sequence: np.ndarray, n: int) -> list:
        if len(sequence) < n: return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

    def _get_shannon_entropy(self, items: list) -> float:
        if not items: return 0.0
        counts = Counter(items)
        total_items = len(items)
        entropy = -np.sum([(c / total_items) * np.log2(c / total_items) for c in counts.values()])
        return entropy
    
    def calculate_ipe(self, file_path: str) -> dict:
        ioi_sequence = self._get_ioi_from_file(file_path)
        if ioi_sequence.size == 0:
            return {"error": "Could not extract a valid IOI sequence."}

        symbol_sequence = self._quantize_ioi_to_symbols(ioi_sequence)
        if symbol_sequence.size == 0:
            return {"error": "Could not quantize IOI sequence."}
        
        ngrams = self._get_ngrams_from_sequence(symbol_sequence, self.n_gram)
        entropy = self._get_shannon_entropy(ngrams)
        
        return {"ipe_score": entropy}