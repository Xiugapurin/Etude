# etude/evaluation/metrics/ipe.py

import warnings
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from .base_metric import get_onsets_from_file

class IPECalculator:
    """
    Calculates the Inter-Onset Interval (IOI) Pattern Entropy (IPE).

    IPE measures the complexity and predictability of rhythmic patterns in a piece.
    It involves quantizing log-scale IOIs into a finite set of symbols using
    K-Means clustering and then calculating the Shannon entropy of N-grams
    of these symbols. A higher IPE score suggests a more complex and less
    predictable rhythmic structure.
    """
    def __init__(
            self,
            n_gram: int = 8,
            n_clusters: int = 8,
            min_ioi: float = 0.0625,
            max_ioi: float = 4.0,
            **kwargs
        ):
        """
        Initializes the IPE calculator.

        Args:
            n_gram (int): The length of N-grams for entropy calculation.
            n_clusters (int): The number of clusters for K-Means quantization of IOIs.
            min_ioi (float): The minimum IOI to consider, in seconds.
            max_ioi (float): The maximum IOI to consider, in seconds.
        """
        self.n_gram = n_gram
        self.n_clusters = n_clusters
        self.min_ioi = min_ioi
        self.max_ioi = max_ioi

    def _process_raw_ioi(self, ioi_sequence: np.ndarray) -> np.ndarray:
        """Clips the raw IOI sequence to a valid range."""
        return np.clip(ioi_sequence, self.min_ioi, self.max_ioi)

    def _quantize_ioi_to_symbols(self, ioi_sequence: np.ndarray) -> np.ndarray:
        """Converts a continuous IOI sequence into a discrete symbol sequence using K-Means."""
        if ioi_sequence.size == 0:
            return np.array([])
        
        # Reshape log-IOIs for clustering
        log_ioi = np.log(ioi_sequence).reshape(-1, 1)
        
        # Ensure n_clusters is not greater than the number of unique points
        n_unique_points = len(np.unique(log_ioi))
        actual_n_clusters = min(self.n_clusters, n_unique_points)
        
        if actual_n_clusters < 2:
            return np.array([])

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            kmeans.fit(log_ioi)
        return kmeans.labels_
    
    def _get_ngrams_from_sequence(self, sequence: np.ndarray) -> list:
        """Extracts N-grams from a sequence of symbols."""
        if len(sequence) < self.n_gram:
            return []
        return [tuple(sequence[i:i+self.n_gram]) for i in range(len(sequence) - self.n_gram + 1)]

    def _get_shannon_entropy(self, items: list) -> float:
        """Calculates the Shannon entropy for a list of items (N-grams)."""
        if not items:
            return 0.0
        
        counts = Counter(items)
        total_items = len(items)
        entropy = -np.sum([(c / total_items) * np.log2(c / total_items) for c in counts.values()])
        return entropy
    
    def calculate(self, file_path) -> dict:
        """
        Executes the full IPE calculation pipeline for a given music file.

        Args:
            file_path (Path): Path to the MIDI or JSON file to be analyzed.

        Returns:
            dict: A dictionary containing the 'ipe_score' or an 'error' message.
        """
        unique_onsets = get_onsets_from_file(file_path)
        if len(unique_onsets) < 2:
            return {"error": "Not enough onsets for IOI calculation."}

        raw_ioi = np.diff(unique_onsets)
        ioi_sequence = self._process_raw_ioi(raw_ioi)
        if ioi_sequence.size == 0:
            return {"error": "Could not extract a valid IOI sequence after processing."}

        symbol_sequence = self._quantize_ioi_to_symbols(ioi_sequence)
        if symbol_sequence.size == 0:
            return {"error": "Could not quantize IOI sequence into symbols."}
        
        ngrams = self._get_ngrams_from_sequence(symbol_sequence)
        entropy = self._get_shannon_entropy(ngrams)
        
        return {"ipe_score": entropy}