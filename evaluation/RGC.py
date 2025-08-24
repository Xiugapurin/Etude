import numpy as np
import pretty_midi
from collections import Counter
import json
import os

class RGCCalculator:
    def __init__(
        self, 
        top_k: int = 8, 
        precision_digits: int = 4
    ):
        """
        初始化 RGC 計算器。

        Args:
            top_k (int): 選取最常見的 k 個 IOI 進行分析。
            precision_digits (int): 對 IOI 進行四捨五入的精度。
        """
        self.top_k = top_k
        self.precision_digits = precision_digits

    def _get_raw_ioi(self, file_path: str) -> np.ndarray:
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
            return raw_ioi[raw_ioi > 1e-9]
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            return np.array([])

    def calculate_rgc(self, file_path: str) -> dict:
        """
        執行完整的 RGC 計算流程。
        """
        ioi_sequence = self._get_raw_ioi(file_path)
        if len(ioi_sequence) < self.top_k:
            return {"error": "Not enough IOIs to analyze."}

        rounded_ioi = np.round(ioi_sequence, self.precision_digits)
        ioi_counts = Counter(rounded_ioi)
        
        if len(ioi_counts) < 2:
            return {"error": "Not enough unique IOIs to determine a grid."}

        top_k_iois = np.array([item for item, count in ioi_counts.most_common(self.top_k)])
        
        best_tau = -1
        lowest_total_deviation = float('inf')

        for tau_candidate in top_k_iois:
            if tau_candidate < 0.01: continue
            ratios = top_k_iois / tau_candidate
            remainders = ratios - np.round(ratios)
            total_deviation = np.mean(np.abs(remainders))
            
            if total_deviation < lowest_total_deviation:
                lowest_total_deviation = total_deviation
                best_tau = tau_candidate
        
        if best_tau == -1:
            return {"error": "Could not infer a valid tau."}
        
        return {
            "rgc_score": lowest_total_deviation,
            "inferred_tau": best_tau
        }