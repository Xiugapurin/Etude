import os
import json
import numpy as np
from collections import Counter
import pretty_midi

class RGCCalculator:
    def __init__(
            self, 
            top_k: int = 12, 
            precision_digits: int = 4,
            reasonable_bpm_range: tuple = (60, 240),
            base_note_division: int = 4,
            tau_falloff_sigma: float = 0.03,
            lambda_grid_fit: float = 10.0
        ):
        self.top_k = top_k
        self.precision_digits = precision_digits
        self.lambda_grid_fit = lambda_grid_fit
        
        self.min_ideal_tau = (60 / reasonable_bpm_range[1]) / base_note_division
        self.max_ideal_tau = (60 / reasonable_bpm_range[0]) / base_note_division
        self.tau_falloff_sigma = tau_falloff_sigma
        
        print(f"RGC Calculator (Plateau Model) initialized.")
        print(f"Reasonable tau range (for 16th notes): [{self.min_ideal_tau:.4f}s, {self.max_ideal_tau:.4f}s]")


    def _get_raw_ioi(self, file_path: str) -> np.ndarray:
        onsets = []
        try:
            if file_path.lower().endswith('.mid'):
                midi_data = pretty_midi.PrettyMIDI(file_path)
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        onsets.extend([note.start for note in instrument.notes])
            elif file_path.lower().endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f: notes = json.load(f)
                if notes: onsets = [note['onset'] for note in notes]
            
            if len(onsets) < 2: return np.array([])
            unique_onsets = np.unique(onsets)
            if len(unique_onsets) < 2: return np.array([])
            
            raw_ioi = np.diff(unique_onsets)
            return raw_ioi[raw_ioi > 1e-9]
        except Exception:
            return np.array([])

    def calculate_rgc(self, file_path: str, verbose: bool = False) -> dict:
        # ... (步驟 1-3，即推斷出 best_tau 的部分，保持不變) ...
        ioi_sequence = self._get_raw_ioi(file_path)
        if len(ioi_sequence) < self.top_k:
            return {"error": "Not enough IOIs to analyze."}
        rounded_ioi = np.round(ioi_sequence, self.precision_digits)
        ioi_counts = Counter(rounded_ioi)
        if len(ioi_counts) < 2:
            return {"error": "Not enough unique IOIs to determine a grid."}
        top_k_iois = np.array([item for item, count in ioi_counts.most_common(self.top_k)])
        best_tau, lowest_total_deviation = -1, float('inf')
        for tau_candidate in top_k_iois:
            if tau_candidate < 0.01: continue
            total_deviation = np.mean(np.abs((top_k_iois / tau_candidate) - np.round(top_k_iois / tau_candidate)))
            if total_deviation < lowest_total_deviation:
                lowest_total_deviation, best_tau = total_deviation, tau_candidate
        if best_tau == -1:
            return {"error": "Could not infer a valid tau."}
        
        final_tau = best_tau
        avg_grid_deviation = lowest_total_deviation
        grid_fit_score = np.exp(-self.lambda_grid_fit * avg_grid_deviation)
        
        # --- 【關鍵修改】新的 τ 合理性分數計算邏輯 ---
        tau_val = final_tau
        if self.min_ideal_tau <= tau_val <= self.max_ideal_tau:
            # 在理想區間內，直接給滿分
            tau_reasonableness_score = 1.0
        elif tau_val < self.min_ideal_tau:
            # 小於下限，計算與下限的距離
            error = self.min_ideal_tau - tau_val
            tau_reasonableness_score = np.exp(-(error**2) / (2 * self.tau_falloff_sigma**2))
        else: # tau_val > self.max_ideal_tau
            # 大於上限，計算與上限的距離
            error = tau_val - self.max_ideal_tau
            tau_reasonableness_score = np.exp(-(error**2) / (2 * self.tau_falloff_sigma**2))
        
        # ---
        
        rgc_score = grid_fit_score * tau_reasonableness_score

        return {
            "rgc_score": rgc_score, "grid_fit_score": grid_fit_score,
            "tau_reasonableness_score": tau_reasonableness_score,
            "inferred_tau": final_tau, "avg_grid_deviation": avg_grid_deviation
        }