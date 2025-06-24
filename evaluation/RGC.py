import os
import json
import numpy as np
from collections import Counter
import pretty_midi

class RGCCalculator:
    def __init__(
            self, top_k: int = 16, 
            precision_digits: int = 4,
            ideal_tau: float = 0.125, 
            sigma_tau: float = 0.05,
            lambda_grid_fit: float = 20.0
        ):
        self.top_k = top_k
        self.precision_digits = precision_digits
        self.ideal_tau = ideal_tau
        self.sigma_tau = sigma_tau
        self.lambda_grid_fit = lambda_grid_fit

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
        except Exception:
            return np.array([])

    def calculate_rgc(self, file_path: str, verbose: bool = False) -> dict:
        ioi_sequence = self._get_raw_ioi(file_path)
        if len(ioi_sequence) < self.top_k:
            if verbose: 
                print(f"  -> Error for {os.path.basename(file_path)}: Not enough IOIs (found {len(ioi_sequence)}, need {self.top_k}).")
            return {"error": "Not enough IOIs to analyze."}

        rounded_ioi = np.round(ioi_sequence, self.precision_digits)
        ioi_counts = Counter(rounded_ioi)
        
        if len(ioi_counts) < 2:
            if verbose: 
                print(f"  -> Error for {os.path.basename(file_path)}: Not enough unique IOIs (found {len(ioi_counts)}).")
            return {"error": "Not enough unique IOIs to determine a grid."}

        top_k_iois = np.array([item for item, _ in ioi_counts.most_common(self.top_k)])
        
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
            if verbose: print(f"  -> Error for {os.path.basename(file_path)}: Could not find a suitable tau candidate > 0.01s.")
            return {"error": "Could not infer a valid tau."}
        
        final_tau = best_tau
        avg_grid_deviation = lowest_total_deviation
        
        grid_fit_score = np.exp(-self.lambda_grid_fit * avg_grid_deviation)
        
        numerator = (final_tau - self.ideal_tau) ** 2
        denominator = 2 * self.sigma_tau ** 2
        tau_reasonableness_score = np.exp(-numerator / denominator)
        
        rgc_score = grid_fit_score * tau_reasonableness_score

        return {
            "rgc_score": rgc_score, 
            "grid_fit_score": grid_fit_score,
            "tau_reasonableness_score": tau_reasonableness_score,
            "inferred_tau": final_tau, 
            "avg_grid_deviation": avg_grid_deviation
        }