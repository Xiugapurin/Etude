# etude/evaluation/metrics/rgc.py

from collections import Counter
import numpy as np

from .base_metric import get_onsets_from_file

class RGCCalculator:
    """
    Calculates the Rhythmic Grid Consistency (RGC).

    RGC measures how well the Inter-Onset Intervals (IOIs) of a piece align
    with a single underlying rhythmic grid (tau). It infers the most likely
    grid period (tau) from the most common IOIs and then calculates the average
    deviation of these IOIs from integer multiples of tau. A lower RGC score
    indicates a more consistent and stable rhythm.
    """
    def __init__(self, top_k: int = 8, precision_digits: int = 4, **kwargs):
        """
        Initializes the RGC calculator.

        Args:
            top_k (int): The number of most common IOIs to use for analysis.
            precision_digits (int): The number of decimal places to round IOIs to.
        """
        self.top_k = top_k
        self.precision_digits = precision_digits

    def calculate(self, file_path) -> dict:
        """
        Executes the full RGC calculation pipeline.

        Args:
            file_path (Path): Path to the MIDI or JSON file to be analyzed.

        Returns:
            dict: A dictionary containing the 'rgc_score' and 'inferred_tau', or an 'error' message.
        """
        unique_onsets = get_onsets_from_file(file_path)
        if len(unique_onsets) < 2:
            return {"error": "Not enough onsets for IOI calculation."}
            
        ioi_sequence = np.diff(unique_onsets)
        if len(ioi_sequence) < self.top_k:
            return {"error": "Not enough IOIs to analyze."}

        rounded_ioi = np.round(ioi_sequence, self.precision_digits)
        ioi_counts = Counter(rounded_ioi)
        
        if len(ioi_counts) < 2:
            return {"error": "Not enough unique IOIs to determine a grid."}

        top_k_iois = np.array([item for item, count in ioi_counts.most_common(self.top_k)])
        
        best_tau = -1.0
        lowest_total_deviation = float('inf')

        # Iterate through the most common IOIs as candidates for the grid period (tau)
        for tau_candidate in top_k_iois:
            if tau_candidate < 0.01: continue # Ignore unrealistically small tau values
            
            # Calculate how well other IOIs fit as integer multiples of the candidate tau
            ratios = top_k_iois / tau_candidate
            remainders = ratios - np.round(ratios)
            total_deviation = np.mean(np.abs(remainders))
            
            if total_deviation < lowest_total_deviation:
                lowest_total_deviation = total_deviation
                best_tau = tau_candidate
        
        if best_tau == -1.0:
            return {"error": "Could not infer a valid rhythmic grid period (tau)."}
        
        return {
            "rgc_score": lowest_total_deviation,
            "inferred_tau": best_tau
        }