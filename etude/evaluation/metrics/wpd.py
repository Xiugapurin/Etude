# etude/evaluation/metrics/wpd.py

import numpy as np
from typing import Dict

class WPDCalculator:
    """
    Calculates the Warping Path Deviation (WPD) from a pre-computed alignment result.

    WPD measures the temporal alignment consistency between two audio files.
    This class takes the result from an AudioAligner, fits a linear regression
    to the warping path, and then calculates the standard deviation of the path's
    deviation from this ideal linear alignment. A lower WPD score indicates
    better and more stable temporal alignment.
    """
    def __init__(self, subsample_step: int = 1, trim_seconds: float = 0, **kwargs):
        """
        Initializes the WPD calculator.

        Args:
            subsample_step (int): The step size for subsampling the warping path.
            trim_seconds (float): Seconds to trim from the beginning and end of the path.
        """
        if not isinstance(subsample_step, int) or subsample_step < 1:
            raise ValueError("subsample_step must be an integer >= 1.")
        if not isinstance(trim_seconds, (int, float)) or trim_seconds < 0:
            raise ValueError("trim_seconds must be a number >= 0.")
            
        self.subsample_step = subsample_step
        self.trim_seconds = trim_seconds

    def calculate(self, align_result: Dict, feature_rate: int = 50) -> Dict:
        """
        Executes the WPD calculation on a given alignment result.

        Args:
            align_result (Dict): The dictionary returned by an AudioAligner, containing
                                 'wp', 'num_frames_cover', and 'num_frames_origin'.
            feature_rate (int): The feature rate (in Hz) used during alignment.

        Returns:
            Dict: A dictionary containing the 'wpd_score' or an 'error' message.
        """
        try:
            # Unpack the necessary data from the alignment result
            wp = align_result.get('wp')
            num_frames_cover = align_result.get('num_frames_cover')
            num_frames_origin = align_result.get('num_frames_origin')

            if wp is None or num_frames_cover is None or num_frames_origin is None:
                return {"error": "Alignment result is missing required keys ('wp', 'num_frames_cover', 'num_frames_origin')."}

            # Generate time sequences for both audio files
            t_cover = np.arange(num_frames_cover) / feature_rate
            t_orig = np.arange(num_frames_origin) / feature_rate
            
            # Subsample the warping path if specified
            wp_to_process = wp[:, ::self.subsample_step]
            
            if wp_to_process.shape[1] < 10:
                return {"error": "Not enough points after subsampling."}
            
            wp_indices_cover = wp_to_process[0]
            wp_indices_orig = wp_to_process[1]

            wp_indices_cover_clipped = np.clip(wp_indices_cover, 0, num_frames_cover - 1)
            wp_indices_orig_clipped = np.clip(wp_indices_orig, 0, num_frames_origin - 1)
            
            path_t_cover = t_cover[wp_indices_cover_clipped]
            path_t_orig = t_orig[wp_indices_orig_clipped]

            # Trim the path from the beginning and end if specified
            if self.trim_seconds > 0 and path_t_orig[-1] > (2 * self.trim_seconds):
                start_time = self.trim_seconds
                end_time = path_t_orig[-1] - self.trim_seconds
                mask = (path_t_orig >= start_time) & (path_t_orig <= end_time)
                if np.sum(mask) > 10:
                    path_t_cover = path_t_cover[mask]
                    path_t_orig = path_t_orig[mask]

            # Fit a linear regression model (y = ax + b) to the time-aligned path
            coeffs = np.polyfit(path_t_cover, path_t_orig, 1)
            a, b = coeffs[0], coeffs[1]

            # Calculate the deviation of the actual path from the idealized linear path
            t_orig_predicted = a * path_t_cover + b
            deviation = path_t_orig - t_orig_predicted
            
            # The WPD score is the standard deviation of this deviation
            sigma_dev = np.std(deviation)
            
            return {"wpd_score": sigma_dev}
        
        except Exception as e:
            return {"error": str(e)}