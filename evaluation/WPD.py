import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from corpus import Synchronizer

class WPDCalculator:
    """
    一個用於計算正規化路徑偏差 (WPD) 的工具類別。
    """
    def __init__(self, subsample_step: int = 1, trim_seconds: float = 0):
        """
        初始化 WPD 計算器。

        Args:
            subsample_step (int): 對齊路徑的降採樣步長。
            trim_seconds (float): 從路徑的頭尾各裁去幾秒不參與計算。
        """
        if not isinstance(subsample_step, int) or subsample_step < 1:
            raise ValueError("subsample_step 必須是 >= 1 的整數。")
        if not isinstance(trim_seconds, (int, float)) or trim_seconds < 0:
            raise ValueError("trim_seconds 必須是 >= 0 的數字。")
            
        self.subsample_step = subsample_step
        self.trim_seconds = trim_seconds
        
        self.synchronizer = Synchronizer()


    def calculate_wpd(self, origin_path: str, cover_path: str, song_dir: str) -> dict:
        """
        執行完整的 WPD 計算流程。
        """
        try:
            wp = self.synchronizer.get_wp(origin_path, cover_path, song_dir)
            
            t_cover = self.synchronizer.t1
            t_orig = self.synchronizer.t2
            
            if t_cover is None or t_orig is None:
                return {"error": "Timestamp sequence not generated."}

            wp_int = wp.astype(int)
            
            wp_to_process = wp_int[:, ::self.subsample_step] if self.subsample_step > 1 else wp_int
            
            if wp_to_process.shape[1] < 10:
                return {"error": "Not enough points after subsampling."}

            path_t_cover = t_cover[wp_to_process[0]]
            path_t_orig = t_orig[wp_to_process[1]]

            if self.trim_seconds > 0:
                total_duration = path_t_orig[-1]
                if total_duration > (2 * self.trim_seconds):
                    start_time = self.trim_seconds
                    end_time = total_duration - self.trim_seconds
                    mask = (path_t_orig >= start_time) & (path_t_orig <= end_time)
                    if np.sum(mask) > 10:
                        path_t_cover = path_t_cover[mask]
                        path_t_orig = path_t_orig[mask]

            coeffs = np.polyfit(path_t_cover, path_t_orig, 1)
            a, b = coeffs[0], coeffs[1]

            t_orig_predicted = a * path_t_cover + b
            deviation = path_t_orig - t_orig_predicted
            sigma_dev = np.std(deviation)
            
            return {"wpd_score": sigma_dev}
        
        except Exception as e:
            return {"error": str(e)}