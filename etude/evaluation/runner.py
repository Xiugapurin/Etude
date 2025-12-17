# etude/evaluation/runner.py

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from ..data.aligner import AudioAligner
from ..utils.logger import logger
from .metrics.wpd import WPDCalculator
from .metrics.rgc import RGCCalculator
from .metrics.ipe import IPECalculator


class EvaluationRunner:
    """Orchestrates the entire evaluation pipeline."""

    def __init__(self, config: Dict):
        """
        Initializes the EvaluationRunner with a configuration dictionary.

        Args:
            config (Dict): The loaded evaluation configuration from a YAML file.
        """
        self.config = config
        self.eval_dir = Path(config['eval_dir'])
        self.metadata_path = Path(config['metadata_path'])

        # Initialize the tools needed for the evaluation pipeline
        self.aligner = AudioAligner()
        self.calculators = {
            "wpd": WPDCalculator(**config['metrics'].get('wpd', {})),
            "rgc": RGCCalculator(**config['metrics'].get('rgc', {})),
            "ipe": IPECalculator(**config['metrics'].get('ipe', {})),
        }

    def run(self, versions_to_run: Optional[List[str]] = None, metrics_to_run: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Executes the full evaluation run with enhanced error logging.
        """
        if versions_to_run is None:
            versions_to_run = list(self.config['versions'].keys())
        if metrics_to_run is None:
            metrics_to_run = list(self.calculators.keys())

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logger.error(f"Metadata file not found at {self.metadata_path}")
            return pd.DataFrame()

        results_list = []
        pbar = tqdm(metadata, desc="Running Evaluation")
        for song_data in pbar:
            dir_name = song_data.get("dir_name")
            if not dir_name: continue

            song_dir = self.eval_dir / dir_name

            for version in versions_to_run:
                result_row = {'song': dir_name, 'version': version}
                
                # --- WPD Calculation Block ---
                if "wpd" in metrics_to_run:
                    origin_wav_path = song_dir / "origin.wav"
                    cover_wav_path = song_dir / f"{version}.wav"

                    align_result = self.aligner.align(origin_wav_path, cover_wav_path, song_dir)
                    
                    if align_result:
                        res = self.calculators['wpd'].calculate(align_result)
                        if "error" in res:
                            logger.warn(f"WPD calculation failed for '{dir_name}/{version}': {res['error']}")
                        else:
                            result_row.update(res)
                    else:
                        logger.skip(f"WPD for '{dir_name}/{version}': Alignment unavailable.")
                
                # --- RGC/IPE Calculation Block ---
                mid_path = song_dir / f"{version}.mid"
                json_path = song_dir / f"{version}.json"
                eval_file_path = mid_path if mid_path.exists() else json_path if json_path.exists() else None

                if eval_file_path:
                    if "rgc" in metrics_to_run:
                        res = self.calculators['rgc'].calculate(eval_file_path)
                        if "error" in res:
                            logger.warn(f"RGC calculation failed for '{dir_name}/{version}': {res['error']}")
                        else:
                            result_row.update(res)

                    if "ipe" in metrics_to_run:
                        res = self.calculators['ipe'].calculate(eval_file_path)
                        if "error" in res:
                            logger.warn(f"IPE calculation failed for '{dir_name}/{version}': {res['error']}")
                        else:
                            result_row.update(res)
                
                if len(result_row) > 2:
                    results_list.append(result_row)
        
        return pd.DataFrame(results_list)