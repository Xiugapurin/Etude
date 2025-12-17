# etude/evaluation/reporting.py

import pandas as pd
from typing import Dict

from ..utils.logger import logger

class ReportGenerator:
    """
    Generates text-based and visual reports from a DataFrame of evaluation results.

    This class handles statistical analysis (grouping, describing) and plotting
    (boxplots) of the metric scores produced by the EvaluationRunner.
    """
    def __init__(self, results_df: pd.DataFrame, config: Dict):
        """
        Initializes the ReportGenerator.

        Args:
            results_df (pd.DataFrame): The DataFrame containing raw metric scores.
            config (Dict): The evaluation configuration dictionary, used for version
                           display names and output paths.
        """
        if results_df.empty:
            raise ValueError("Input DataFrame cannot be empty.")
            
        self.df = results_df
        self.config = config
        
        # Map version keys to their display names for prettier reports
        display_names = self.config.get('versions', {})
        self.df['display_name'] = self.df['version'].map(display_names).fillna(self.df['version'])
        
        # Identify which metrics were successfully calculated and are present in the DataFrame
        self.metrics = [
            col for col in ['wpd_score', 'rgc_score', 'ipe_score'] 
            if col in self.df.columns
        ]

    def print_summary(self):
        """
        Calculates and prints detailed statistical summaries for each metric to the console.
        """
        if not self.metrics:
            logger.warn("No valid metric columns found in the results. Cannot generate summary.")
            return

        logger.report_header("Evaluation Summary Report")

        # Print detailed statistics for each metric individually
        for metric in self.metrics:
            logger.report_section(f"Metric: {metric.upper()}")
            # Use groupby() and describe() for a comprehensive statistical summary
            summary = self.df.groupby('display_name')[metric].describe()
            # Sort by the mean score to easily see the best performing versions
            summary = summary.sort_values('mean', ascending=False).rename_axis('Model')
            print(summary.to_string())

        # Print a final, combined overview table of mean scores
        if len(self.metrics) > 1:
            logger.report_section("Overall Mean Scores")
            mean_summary = self.df.groupby('display_name')[self.metrics].mean()
            # Sort by the first available metric as a primary sorting key
            mean_summary = mean_summary.sort_values(self.metrics[0], ascending=False).rename_axis('Model')
            print(mean_summary.to_string())

        logger.report_separator()
