# evaluate.py

import argparse
import yaml
from pathlib import Path

from etude.evaluation.runner import EvaluationRunner
from etude.evaluation.reporting import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline for the Etude project.")
    parser.add_argument("--config", type=str, default="configs/evaluate_config.yaml", help="Path to the evaluation configuration file.")
    parser.add_argument("--metrics", nargs='+', choices=['wpd', 'rgc', 'ipe'], help="Specify which metrics to run. Runs all by default.")
    parser.add_argument("--versions", nargs='+', help="Specify which versions to evaluate. Runs all by default.")
    parser.add_argument("--output-csv", type=str, help="Path to save the raw results to a CSV file.")
    parser.add_argument("--no-report", action="store_true", help="Only run calculations and save CSV, do not print summary or plot.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run the calculations
    runner = EvaluationRunner(config)
    results_df = runner.run(versions_to_run=args.versions, metrics_to_run=args.metrics)

    if results_df.empty:
        print("\n[WARN] No valid data could be processed. Aborting.")
        return

    # 2. Save raw data if requested
    csv_path = args.output_csv or Path(config['output_dir']) / config['report_csv_filename']
    results_df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Raw results saved to: {csv_path}")

    # 3. Generate report and plot unless suppressed
    if not args.no_report:
        reporter = ReportGenerator(results_df, config)
        reporter.print_summary()

    print("\n[INFO] Evaluation pipeline finished.")

if __name__ == "__main__":
    main()