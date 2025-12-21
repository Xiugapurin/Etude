# evaluate.py

import argparse

from etude.config import load_config
from etude.evaluation.runner import EvaluationRunner
from etude.evaluation.reporting import ReportGenerator
from etude.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run the evaluation pipeline for the Etude project."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--metrics", nargs='+', choices=['wpd', 'rgc', 'ipe'],
        help="Specify which metrics to run. Runs all by default."
    )
    parser.add_argument(
        "--versions", nargs='+',
        help="Specify which versions to evaluate. Runs all by default."
    )
    parser.add_argument(
        "--output-csv", type=str,
        help="Path to save the raw results to a CSV file."
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Only run calculations and save CSV, do not print summary or plot."
    )

    args = parser.parse_args()

    config = load_config(args.config)
    eval_config = config.eval

    output_dir = config.paths.eval_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run the calculations
    logger.step("Running evaluation")
    runner = EvaluationRunner(eval_config)
    results_df = runner.run(versions_to_run=args.versions, metrics_to_run=args.metrics)

    if results_df.empty:
        logger.warn("No valid data could be processed.")
        return

    # 2. Save raw data
    logger.step("Saving results")
    csv_path = args.output_csv or output_dir / eval_config.report_csv_filename
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # 3. Generate report unless suppressed
    if not args.no_report:
        logger.step("Generating report")
        reporter = ReportGenerator(results_df, eval_config)
        reporter.print_summary()

    logger.success("Evaluation pipeline finished.")


if __name__ == "__main__":
    main()
