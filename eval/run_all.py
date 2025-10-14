# ============================================================
# SynDORBench: Evaluation Orchestrator
# ------------------------------------------------------------
# This script automates benchmark evaluation across:
#   • Multiple vision-language models (e.g., LLaVAction, InternVL3, etc.)
#   • Multiple dataset splits (e.g., day, night, nv)
#   • Multiple metrics (e.g., accuracy, F1, BLEU)
#
# It calls `eval/run_eval.py` for each model–dataset pair,
# aggregates the outputs, computes metrics, and generates
# a summary CSV suitable for leaderboard reporting.
# ============================================================

import argparse
import yaml
import subprocess
import pandas as pd
from pathlib import Path
from eval.metrics.loader import get_metric_function


# ------------------------------------------------------------
# 1. Core Evaluation Logic
# ------------------------------------------------------------
def main(eval_config_path: str):
    """
    Execute benchmark evaluations based on the provided config file.

    Parameters
    ----------
    eval_config_path : str
        Path to the YAML configuration file defining models, datasets, metrics, and output paths.
    """

    # ------------------------------------------------------------
    # Load evaluation configuration
    # ------------------------------------------------------------
    with open(eval_config_path, "r") as f:
        eval_cfg = yaml.safe_load(f)

    dataset_root = eval_cfg['dataset_root']

    output_root = Path(eval_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    summary_records = []  # accumulate results across models/datasets

    # ------------------------------------------------------------
    # Iterate over all models and datasets
    # ------------------------------------------------------------
    for model_entry in eval_cfg["models"]:
        model_name = model_entry["name"]
        model_cfg_path = model_entry["config"]

        for dataset_entry in eval_cfg["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_path = dataset_entry["path"]
            output_path = output_root / f"{model_name}_{dataset_name}.jsonl"

            print(f"\n🚀 Evaluating {model_name} on {dataset_name}...")

            # ----------------------------------------------------
            # Run the model inference via run_eval.py
            # ----------------------------------------------------
            subprocess.run([
                "python", "-m", "eval.run_eval",
                "--model_config", model_cfg_path,
                "--dataset_root", dataset_root,
                "--dataset_path", dataset_path,
                "--save_path", str(output_path),
                "--dataset_limit", str(eval_cfg.get("dataset_limit", ""))  # optional limit for quick tests
            ], check=True)

            # Define ground truth reference file
            ref_path = Path(dataset_root) / dataset_path

            record = {"model": model_name, "dataset": dataset_name}

            # ----------------------------------------------------
            # Compute metrics defined in evaluation config
            # ----------------------------------------------------
            for metric_name in dataset_entry['metrics']:
                try:
                    metric_fn = get_metric_function(metric_name)

                    print(f"reference: {ref_path}")
                    print(f"prediction: {output_path}")
                    result = metric_fn(pred_path=output_path, ref_path=ref_path)

                    print(f"   • {metric_name}: {result}")
                    # Handle dict metrics (e.g., {"f1": 0.83, "acc": 0.90})
                    if isinstance(result, dict):
                        for k, v in result.items():
                            record[f"{k}" if k != metric_name else k] = v
                    else:
                        record[metric_name] = result

                except Exception as e:
                    print(f"⚠️ Metric '{metric_name}' failed: {e}")
                    record[metric_name] = None

            summary_records.append(record)

    # ------------------------------------------------------------
    # Save consolidated evaluation results
    # ------------------------------------------------------------
    summary_df = pd.DataFrame(summary_records)
    summary_csv = output_root / "evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n✅ Summary saved to: {summary_csv}")
    print(summary_df)


# ------------------------------------------------------------
# 2. Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    # CLI argument parser (kept near entry point for clarity)
    parser = argparse.ArgumentParser(
        description="Run all SynDORBench evaluations for multiple models and datasets."
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="configs/evaluation_config.yaml",
        help="Path to evaluation configuration YAML (default: configs/evaluation_config.yaml)",
    )

    args = parser.parse_args()

    # Execute main evaluation loop
    main(args.eval_config)
