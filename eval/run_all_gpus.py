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
import wandb
import os
import sys
import io
import logging

# Basic logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/run_all.log"),   # saves to file
        logging.StreamHandler()               # prints to console
    ]
)

logger = logging.getLogger(__name__)

logger.info("Logger initialized.")

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

    dataset_root_name = dataset_root.strip('/').split('/')[-1]

    output_root = Path(eval_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    summary_records = []  # accumulate results across models/datasets

    redo = eval_cfg['redo']

    GPUs = eval_cfg['GPUs']

    logging.info(f"Using GPUs: {GPUs}")



    # ------------------------------------------------------------
    # Iterate over all models and datasets
    # ------------------------------------------------------------
    for model_entry in eval_cfg["models"]:
        model_name = model_entry["name"]
        model_cfg_path = model_entry["config"]

        for dataset_entry in eval_cfg["datasets"]:
            
            

            dataset_name = dataset_entry["name"]
            dataset_path = dataset_entry["path"]
            inference_output_path = output_root / f"{dataset_root_name}_inference" / f"{model_name}" /f"{model_name}_{dataset_name}.jsonl"
            (output_root / f"{dataset_root_name}_inference" / f"{model_name}").mkdir(parents=True, exist_ok=True)
            print(f"\n🚀 Evaluating {model_name} on {dataset_name}...")

            
            if inference_output_path.exists() and not redo:
                logger.info(f"Skipping inference for {model_name} on {dataset_name} as output already exists at {inference_output_path} and redo is False.")
                # jump to metrics calculation
                pass
                

            else:
                # ----------------------------------------------------
                # Calculating the sharding for each gpu
                # ----------------------------------------------------
                
                # ensure as close as possible even distribution of datasets across GPUs
                dataset_len = len(open(Path(dataset_root) / dataset_path, 'r').readlines())

                # dataset_len = 10

                gpu_shard_dict = {}
                num_gpus = len(GPUs)

                base_shard = dataset_len // num_gpus
                remainder = dataset_len % num_gpus

                start = 0
                for i, gpu in enumerate(GPUs):
                    extra = 1 if i < remainder else 0
                    end = start + base_shard + extra
                    gpu_shard_dict[gpu] = {'start_idx': start, 'end_idx': end}
                    start = end

                
                processes = []

                for gpu in GPUs:
                    logger.info(f"GPU {gpu} assigned lines {gpu_shard_dict[gpu]['start_idx']} to {gpu_shard_dict[gpu]['end_idx']}")
                    gpu_shard_output_path = output_root / f"{dataset_root_name}_inference" / "shard"/ f"{model_name}_{dataset_name}_GPU{gpu}.jsonl"
                    logger.info(f"save path: {gpu_shard_output_path}")

                    # ----------------------------------------------------
                    # Run the model inference via run_eval.py
                    # ----------------------------------------------------
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["Local_rank"] = str(gpu)

                    p = subprocess.Popen([
                        "python", "-m", "eval.run_eval_gpus",
                        "--model_config", model_cfg_path,
                        "--dataset_root", dataset_root,
                        "--dataset_path", dataset_path,
                        "--save_path", str(gpu_shard_output_path),
                        "--dataset_limit", str(eval_cfg.get("dataset_limit", "")),  # optional limit for quick tests
                        "--start_idx", str(gpu_shard_dict[gpu]['start_idx']),
                        "--end_idx", str(gpu_shard_dict[gpu]['end_idx']),
                        "--gpu_id", str(gpu),
                        *(["--redo"] if redo else [])
                    ], env=env)
                    
                    processes.append(p)

                for p in processes:
                    p.wait()

                logger.info(f"All GPU shards for {model_name} on {dataset_name} completed.")

                # ----------------------------------------------------
                # Merge all GPU shard outputs into single inference file
                # ----------------------------------------------------

                # make shard folder if not exists
                (output_root / f"{dataset_root_name}_inference" / "shard").mkdir(parents=True, exist_ok=True)

                for gpu in GPUs:
                    gpu_shard_output_path = output_root / f"{dataset_root_name}_inference" / "shard"/ f"{model_name}_{dataset_name}_GPU{gpu}.jsonl"
                    logger.info(f"Merging shard file from GPU {gpu}: {gpu_shard_output_path}")

                    with open(gpu_shard_output_path, 'r') as shard_f, open(inference_output_path, 'a') as final_f:
                        for line in shard_f:
                            final_f.write(line)

                    # Optionally delete shard file after merging
                    # os.remove(gpu_shard_output_path)
                    # logger.info(f"Deleted shard file: {gpu_shard_output_path}")

            # Define ground truth reference file
            ref_path = Path(dataset_root) / dataset_path

            record = {"model": model_name, "dataset": dataset_name}

            # ----------------------------------------------------
            # Compute metrics defined in evaluation config
            # ----------------------------------------------------
            
            
            for metric_name in dataset_entry['metrics']:
                try:
                    metric_fn = get_metric_function(metric_name)
                    metric_output_path = output_root / metric_name /f"{model_name}_{dataset_name}.jsonl"
                    print(f"reference saved at: {ref_path}")
                    print(f"inference saved at: {inference_output_path}")
                    print(f"metric saved at: {metric_output_path}")
                    result = metric_fn(pred_path=inference_output_path, ref_path=ref_path)

                    print(f"   • {metric_name}: {result}")
                    # Handle dict metrics (e.g., {"f1": 0.83, "acc": 0.90})
                    if isinstance(result, dict):
                        for k, v in result.items():
                            record[f"{k}" if k != metric_name else k] = v
                    else:
                        record[metric_name] = result

                except Exception as e:
                    logging.warning(f"⚠️ Metric '{metric_name}' failed: {e}")
                    record[metric_name] = None

                

            summary_records.append(record)

    # ------------------------------------------------------------
    # Save consolidated evaluation results
    # ------------------------------------------------------------
    summary_df = pd.DataFrame(summary_records)
    summary_csv = output_root / f"{dataset_root_name}_inference" / "evaluation_summary.csv"
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
        default="configs/evaluation_config_gpus.yaml",
        help="Path to evaluation configuration YAML (default: configs/evaluation_config.yaml)",
    )

    args = parser.parse_args()

    
    with open(args.eval_config, "r") as f:
        eval_cfg = yaml.safe_load(f)
    
    
    # ------------------------------------------------------------
    # Initialize WandB run
    # ------------------------------------------------------------
    
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    
    wandb.init(
        project=eval_cfg.get("wandb_project", "SynDORBench"),
        name=eval_cfg.get("wandb_run_name", f"eval_orchestrator"),
        config=eval_cfg,
        reinit=True,
    )

    

    # Execute main evaluation loop
    main(args.eval_config)
