# eval/run_eval.py
import argparse
import yaml
import pandas as pd
from pathlib import Path
from eval.models import get_model_class
from eval.inference.inference import get_inference_from_df
from eval.models.base import ModelConfig
import warnings
import torch

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=r".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=r".*Some weights of the model checkpoint.*")

def main():
    parser = argparse.ArgumentParser(description="Run SynDORBench model evaluation")
    parser.add_argument("--model_config", required=True, help="Path to model config YAML")
    parser.add_argument("--dataset_root", help="Override dataset root directory")
    parser.add_argument("--dataset_path", help="Override dataset path")
    parser.add_argument("--save_path", help="Override output path")
    parser.add_argument("--dataset_limit", type=int, default=None, help="Limit number of dataset samples for quick testing")

    
    args = parser.parse_args()

    # --- Load config ---
    with open(args.model_config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig.from_dict(cfg["model"])
    dataset_root = Path(args.dataset_root or cfg["dataset"].get("root", ""))
    dataset_path = Path(args.dataset_path or cfg["dataset"]["path"])
    abs_dataset_path = dataset_root / dataset_path if dataset_root else dataset_path
    save_path = Path(args.save_path or cfg["output"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Model: {model_cfg.model_name} ({model_cfg.model_path})")
    print(f"📁 Dataset: {abs_dataset_path} (limit={args.dataset_limit})")
    print(f"💾 Save to: {save_path}")
    print(f"🧪 Device: {model_cfg.device} | dtype: {model_cfg.dtype} | seed: {model_cfg.seed}")


    # --- Load model ---
    model_class = get_model_class(model_cfg.name)
    model = model_class(model_cfg)

    # --- Load dataset ---
    df = pd.read_json(abs_dataset_path, lines=True)
    if args.dataset_limit:
        df = df.head(args.dataset_limit)
    print(f"📊 Loaded dataset with {len(df)} samples")

    # --- Run inference ---
    results_df = get_inference_from_df(model, df, root_dir=dataset_root, limit=args.dataset_limit)
    results_df.to_json(save_path, lines=True, orient="records")

    print(f"\n✅ Inference complete. Saved to: {save_path}")


if __name__ == "__main__":
    main()
