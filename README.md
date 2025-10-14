
# to install llavaction

pip install av
pip install open-clip-torch
pip install transformers==4.40.1 --force-reinstall
pip install flash-attn --no-build-isolation --no-cache-dir

# Example 

## To run a single evaluation
```
python -m eval.run_eval \
  --model_config ./configs/llavaction_config.yaml \
  --dataset_path index/human_detection/day.jsonl \
  --dataset_root /mnt/harddisk/jeremy/synthetic_dataset_output/SynDORBench \
  --save_path ./results/llavaction_action_day_test.jsonl \
  --dataset_limit 10

```
