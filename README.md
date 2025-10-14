
# to install llavaction

pip install av
pip install open-clip-torch
pip install transformers==4.40.1 --force-reinstall
pip install flash-attn --no-build-isolation --no-cache-dir


# models

- HuggingFaceTB/SmolVLM-Instruct (2B)
- deepseek-ai/deepseek-vl2-tiny (3B)
- microsoft/Phi-4-multimodal-instruct (5B)
- MLAdaptiveIntelligence/LLaVAction-7B // done
- Qwen/Qwen2.5-VL-7B-Instruct (7B)
- allenai/Molmo-7B-D-0924 (7B)
- lmms-lab/LLaVA-NeXT-Video-7B (7B) 
- ICTNLP/llava-mini-llama-3.1-8b (8B) // midway
- OpenGVLab/InternVL3_5-8B (8B)

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
