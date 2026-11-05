# Evaluation

## Example of Multiple Task Evaluation

1. Create your own config yaml to orchestrate the evaluation
   1. an example orchestration config is given at: [configs/evaluation_config.yaml](configs/evaluation_config.yaml)
      1. In the example config, we use:
         1. internvl3-1B 
         2. internvl3-8B
         3. smolVLM2-2.2B
   
   2. change the dataset path to where you have SynDORBench placed in configs/evaluation_config.yaml
        ```
        dataset_root: <dataset_path_here>
        ```
2. Activate your env
   1. should be created as mentioned in [INSTALL.md](INSTALL.md)
   ```
   conda activate syndorbench_env
   ```
3. Add your wandb api key as an env var to facilitate bug tracking and logging
   1. run the followng code in terminal, with your own wandb api key replacing xxxxx
   ```
    export WANDB_API_KEY="xxxxx"
   ```
4. Run the following code
   ```
   python -m eval.run_all_gpus --eval_config configs/evaluation_config.yaml
   ```
   
5. You can view the raw inferences from each model under the results folder once the evaluation code has completed. Each model's inferences are under its own model name. For e.g., for model internvl3-1B -> results/internvl3-1B
   
6. The results from each metric in the config file will be found in the overall evaluation summary csv: results/SynDORBench/evaluation_summary.csv 


If you want to evaluate your own models, you need to wrap your models in a class, following the style of models in [eval/models](eval/models). For e.g., [eval/models/gemma3n.py](eval/models/gemma3.py) 

## Example of Single Task Evaluation

### Action-MCQ evaluation (8-way action recognition)

The action-MCQ task scores models on an 8-way multiple-choice action-recognition
question per frame. Unlike the generative tasks above, it does **not** parse free
text: each model does a single forward pass and we take a softmax over just the
8 option-letter token ids (A–H) at the answer position — argmax is the prediction
(`score_options()`, see [eval/models/mcq_utils.py](eval/models/mcq_utils.py)).

The **primary metric is macro-F1, not accuracy**. Near-chance accuracy can hide a
majority-class prediction collapse (a model predicting one letter most of the
time); macro-F1, the reported prediction-letter entropy (bits), and the 8×8
confusion matrix all surface that collapse. See
[eval/metrics/mcq.py](eval/metrics/mcq.py).

### Running it

1. Use the action-MCQ orchestration config:
   [configs/action_mcq_evaluation_config.yaml](configs/action_mcq_evaluation_config.yaml).
   Set `dataset_root` to where SynDORBench is placed (its `index/action_mcq/{day,night,nv}.jsonl`
   files hold `prompt`, `options`, `gt_letter`, and `radius`).
2. Activate the env and export `WANDB_API_KEY` as above.
3. Run:
   ```
   python -m eval.run_all_gpus --eval_config configs/action_mcq_evaluation_config.yaml
   ```
   The orchestrator detects `action_mcq` datasets and routes each shard through the
   `score_options()` path (`--mcq`) instead of the generative `predict()` path.
4. Per-model predictions (`{"img_path","pred_letter","probs"}`) land under
   `results_action_mcq/...`; the `action_mcq` / `action_mcq_by_radius` metrics
   (macro-F1, accuracy, prediction entropy, confusion, and per-DORI-radius
   breakdowns) are written to the evaluation summary CSV.


