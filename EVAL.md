# Evaluation

## Example of evaluation

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