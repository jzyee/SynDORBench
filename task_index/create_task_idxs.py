from pathlib import Path
import yaml
import pandas as pd
import os
import sys
import logging

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "create_task_idx.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_task_config(config_path: str):
    """Load YAML configuration file."""
    logger.info(f"Loading task configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Config loaded: {config}")
    return config

def create_task_idx(task_config_path:str)->None:
    '''
    task yaml in format:
        tasks:
            human_detection:
                # the task is to check for human presence in the image
                prompt: "Is there a human in the image?"
                output: "human_detection_idx.jsonl"
    '''
    def create_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
            logging.info(f"Created directory: {dir}")
        else:
            logging.info(f"Directory already exists: {dir}")

    task_config = load_task_config(config_path=task_config_path)
    
    dataset_root = Path(task_config['dataset_root'])
    rel_master_idx_path = Path(task_config['master_idx_path'])

    abs_master_idx_path = dataset_root / rel_master_idx_path

    logging.info(f"Absolute master index path: {abs_master_idx_path}")
    

    # read in master index
    if not abs_master_idx_path.exists():
        raise FileNotFoundError(f"Master index file not found: {abs_master_idx_path}")
    with open(abs_master_idx_path, "r") as f:
        master_index = pd.read_json(f, lines=True)


    # partion master_index into 3 partions
    # (day, night, nv)

    day_lightings = ['morning_north', 'morning_south', 'morning_east', 'morning_west',
                     'noon_north', 'noon_south', 'noon_east', 'noon_west',]
    night_lightings = ['evening_north', 'evening_south', 'evening_east', 'evening_west',
                       'night']

    day_df = master_index[master_index['lighting'].isin(day_lightings)]
    night_df = master_index[~master_index['lighting'].isin(day_lightings)]
    nv_df = night_df.copy()
    nv_df['img_path'] = nv_df['nv_path']
    nv_df['nv_path'] = None


    # display(day_df.head())
    # display(night_df.head())
    # display(nv_df.head())
    logger.info(f"→ Day samples:  {len(day_df)}")
    logger.info(f"→ Night samples: {len(night_df)}")
    logger.info(f"→ NV samples:    {len(nv_df)}")

    tasks = task_config.get("tasks", {})

    if not tasks:
        logger.warning("No tasks found in the configuration.")
        return

    for task_name, task_info in tasks.items():
        prompt = task_info['prompt']
        logger.info(f"Processing task: {task_name} with prompt: {prompt}")
        

        # create task dir if not exist
        task_dir = dataset_root / "index" / task_name
        create_dir(task_dir)

        tmp_df = day_df.copy()
        tmp_df['prompt'] = prompt
        tmp_df['task'] = task_name
        # save to output file
        tmp_df.to_json(task_dir / "day.jsonl", lines=True, orient="records")

        tmp_df = night_df.copy()
        tmp_df['prompt'] = prompt
        tmp_df['task'] = task_name
        tmp_df.to_json(task_dir / "night.jsonl", lines=True, orient="records")

        tmp_df = nv_df.copy()
        tmp_df['prompt'] = prompt
        tmp_df['task'] = task_name
        tmp_df.to_json(task_dir / "nv.jsonl", lines=True, orient="records")

        print(f"→ task index completed: {task_name}")
        print(f"→ Output file: {task_dir / 'day.jsonl'}")
        print(f"→ Output file: {task_dir / 'night.jsonl'}")
        print(f"→ Output file: {task_dir / 'nv.jsonl'}")

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create task indices from configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tasks_config.yaml",
        help="Path to the task configuration YAML file."
    )
    args = parser.parse_args()

    config_path = args.config
    logger.info(f"Starting task index creation using config: {config_path}")

    try:
        create_task_idx(task_config_path=config_path)
        logger.info("Finished task index creation successfully.")
    except Exception as e:
        logger.exception(f"Task index creation failed: {e}")