import pandas as pd
from pathlib import Path

def get_inference_from_df(model, df, root_dir="", limit=5):
    results = []

    # for prototyping, limit to head
    if limit:
        df = df.head(limit)
    
    for i in range(len(df)):
        img_path = df.loc[i, 'img_path']
        task_prompt = df.loc[i, 'prompt']

        abs_img_path = Path(img_path)
        if root_dir:
            abs_img_path = root_dir / img_path
        
        
        print(img_path, task_prompt)
        pred = model.predict(abs_img_path, task_prompt) 
        

        results.append({
            "img_path": str(img_path),
            "task_prompt": task_prompt,
            "prediction": pred
        })
    return pd.DataFrame(results)