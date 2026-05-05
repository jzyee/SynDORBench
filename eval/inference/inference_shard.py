import json
import pandas as pd
from pathlib import Path
import time
import logging

def get_inferences(
    model,
    input_path,
    output_path,
    root_dir="",
    limit=None,
    redo=False,
    chunk_size=1000,
    start_idx=None,
    end_idx=None,
):
    """
    Inference with resumable JSONL output.
    Processes only missing keys; appends new results line-by-line.

    Parameters
    ----------
    model : object
        Model exposing a .predict(image_path, prompt) method.
    input_path : str or Path
        Path to input JSONL file containing {"img_path": ..., "prompt": ...}.
    output_path : str or Path
        Path to output JSONL file for predictions.
    root_dir : str
        Root directory for relative image paths.
    limit : int, optional
        Limit number of samples for quick testing.
    redo : bool
        If True, overwrite existing file.
    chunk_size : int
        Write to disk every N samples.
    start_idx, end_idx : int, optional
        Optional slicing indices (for multi-GPU sharding).
    """

    root_dir = Path(root_dir)
    input_path, output_path = Path(input_path), Path(output_path)

    # --- Load input data (safe, positional slicing)
    df_input = pd.read_json(input_path, lines=True)
    df_input = df_input.reset_index(drop=True)

    if start_idx is not None and end_idx is not None:
        df_input = df_input.iloc[int(start_idx):int(end_idx)]
        logging.info(f"Sliced dataset [{start_idx}:{end_idx}) → {len(df_input)} rows")

    if limit is not None and limit > 0:
        df_input = df_input.head(limit)

    logging.info(f"Final dataset length for inference: {len(df_input)}")

    # --- Read existing output keys
    done_keys = set()
    if output_path.exists() and not redo:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add(str(rec["img_path"]))
                except Exception:
                    continue
        logging.info(f"[Resume] Found {len(done_keys)} completed entries.")
    else:
        if redo:
            logging.info("[Redo] Overwriting existing file.")
            output_path.unlink(missing_ok=True)
        logging.info("[Start] No existing file; starting fresh.")

    # --- Main inference loop
    buffer = []
    processed = 0
    start_time = time.time()

    for i, row in df_input.iterrows():
        img_path = str(row["img_path"])
        if not redo and img_path in done_keys:
            continue

        abs_img_path = Path(root_dir) / img_path if root_dir else Path(img_path)
        prompt = row["prompt"]

        try:
            pred = model.predict(abs_img_path, prompt)
        except Exception as e:
            logging.warning(f"[Error] {i}: {img_path} | {e}")
            pred = None

        buffer.append({
            "img_path": img_path,
            "prompt": prompt,
            "prediction": pred
        })
        processed += 1

        # Chunked flush
        if processed % chunk_size == 0:
            with open(output_path, "a", encoding="utf-8") as f:
                for rec in buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logging.info(f"[Checkpoint] {processed} new lines written.")
            buffer.clear()

    # --- Final flush
    if buffer:
        with open(output_path, "a", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = (time.time() - start_time) / 60
    logging.info(f"[Done] Total written: {processed} | Time: {elapsed:.1f} min")
    logging.info(f"✅ Inference complete. Output saved at: {output_path}")

    return processed
