"""
Resumable MCQ inference, mirroring eval/inference/inference_shard.py's
JSONL resume + positional-sharding strategy, but scoring via
model.score_options() instead of model.predict().

score_options() does a single forward pass and returns a softmax over the
row's option-letter logits -- no generation, no text-parsing, no fallback.
See eval/models/base.py for the contract and eval/models/mcq_utils.py for
a reusable chat-template-based implementation.
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd


def get_inferences_mcq(
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
    MCQ inference with resumable JSONL output. Each row's `options` list
    (always 8-way for action_mcq) determines the option letters A..H passed
    to model.score_options(); the written record is
    {"img_path", "pred_letter", "probs"} where probs is aligned to the
    row's own option order (so it can be cross-checked against `options`
    and `gt_letter` in the reference index at metric time).

    start_idx / end_idx give the same positional sharding used by
    inference_shard.get_inferences(), so the orchestrator can fan this out
    across GPUs identically.
    """
    root_dir = Path(root_dir)
    input_path, output_path = Path(input_path), Path(output_path)

    df_input = pd.read_json(input_path, lines=True)
    df_input = df_input.reset_index(drop=True)

    if start_idx is not None and end_idx is not None:
        df_input = df_input.iloc[int(start_idx):int(end_idx)]
        logging.info(f"Sliced dataset [{start_idx}:{end_idx}) → {len(df_input)} rows")

    if limit is not None and limit > 0:
        df_input = df_input.head(limit)

    logging.info(f"Final dataset length for MCQ inference: {len(df_input)}")

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

    buffer = []
    processed = 0
    start = time.time()

    for i, row in df_input.iterrows():
        img_path = str(row["img_path"])
        if not redo and img_path in done_keys:
            continue

        abs_img_path = Path(root_dir) / img_path if root_dir else Path(img_path)
        prompt = row["prompt"]
        options = row["options"]
        option_letters = [chr(ord("A") + j) for j in range(len(options))]

        try:
            pred_letter, probs = model.score_options(abs_img_path, prompt, option_letters)
        except Exception as e:
            logging.warning(f"[Error] {i}: {img_path} | {e}")
            pred_letter, probs = None, None

        buffer.append({
            "img_path": img_path,
            "pred_letter": pred_letter,
            "probs": probs,
        })
        processed += 1

        if processed % chunk_size == 0:
            with open(output_path, "a", encoding="utf-8") as f:
                for rec in buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logging.info(f"[Checkpoint] {processed} new lines written.")
            buffer.clear()

    if buffer:
        with open(output_path, "a", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = (time.time() - start) / 60
    logging.info(f"[Done] Total written: {processed} | Time: {elapsed:.1f} min")
    logging.info(f"✅ MCQ inference complete. Output saved at: {output_path}")

    return processed
