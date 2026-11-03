# eval/models/mcq_utils.py
"""
Shared helpers for action-MCQ scoring (BaseModel.score_options()).
---------------------------------------------------------------------

Logit-probe scoring policy (A-only, no text-parse fallback): a single
forward pass, softmax restricted to the option-letter token ids at the
answer position, argmax = prediction. This holds even for CoT-style
models (e.g. Cosmos-Reason2) -- restricting softmax to just the 8 option
tokens makes the relative ranking among letters valid regardless of what
the full-vocab top token would have been.

`letter_token_ids` and `score_options_via_chat_template` are reusable
building blocks for adapters whose generate() already constructs inputs
via `processor.apply_chat_template(...)` into a single
`model(**inputs)`-compatible dict (e.g. qwen2_5_vl.py, internvl3.py). The
forward pass and chat-template construction are identical to those
adapters' existing generate() path; only the final `.generate()` call is
swapped for a direct forward + logit read.

Each adapter's score_options() override must be added (and smoke-tested
against its real checkpoint) explicitly, per-model, since adapter
input-construction is not uniform across the repo (e.g. llavaction.py's
custom tokenizer_image_token pipeline has no `processor.apply_chat_template`
at all and cannot use this helper as-is).
"""
import torch


def letter_token_ids(tokenizer, letters):
    """
    Resolve each letter ("A", "B", ...) to a single token id, checking both
    the bare form ("A") and the leading-space form (" A") and using
    whichever tokenizes to exactly one token.

    Raises ValueError -- does not silently fall back -- if a letter is not
    single-token in either form for this tokenizer. Per the A-only scoring
    policy, that case must be surfaced as a per-model gap, not guessed at.
    """
    ids = []
    for letter in letters:
        bare = tokenizer.encode(letter, add_special_tokens=False)
        spaced = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(bare) == 1:
            ids.append(bare[0])
        elif len(spaced) == 1:
            ids.append(spaced[0])
        else:
            raise ValueError(
                f"Letter '{letter}' is not single-token in either bare "
                f"({bare}) or leading-space ({spaced}) form for this "
                f"tokenizer -- score_options() cannot proceed without a "
                f"per-model fallback, which the A-only scoring policy forbids."
            )
    return ids


@torch.inference_mode()
def score_options_via_chat_template(
    model,
    processor,
    img_path,
    prompt,
    option_letters,
    device=None,
    image_key="image",
):
    """
    Generic single-forward-pass MCQ scorer for processor.apply_chat_template
    adapters.

    Parameters
    ----------
    image_key : str
        The content-block key an adapter's own generate() uses for the image
        (e.g. "image" for qwen2_5_vl.py's {"type": "image", "image": path},
        "path" for internvl3.py's {"type": "image", "path": path}). Callers
        must match their adapter's existing convention.

    Returns
    -------
    (pred_letter, probs) : (str, list[float])
        probs is aligned to `option_letters` order.
    """
    device = device or getattr(model, "device", "cpu")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", image_key: str(img_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    tokenizer = getattr(processor, "tokenizer", processor)
    option_ids = letter_token_ids(tokenizer, option_letters)

    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].squeeze(0)
    option_logits = last_logits[option_ids].float()
    probs = torch.softmax(option_logits, dim=-1).tolist()
    pred_letter = option_letters[int(torch.argmax(option_logits).item())]
    return pred_letter, probs
