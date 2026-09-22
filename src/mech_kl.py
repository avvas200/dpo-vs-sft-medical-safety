"""
Mechanistic — per-token KL divergence from the base model.

For each held-out prompt, compute KL(adapted || base) at each next-token
position over a fixed maximum span, then report the mean per token.

Writes: {"mean_token_kl": float, "n_tokens": int, "per_prompt_mean_kl": [...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


@__import__("contextlib").contextmanager
def _no_grad():
    import torch
    with torch.no_grad():
        yield


def compute_kl(
    model_dir: Path,
    base_name: str,
    prompts_jsonl: Path,
    output_json: Path,
    max_response_tokens: int = 64,
) -> Path:
    """
    Compute mean per-token KL(adapted || base) over a held-out prompt set.

    The KL is measured on tokens the ADAPTED model would generate under greedy
    decoding, evaluated against the base model's next-token distribution at
    the same position. This measures how far the adapted model has drifted
    in output-space, matching the definition used in the candidate's GSM8K
    prior work.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    prompts = []
    with prompts_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base.eval()
    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto",
        ),
        str(model_dir),
    )
    adapted.eval()

    per_prompt_mean = []
    total_kl = 0.0
    total_tokens = 0

    for record in prompts:
        prompt = record["prompt"]
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(adapted.device)

        # Greedy-generate `max_response_tokens` with the adapted model
        with _no_grad():
            gen = adapted.generate(
                input_ids,
                max_new_tokens=max_response_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        full = gen[0]  # includes prompt + generated
        n_prompt = input_ids.shape[-1]
        n_full = full.shape[-1]
        if n_full <= n_prompt:
            continue

        # Compute both models' logits on the full sequence in one pass each
        with _no_grad():
            base_logits = base(full.unsqueeze(0).to(base.device)).logits[0]
            adapted_logits = adapted(full.unsqueeze(0)).logits[0]

        # Compare at positions [n_prompt - 1, n_full - 2] (next-token positions
        # of the GENERATED span, exclusive of the very last no-target position).
        start, end = n_prompt - 1, n_full - 1
        adapted_lp = F.log_softmax(adapted_logits[start:end].float(), dim=-1)
        base_lp = F.log_softmax(base_logits[start:end].float(), dim=-1)
        adapted_p = adapted_lp.exp()
        # KL(adapted || base) = sum p_a * (log p_a - log p_b)
        kl = (adapted_p * (adapted_lp - base_lp)).sum(dim=-1)  # [span]
        span_kl = kl.mean().item()
        per_prompt_mean.append(span_kl)
        total_kl += kl.sum().item()
        total_tokens += kl.shape[0]

    result = {
        "mean_token_kl": total_kl / max(total_tokens, 1),
        "n_tokens": total_tokens,
        "per_prompt_mean_kl": per_prompt_mean,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(result, f, indent=2)
    return output_json


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--base-name", type=str, required=True)
    ap.add_argument("--prompts-jsonl", type=Path, required=True,
                    help="Held-out prompts JSONL (prompt_id, prompt).")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-response-tokens", type=int, default=64)
    args = ap.parse_args()
    compute_kl(
        model_dir=args.model_dir,
        base_name=args.base_name,
        prompts_jsonl=args.prompts_jsonl,
        output_json=args.output,
        max_response_tokens=args.max_response_tokens,
    )


if __name__ == "__main__":
    _cli()
