"""
Medical evaluation — MedMCQA (primary) and MedQA-USMLE (secondary).

Writes one JSONL per (benchmark, condition, seed) with fields:
  {"prompt_id": int, "correct": bool, "predicted": "A"|"B"|"C"|"D"|null, "gold": ...}

Consumed by src/stats.py::_load_outcome_jsonl. Prompt IDs are stable and
sorted; the stats module orders by prompt_id, so a partial (interrupted)
JSONL is still safe as long as records are complete lines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LETTER_RX = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> str | None:
    """Extract the first A/B/C/D letter from a model response."""
    if not text:
        return None
    stripped = text.lstrip()
    if stripped and stripped[0].upper() in "ABCD":
        return stripped[0].upper()
    m = LETTER_RX.search(text.upper())
    return m.group(1) if m else None


def evaluate_one_model(
    model_dir: Path,
    benchmark_records: list[dict],  # each has: prompt_id, prompt, gold_letter
    output_jsonl: Path,
    instruction: str,
    max_new_tokens: int = 64,
    resume: bool = True,
) -> Path:
    """
    Run greedy decoding on all prompts and write per-prompt correctness.

    Resumable: if output_jsonl already contains N complete lines, the first N
    prompts are skipped.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    already = 0
    if resume and output_jsonl.exists():
        with output_jsonl.open() as f:
            for _ in f:
                already += 1
        print(f"[resume] {output_jsonl.name}: {already} records already present")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if already else "w"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base + adapter
    with (model_dir / "adapter_config.json").open() as f:
        base_name = json.load(f)["base_model_name_or_path"]
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()

    with output_jsonl.open(mode) as f:
        for i, record in enumerate(benchmark_records):
            if i < already:
                continue
            prompt = f"{instruction}\n\n{record['prompt']}"
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(
                out[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            pred = parse_letter(text)
            gold = record["gold_letter"]
            f.write(json.dumps({
                "prompt_id": record["prompt_id"],
                "correct": bool(pred == gold),
                "predicted": pred,
                "gold": gold,
            }) + "\n")
            f.flush()  # partial-file safety on Colab interruption
    return output_jsonl


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--benchmark-jsonl", type=Path, required=True,
                    help="Preprocessed benchmark: prompt_id, prompt, gold_letter per line.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--instruction", type=str, default=(
        "Answer the following medical multiple-choice question. First give the "
        "single-letter answer (A, B, C, or D), then a brief explanation."
    ))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    records = []
    with args.benchmark_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    evaluate_one_model(
        model_dir=args.model_dir,
        benchmark_records=records,
        output_jsonl=args.output,
        instruction=args.instruction,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    _cli()
