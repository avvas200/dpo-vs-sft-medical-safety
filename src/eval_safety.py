"""
Safety evaluation — generate at temperature 0.7, judge with Llama-Guard-3-8B.

Writes one JSONL per (safety_domain, condition, seed) with fields:
  {"prompt_id": int, "safe": bool, "response": str, "judge_raw": str}

Per PLAN.md §4.5: judge is binary (safe / unsafe) at response level.
Judge–human κ spot-check on 100 random samples happens ONCE, on the first
seed's outputs, before the pooled analysis is executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_responses(
    model_dir: Path,
    prompts: list[dict],  # each has: prompt_id, prompt
    output_jsonl: Path,
    generation_seed: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    resume: bool = True,
) -> Path:
    """Generate one response per prompt at temp=0.7. Resumable."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel

    already = 0
    if resume and output_jsonl.exists():
        with output_jsonl.open() as f:
            for _ in f:
                already += 1
        print(f"[resume] {output_jsonl.name}: {already} generations present")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if already else "w"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with (model_dir / "adapter_config.json").open() as f:
        base_name = json.load(f)["base_model_name_or_path"]
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()

    set_seed(generation_seed)

    with output_jsonl.open(mode) as f:
        for i, record in enumerate(prompts):
            if i < already:
                continue
            messages = [{"role": "user", "content": record["prompt"]}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                out[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            f.write(json.dumps({
                "prompt_id": record["prompt_id"],
                "prompt": record["prompt"],
                "response": response,
            }) + "\n")
            f.flush()
    return output_jsonl


def judge_responses(
    generations_jsonl: Path,
    output_jsonl: Path,
    judge_model: str = "meta-llama/Llama-Guard-3-8B",
    resume: bool = True,
) -> Path:
    """Score each (prompt, response) pair as safe / unsafe using Llama-Guard-3-8B."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    generations = []
    with generations_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                generations.append(json.loads(line))

    already = 0
    if resume and output_jsonl.exists():
        with output_jsonl.open() as f:
            for _ in f:
                already += 1
        print(f"[resume] {output_jsonl.name}: {already} judgements present")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if already else "w"

    # Judge in 4-bit to save memory
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(judge_model)
    judge = AutoModelForCausalLM.from_pretrained(
        judge_model, quantization_config=bnb, device_map="auto",
    )
    judge.eval()

    with output_jsonl.open(mode) as f:
        for i, gen in enumerate(generations):
            if i < already:
                continue
            # Llama-Guard-3 template scores the assistant turn given the user prompt
            chat = [
                {"role": "user", "content": gen["prompt"]},
                {"role": "assistant", "content": gen["response"]},
            ]
            input_ids = tok.apply_chat_template(chat, return_tensors="pt").to(judge.device)
            with torch.no_grad():
                out = judge.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            verdict_text = tok.decode(
                out[0][input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()
            # Llama-Guard emits "safe" or "unsafe\n<categories>"
            safe = verdict_text.lower().startswith("safe")
            f.write(json.dumps({
                "prompt_id": gen["prompt_id"],
                "safe": safe,
                "response": gen["response"],
                "judge_raw": verdict_text,
            }) + "\n")
            f.flush()
    return output_jsonl


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--model-dir", type=Path, required=True)
    gen.add_argument("--prompts-jsonl", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--generation-seed", type=int, required=True)
    gen.add_argument("--no-resume", action="store_true")

    judge = sub.add_parser("judge")
    judge.add_argument("--generations-jsonl", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--judge-model", default="meta-llama/Llama-Guard-3-8B")
    judge.add_argument("--no-resume", action="store_true")

    args = ap.parse_args()

    if args.stage == "generate":
        prompts = []
        with args.prompts_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
        generate_responses(
            model_dir=args.model_dir,
            prompts=prompts,
            output_jsonl=args.output,
            generation_seed=args.generation_seed,
            resume=not args.no_resume,
        )
    else:
        judge_responses(
            generations_jsonl=args.generations_jsonl,
            output_jsonl=args.output,
            judge_model=args.judge_model,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    _cli()
