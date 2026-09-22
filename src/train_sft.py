"""
SFT training — one training run at (base_model, seed).

The full 3-seed sweep is driven by notebooks/run_all_colab.ipynb, which
loops this function over configs/seeds.yaml::training_seeds.

Design constraint (PLAN.md §5.1): everything except the adaptation objective
must be byte-identical between SFT and DPO. The two scripts share the same
base model load, tokenizer, LoRA config, effective batch size, LR schedule
shape, sequence length, and evaluation harness. The only differences are the
learning rate, the loss, and the data collator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def train_one_seed(
    sft_config: dict,
    seed: int,
    dataset_path: Path,
    output_dir: Path,
    resume_from_checkpoint: bool = True,
) -> Path:
    """
    Train one SFT model at (base_model, seed) → save LoRA adapter to output_dir.

    Fully resumable per PLAN.md §6.2: if `output_dir` already contains a
    completed adapter, this function returns without retraining.

    NOTE: This function imports transformers/trl/peft lazily so the module
    can be imported for docs or dry-runs on a machine without a GPU.
    """
    completed_marker = output_dir / "TRAINING_COMPLETE"
    if completed_marker.exists() and resume_from_checkpoint:
        print(f"[skip] {output_dir} already trained (marker present).")
        return output_dir

    # Lazy imports — heavy dependencies
    import transformers
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    set_seed(seed)
    torch.manual_seed(seed)

    # 4-bit quantized base
    bnb = BitsAndBytesConfig(
        load_in_4bit=sft_config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=sft_config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=sft_config["quantization"][
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_compute_dtype=getattr(
            torch, sft_config["quantization"]["bnb_4bit_compute_dtype"]
        ),
    )
    model_name = sft_config["base_model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=sft_config["lora"]["r"],
        lora_alpha=sft_config["lora"]["alpha"],
        lora_dropout=sft_config["lora"]["dropout"],
        bias=sft_config["lora"]["bias"],
        task_type=sft_config["lora"]["task_type"],
        target_modules=sft_config["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def formatting(example):
        # Uses the chat template of the base model so tokenization matches
        # deployment.
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft_config["training"]["epochs"],
        per_device_train_batch_size=sft_config["training"]["micro_batch_size"],
        gradient_accumulation_steps=sft_config["training"][
            "gradient_accumulation_steps"
        ],
        learning_rate=sft_config["training"]["learning_rate"],
        lr_scheduler_type=sft_config["training"]["lr_scheduler_type"],
        warmup_ratio=sft_config["training"]["warmup_ratio"],
        weight_decay=sft_config["training"]["weight_decay"],
        max_grad_norm=sft_config["training"]["max_grad_norm"],
        optim=sft_config["training"]["optim"],
        bf16=sft_config["training"]["bf16"],
        logging_steps=sft_config["training"]["logging_steps"],
        save_strategy=sft_config["training"]["save_strategy"],
        save_total_limit=sft_config["training"]["save_total_limit"],
        gradient_checkpointing=sft_config["training"]["gradient_checkpointing"],
        report_to=sft_config["training"]["report_to"],
        max_seq_length=sft_config["training"]["max_seq_length"],
        seed=seed,
        data_seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatting,
    )

    resume = None
    if resume_from_checkpoint:
        ckpts = list(output_dir.glob("checkpoint-*"))
        if ckpts:
            resume = str(max(ckpts, key=lambda p: int(p.name.split("-")[-1])))
            print(f"[resume] {resume}")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    completed_marker.write_text(json.dumps({
        "seed": seed,
        "base_model": model_name,
        "config_hash": hash(json.dumps(sft_config, sort_keys=True)),
    }, indent=2))
    return output_dir


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-config", type=Path, default=Path("configs/sft.yaml"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/sft_dataset.jsonl"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.sft_config)
    train_one_seed(
        sft_config=cfg,
        seed=args.seed,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume,
    )


if __name__ == "__main__":
    _cli()
