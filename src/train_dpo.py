"""
DPO training — one training run at (base_model, seed).

Design constraint (PLAN.md §5.1): byte-identical to src/train_sft.py except
for learning rate, DPO beta, loss, and data collator. Any divergence beyond
those four items is a violation of the pre-registration.
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
    dpo_config: dict,
    seed: int,
    dataset_path: Path,
    output_dir: Path,
    resume_from_checkpoint: bool = True,
) -> Path:
    completed_marker = output_dir / "TRAINING_COMPLETE"
    if completed_marker.exists() and resume_from_checkpoint:
        print(f"[skip] {output_dir} already trained (marker present).")
        return output_dir

    import transformers
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig

    set_seed(seed)
    torch.manual_seed(seed)

    bnb = BitsAndBytesConfig(
        load_in_4bit=dpo_config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=dpo_config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=dpo_config["quantization"][
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_compute_dtype=getattr(
            torch, dpo_config["quantization"]["bnb_4bit_compute_dtype"]
        ),
    )
    model_name = dpo_config["base_model"]["name"]
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
        r=dpo_config["lora"]["r"],
        lora_alpha=dpo_config["lora"]["alpha"],
        lora_dropout=dpo_config["lora"]["dropout"],
        bias=dpo_config["lora"]["bias"],
        task_type=dpo_config["lora"]["task_type"],
        target_modules=dpo_config["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # DPO requires (prompt, chosen, rejected) columns
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def apply_chat(example):
        # Convert bare-prompt to chat-templated text so tokenization matches
        # SFT and deployment.
        return {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            ),
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    dataset = dataset.map(apply_chat)

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_config["training"]["epochs"],
        per_device_train_batch_size=dpo_config["training"]["micro_batch_size"],
        gradient_accumulation_steps=dpo_config["training"][
            "gradient_accumulation_steps"
        ],
        learning_rate=dpo_config["training"]["learning_rate"],
        lr_scheduler_type=dpo_config["training"]["lr_scheduler_type"],
        warmup_ratio=dpo_config["training"]["warmup_ratio"],
        weight_decay=dpo_config["training"]["weight_decay"],
        max_grad_norm=dpo_config["training"]["max_grad_norm"],
        optim=dpo_config["training"]["optim"],
        bf16=dpo_config["training"]["bf16"],
        logging_steps=dpo_config["training"]["logging_steps"],
        save_strategy=dpo_config["training"]["save_strategy"],
        save_total_limit=dpo_config["training"]["save_total_limit"],
        gradient_checkpointing=dpo_config["training"]["gradient_checkpointing"],
        report_to=dpo_config["training"]["report_to"],
        max_length=dpo_config["training"]["max_seq_length"],
        max_prompt_length=dpo_config["dpo"]["max_prompt_length"],
        beta=dpo_config["dpo"]["beta"],
        loss_type=dpo_config["dpo"]["loss_type"],
        seed=seed,
        data_seed=seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,      # PEFT LoRA -> reference is the frozen base
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
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
        "config_hash": hash(json.dumps(dpo_config, sort_keys=True)),
    }, indent=2))
    return output_dir


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpo-config", type=Path, default=Path("configs/dpo.yaml"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/dpo_pairs.jsonl"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.dpo_config)
    train_one_seed(
        dpo_config=cfg,
        seed=args.seed,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume,
    )


if __name__ == "__main__":
    _cli()
