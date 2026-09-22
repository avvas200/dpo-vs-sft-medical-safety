"""
Mechanistic — LoRA Frobenius weight-norm audit.

For each LoRA-adapted module, compute ||delta_W||_F where delta_W = (alpha/r) * B @ A.
Report per-module and pooled totals.

Writes: {"total_frobenius": float, "per_module": {module_name: frobenius}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_lora_norms(
    model_dir: Path,
    base_name: str,
    output_json: Path,
) -> Path:
    """
    Load the LoRA adapter, reconstruct delta_W per module, and record its
    Frobenius norm.

    delta_W = scaling * B @ A, where scaling = lora_alpha / lora_r.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, str(model_dir))

    per_module: dict[str, float] = {}
    total_sq = 0.0
    for name, module in peft_model.named_modules():
        # PEFT's LoRA modules expose lora_A and lora_B ModuleDicts keyed by
        # adapter name (default: "default").
        A_dict = getattr(module, "lora_A", None)
        B_dict = getattr(module, "lora_B", None)
        if A_dict is None or B_dict is None:
            continue
        if not hasattr(A_dict, "items"):  # not a ModuleDict; skip
            continue
        for adapter_name, A in A_dict.items():
            B = B_dict[adapter_name]
            # Extract weights as float32 for numerical stability
            A_w = A.weight.detach().float()
            B_w = B.weight.detach().float()
            scaling = getattr(module, "scaling", {}).get(adapter_name, 1.0)
            delta = scaling * (B_w @ A_w)
            fro = torch.linalg.norm(delta, ord="fro").item()
            per_module[name] = fro
            total_sq += fro ** 2

    total_frobenius = total_sq ** 0.5
    result = {
        "total_frobenius": total_frobenius,
        "per_module": per_module,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(result, f, indent=2)
    return output_json


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--base-name", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    audit_lora_norms(
        model_dir=args.model_dir,
        base_name=args.base_name,
        output_json=args.output,
    )


if __name__ == "__main__":
    _cli()
