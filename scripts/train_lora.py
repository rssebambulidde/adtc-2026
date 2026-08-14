#!/usr/bin/env python3
"""LoRA SFT for Kilimo (Qwen2.5-1.5B-Instruct). GPU preferred; CPU config supported."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dtype(name: str | None) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if not name:
        return torch.float32 if not torch.cuda.is_available() else torch.bfloat16
    key = str(name).lower()
    if key not in mapping:
        raise SystemExit(f"Unknown torch_dtype: {name}")
    return mapping[key]


def load_chat_jsonl(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise SystemExit(f"Each train row needs a messages field: {path}")
            rows.append(obj)
    if not rows:
        raise SystemExit(f"No examples in {path}. Run scripts/build_dataset.py first.")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kilimo-1.5b.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    train_path = Path(cfg["data"]["train_file"])
    if not train_path.is_file():
        raise SystemExit(
            f"Missing {train_path}. Run: python scripts/build_dataset.py --out {train_path}"
        )

    model_name = cfg["model"]["base"]
    dtype = resolve_dtype(cfg["model"].get("torch_dtype"))
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} dtype={dtype} device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(**cfg["lora"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_chat_jsonl(train_path)

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    t = cfg["training"]
    training_args = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        bf16=bool(t.get("bf16", torch.cuda.is_available())),
        fp16=bool(t.get("fp16", False)),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        report_to=t.get("report_to", "none"),
        seed=t.get("seed", 42),
        max_length=cfg["data"]["max_seq_length"],
        packing=False,
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"LoRA adapters saved to {t['output_dir']}")


if __name__ == "__main__":
    main()
