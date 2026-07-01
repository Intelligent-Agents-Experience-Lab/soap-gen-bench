"""LoRA fine-tuning of Mistral-7B on the SOAP notes dataset via Unsloth.

Mirrors src/level3/stage_4_finetuning.ipynb for local (non-Colab) execution.
Requires unsloth: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
                  pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

Run:
    python -m source.models.finetune
    python -m source.models.finetune --max_steps 120 --num_examples 500
"""
import argparse
import json
import tempfile
from pathlib import Path

from ..config import DATASET_HF, LORA_ADAPTER_PATH

_INSTRUCTION = (
    "Generate a SOAP note from the clinical conversation. "
    "Output MUST be a valid JSON object with detailed content for the following keys: "
    '"subjective", "objective", "assessment", "plan".'
)

_UNSLOTH_MODEL = "unsloth/mistral-7b-bnb-4bit"  # Unsloth mirror of Mistral-7B-v0.1


def _build_jsonl(output_path: Path) -> int:
    """Write Alpaca-format JSONL, return number of examples written."""
    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset(DATASET_HF, split="train")
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Preparing SFT dataset"):
            conv = (
                row.get("patient_convo")
                or row.get("dialogue")
                or row.get("conversation")
                or row.get("text")
                or ""
            )
            soap = row.get("soap_notes") or row.get("soap_note") or ""
            if not conv or not soap:
                continue
            f.write(json.dumps({"instruction": _INSTRUCTION, "input": conv, "output": soap}) + "\n")
            written += 1
    return written


def train(max_steps: int = 60, num_examples: int | None = None, save_dir: str | None = None) -> None:
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "unsloth is required for fine-tuning.\n"
            'Install: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
            "         pip install --no-deps \"trl<0.9.0\" peft accelerate bitsandbytes"
        ) from e

    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    save_path = Path(save_dir) if save_dir else Path(LORA_ADAPTER_PATH)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Model + LoRA setup (mirrors notebook exactly) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=_UNSLOTH_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # --- Dataset ---
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)

    n_written = _build_jsonl(tmp_path)
    print(f"Dataset: {n_written} examples written to {tmp_path}")

    dataset = load_dataset("json", data_files=str(tmp_path), split="train")
    if num_examples:
        dataset = dataset.select(range(min(len(dataset), num_examples)))

    def _format(examples):
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            texts.append(
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}"
            )
        return {"text": texts}

    dataset = dataset.map(_format, batched=True)

    # --- Training (mirrors notebook exactly) ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    print(f"\nTraining for {max_steps} steps...")
    trainer.train()

    # --- Save ---
    print(f"\nSaving adapters to: {save_path}")
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    tmp_path.unlink(missing_ok=True)
    print(f"Done. Adapter saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Mistral-7B on SOAP notes")
    parser.add_argument("--max_steps", type=int, default=60, help="Training steps (default: 60)")
    parser.add_argument("--num_examples", type=int, default=None, help="Limit training examples")
    parser.add_argument("--save_dir", type=str, default=None, help="Adapter save path (default: LORA_ADAPTER_PATH)")
    args = parser.parse_args()
    train(args.max_steps, args.num_examples, args.save_dir)


if __name__ == "__main__":
    main()
