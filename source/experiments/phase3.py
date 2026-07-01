"""Phase 3: evaluate the LoRA fine-tuned Mistral-7B — results logged to MLflow.

Run:
    python -m source.experiments.phase3 --num_examples 50
    python -m source.experiments.phase3 --condition lora_rag --num_examples 30

Three conditions:
  - base:     FT_BASE_MODEL (mistralai/Mistral-7B-v0.1) — untuned baseline
  - lora:     SaberaBanu/mistral-soap-notes — merged fine-tuned model, no RAG.
              Override: if LORA_ADAPTER_PATH dir exists, uses FT_BASE_MODEL + local adapter.
  - lora_rag: FT_FINETUNED_MODEL + champion RAG config (Hybrid RRF + note-level + section
              queries, k=8). Addresses the "incomplete combined pipeline" limitation (EL2).

Start the MLflow UI: mlflow ui  →  http://127.0.0.1:5000
"""
import argparse
from pathlib import Path

from ..config import FT_BASE_MODEL, FT_FINETUNED_MODEL, LORA_ADAPTER_PATH, MLFLOW_EXPERIMENT_P3
from ..data.dataset import load_examples
from ..evaluation.metrics import RAGAS_JUDGES, STANDARD_SUITE
from ..models.llm import manager
from ..tracking.mlflow_logger import evaluate_and_log

_SFT_INSTRUCTION = (
    "Generate a SOAP note from the clinical conversation. "
    "Output MUST be a valid JSON object with detailed content for the following keys: "
    '"subjective", "objective", "assessment", "plan".'
)


def _sft_prompt(conversation: str) -> str:
    return (
        f"### Instruction:\n{_SFT_INSTRUCTION}\n\n"
        f"### Input:\n{conversation}\n\n"
        "### Response:"
    )


def _clean(text: str) -> str:
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def _run_condition(label: str, model_name: str, adapter_path: str | None, examples: list) -> None:
    """Run one condition (base or lora) and log to MLflow."""
    print(f"\n--- Phase 3 condition: {label} ({model_name}) ---", flush=True)

    manager.load(model_name, adapter_path=adapter_path)

    def target(inputs: dict) -> dict:
        prompt = _sft_prompt(inputs.get("conversation", ""))
        # context is empty for non-RAG Phase 3 — RAGAS context judges will return None (skipped)
        return {"output": _clean(manager.generate([{"role": "user", "content": prompt}])), "context": ""}

    method = "SFT+LoRA (local adapter)" if adapter_path else ("SFT+LoRA (merged)" if label == "lora" else "base")
    try:
        evaluate_and_log(
            experiment_name=MLFLOW_EXPERIMENT_P3,
            run_name=f"P3_{label}_{model_name.split('/')[-1]}",
            params={
                "model": model_name,
                "adapter": adapter_path or "none",
                "condition": label,
                "method": method,
                "n_examples": len(examples),
            },
            target_fn=target,
            examples=examples,
            evaluators=STANDARD_SUITE,
            judges=RAGAS_JUDGES,
        )
    finally:
        manager.unload()


def _run_condition_rag(examples: list) -> None:
    """EL2: RAG+LoRA combined — FT_FINETUNED_MODEL with champion RAG config."""
    from .phase2 import RagPipeline

    model_name = FT_FINETUNED_MODEL
    print(f"\n--- Phase 3 condition: lora_rag ({model_name}) ---", flush=True)

    manager.load(model_name)
    pipeline = RagPipeline(
        retriever_type="hybrid",
        index_name="index_a_note",
        query_strategy="section",
        k=8,
    )

    try:
        evaluate_and_log(
            experiment_name=MLFLOW_EXPERIMENT_P3,
            run_name=f"P3_lora_rag_{model_name.split('/')[-1]}",
            params={
                "model": model_name,
                "condition": "lora_rag",
                "method": "SFT+LoRA+RAG (merged+champion)",
                "retriever": "hybrid",
                "index": "index_a_note",
                "query_strategy": "section",
                "k": 8,
                "n_examples": len(examples),
            },
            target_fn=pipeline,
            examples=examples,
            evaluators=STANDARD_SUITE,
            judges=RAGAS_JUDGES,
        )
    finally:
        manager.unload()


def run(num_examples: int, condition: str = "both") -> None:
    examples = load_examples(num_examples)
    print(f"Loaded {len(examples)} examples.", flush=True)

    if condition in ("base", "both"):
        _run_condition("base", FT_BASE_MODEL, None, examples)

    if condition in ("lora", "both"):
        local_adapter = Path(LORA_ADAPTER_PATH)
        if local_adapter.exists():
            _run_condition("lora", FT_BASE_MODEL, str(local_adapter), examples)
        else:
            _run_condition("lora", FT_FINETUNED_MODEL, None, examples)

    if condition == "lora_rag":
        _run_condition_rag(examples)

    print("\nPhase 3 complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: fine-tuning evaluation → MLflow")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument(
        "--condition",
        choices=["base", "lora", "lora_rag", "both"],
        default="both",
        help="Condition to run: base, lora, lora_rag, or both (default: both)",
    )
    args = parser.parse_args()
    run(args.num_examples, args.condition)


if __name__ == "__main__":
    main()
