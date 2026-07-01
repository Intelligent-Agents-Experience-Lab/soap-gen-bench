"""MLflow-based experiment tracker with GenAI tracing and judge support.

Usage:
    averages, run_id = evaluate_and_log(
        experiment_name="SOAP_Phase1_Prompt_Ablation",
        run_name="Qwen2.5-3B_H5_CoT",
        params={"model": "...", "prompt": "H5_CoT", "n": 50},
        target_fn=lambda inputs: {"output": generate(inputs["conversation"])},
        examples=examples,          # [{"inputs": {...}, "outputs": {...}}]
        evaluators=[bleu, rouge, meteor, completeness],
        judges=[correctness],       # called with (run, example, trace_id) → Judges tab
    )

Traces appear at /#/experiments/{id}/traces  (from @mlflow.trace on generate())
Judges appear at /#/experiments/{id}/judges  (from mlflow.log_feedback with LLM_JUDGE source)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import mlflow

from ..config import MLFLOW_TRACKING_URI


@dataclass
class _Run:
    outputs: dict = field(default_factory=dict)


@dataclass
class _Example:
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)


def evaluate_and_log(
    experiment_name: str,
    run_name: str,
    params: dict,
    target_fn: Callable[[dict], dict],
    examples: list[dict],
    evaluators: list[Callable],
    judges: list[Callable] | None = None,
    resume_run_id: str | None = None,
) -> tuple[dict[str, float], str]:
    """Run target_fn on each example, compute evaluators, log everything to MLflow.

    Args:
        evaluators: (run, example) → {"key", "score"} — logged as step metrics.
        judges:     (run, example, trace_id) → {"key", "score"} — logged via
                    mlflow.log_feedback() so they appear in the Judges tab.
    """
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(experiment_name)

    run_kwargs: dict[str, Any] = (
        {"run_id": resume_run_id} if resume_run_id else {"run_name": run_name}
    )

    with mlflow.start_run(**run_kwargs) as active_run:
        if not resume_run_id:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

        per_metric: dict[str, list[float]] = {}

        for step, example in enumerate(examples):
            inputs = example["inputs"]
            exp_outputs = example.get("outputs", {})

            try:
                output = target_fn(inputs)
            except Exception as e:
                print(f"  [step {step+1}/{len(examples)}] target error: {e}", flush=True)
                continue

            # Capture trace produced by @mlflow.trace inside target_fn
            gen_trace_id = mlflow.get_last_active_trace_id()

            mock_run = _Run(outputs=output)
            mock_example = _Example(inputs=inputs, outputs=exp_outputs)

            step_metrics: dict[str, float] = {}

            # Regular evaluators → step metrics
            for evaluator in evaluators:
                try:
                    result = evaluator(mock_run, mock_example)
                    entries = result if isinstance(result, list) else [result]
                    for r in entries:
                        step_metrics[r["key"]] = float(r.get("score", 0.0))
                except Exception as e:
                    print(f"  [{evaluator.__name__}] error: {e}", flush=True)

            # Judge evaluators → mlflow.log_feedback (Judges tab) + step metrics
            for judge in (judges or []):
                try:
                    result = judge(mock_run, mock_example, gen_trace_id)
                    entries = result if isinstance(result, list) else [result]
                    for r in entries:
                        if r.get("score") is not None:  # None = N/A (e.g. no RAG context)
                            step_metrics[r["key"]] = float(r["score"])
                except Exception as e:
                    print(f"  [{judge.__name__}] judge error: {e}", flush=True)

            for k, v in step_metrics.items():
                per_metric.setdefault(k, []).append(v)

            mlflow.log_metrics(step_metrics, step=step)
            scores_str = "  ".join(f"{k}={v:.3f}" for k, v in step_metrics.items())
            print(f"  [{step+1}/{len(examples)}] {scores_str}", flush=True)

        averages = {k: sum(v) / len(v) for k, v in per_metric.items() if v}
        # Retry to handle transient SQLite contention with the async trace exporter.
        for _attempt in range(4):
            try:
                mlflow.log_metrics({f"avg_{k}": v for k, v in averages.items()})
                break
            except Exception as _e:
                if _attempt == 3:
                    print(f"  Warning: could not log avg metrics to MLflow: {_e}", flush=True)
                else:
                    time.sleep(0.5 * (2 ** _attempt))
        run_id = active_run.info.run_id

    avg_str = "  ".join(f"{k}={v:.4f}" for k, v in averages.items())
    print(f"  averages: {avg_str}", flush=True)
    print(f"  mlflow run_id: {run_id}", flush=True)
    return averages, run_id
