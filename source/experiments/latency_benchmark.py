"""EL4 — Inference latency and VRAM benchmark across all models.

Addresses limitation: no inference efficiency metrics (deployment claim lacks evidence).

For each model: load → 2 warm-up calls → 10 timed calls → unload.
Logged metrics: load_time_s, avg_latency_ms, p95_latency_ms, peak_vram_mb.

Run:
    python -m source.experiments.latency_benchmark
    python -m source.experiments.latency_benchmark --group slm
    python -m source.experiments.latency_benchmark --models Qwen/Qwen2.5-3B-Instruct

MLflow experiment: SOAP_Latency_Benchmark
Run name pattern:  LAT_{ModelShortName}
"""
import argparse
import statistics
import time

import mlflow
import torch

from ..config import MLFLOW_EXPERIMENT_LAT, MODEL_GROUPS
from ..models.llm import manager

_BENCHMARK_PROMPT = (
    "Doctor: How are you feeling today?\n"
    "Patient: I have a headache and mild fever since yesterday.\n"
    "Doctor: Any other symptoms?\n"
    "Patient: No, just fatigue.\n"
    "Doctor: I'll prescribe some rest and paracetamol.\n\n"
    "Generate a SOAP note in JSON format for this conversation."
)

_WARMUP_CALLS = 2
_TIMED_CALLS = 10
_MAX_NEW_TOKENS = 512


def _reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def benchmark_model(model_name: str) -> None:
    short_name = model_name.split("/")[-1]
    print(f"\n{'='*60}\nBenchmarking: {model_name}\n{'='*60}", flush=True)

    _reset_vram_peak()

    # --- Load timing ---
    t0 = time.perf_counter()
    try:
        manager.load(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}", flush=True)
        return
    load_time_s = time.perf_counter() - t0
    print(f"  Loaded in {load_time_s:.2f}s", flush=True)

    messages = [{"role": "user", "content": _BENCHMARK_PROMPT}]

    # --- Warm-up (not timed) ---
    for _ in range(_WARMUP_CALLS):
        manager.generate(messages, max_new_tokens=_MAX_NEW_TOKENS)

    _reset_vram_peak()

    # --- Timed calls ---
    latencies: list[float] = []
    for i in range(_TIMED_CALLS):
        t0 = time.perf_counter()
        manager.generate(messages, max_new_tokens=_MAX_NEW_TOKENS)
        latencies.append((time.perf_counter() - t0) * 1000)
        print(f"  call {i+1}/{_TIMED_CALLS}: {latencies[-1]:.0f}ms", flush=True)

    peak_vram = _peak_vram_mb()
    manager.unload()

    avg_ms = statistics.mean(latencies)
    p95_ms = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    print(
        f"\n  Results — load: {load_time_s:.2f}s | avg: {avg_ms:.0f}ms | "
        f"p95: {p95_ms:.0f}ms | peak VRAM: {peak_vram:.0f}MB",
        flush=True,
    )

    # --- Log to MLflow ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_LAT)
    with mlflow.start_run(run_name=f"LAT_{short_name}"):
        mlflow.log_params({
            "model": model_name,
            "warmup_calls": _WARMUP_CALLS,
            "timed_calls": _TIMED_CALLS,
            "max_new_tokens": _MAX_NEW_TOKENS,
        })
        mlflow.log_metrics({
            "load_time_s": load_time_s,
            "avg_latency_ms": avg_ms,
            "p95_latency_ms": p95_ms,
            "peak_vram_mb": peak_vram,
        })
        # Log individual call latencies as steps
        for step, lat in enumerate(latencies):
            mlflow.log_metric("latency_ms", lat, step=step)


def run(model_names: list[str]) -> None:
    for model_name in model_names:
        benchmark_model(model_name)
    print("\nLatency benchmark complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EL4: latency benchmark → MLflow")
    parser.add_argument("--models", nargs="+", help="Specific model IDs")
    parser.add_argument("--group", choices=["slm", "llm", "all"], default="all")
    args = parser.parse_args()

    if args.models:
        models = args.models
    elif args.group == "all":
        models = MODEL_GROUPS["slm"] + MODEL_GROUPS["llm"]
    else:
        models = MODEL_GROUPS[args.group]

    run(models)


if __name__ == "__main__":
    main()
