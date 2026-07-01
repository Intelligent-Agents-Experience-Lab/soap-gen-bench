"""EL5 — Paired statistical significance tests for pairwise metric comparisons.

Addresses limitation: no paired bootstrap CIs, McNemar tests, or permutation tests.
Per-step metrics from evaluate_and_log (step 0..n-1) are fetched from MLflow and used
for paired analysis. Two test types are applied per metric:

  Binary (all values 0/1, e.g. RAGAS judges):
    - Wilson CI per run
    - McNemar paired test (Yates-corrected)

  Continuous (e.g. BLEU, ROUGE-L, METEOR, correctness):
    - Bootstrap CI per run
    - Paired permutation test

Run:
    python -m source.experiments.statistical_analysis --run_a RUN_ID_A --run_b RUN_ID_B
    python -m source.experiments.statistical_analysis --run_a ID_A --run_b ID_B --metric rouge
    python -m source.experiments.statistical_analysis --run_a ID_A    # single-run CIs only

MLflow experiment: SOAP_Statistical_Analysis
Run name pattern:  STATS_{run_a[:8]}[_vs_{run_b[:8]}]
"""
from __future__ import annotations

import argparse
import math

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from ..config import MLFLOW_EXPERIMENT_STATS, MLFLOW_TRACKING_URI

_N_BOOT = 10_000
_N_PERM = 10_000
_ALPHA = 0.05
_Z95 = 1.96

_STEP_METRICS = [
    "completeness", "bleu", "rouge", "meteor",
    "correctness",
    "answer_relevancy", "faithfulness",
    "contextual_precision", "contextual_recall", "contextual_relevancy",
]


def _set_uri() -> None:
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def fetch_case_scores(run_id: str, metric_key: str) -> list[float]:
    """Return per-case scores (step-level) sorted by step ascending."""
    client = MlflowClient()
    history = client.get_metric_history(run_id, metric_key)
    history.sort(key=lambda m: (m.step, m.timestamp))
    return [m.value for m in history]


def _is_binary(scores: list[float]) -> bool:
    return all(v in (0.0, 1.0) for v in scores)


def wilson_ci(scores: list[float]) -> tuple[float, float]:
    n = len(scores)
    if n == 0:
        return 0.0, 1.0
    k = sum(1 for s in scores if s >= 0.5)
    p = k / n
    denom = 1 + _Z95 ** 2 / n
    center = (p + _Z95 ** 2 / (2 * n)) / denom
    margin = _Z95 * math.sqrt(p * (1 - p) / n + _Z95 ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(scores: list[float], rng: np.random.Generator) -> tuple[float, float]:
    a = np.array(scores)
    boot = rng.choice(a, size=(_N_BOOT, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * _ALPHA / 2)), float(np.percentile(boot, 100 * (1 - _ALPHA / 2)))


def bootstrap_diff_ci(
    a_scores: list[float], b_scores: list[float], rng: np.random.Generator
) -> tuple[float, float]:
    diff = np.array(a_scores) - np.array(b_scores)
    boot = rng.choice(diff, size=(_N_BOOT, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * _ALPHA / 2)), float(np.percentile(boot, 100 * (1 - _ALPHA / 2)))


def mcnemar(a: list[float], b: list[float]) -> tuple[float, float]:
    """Yates-corrected McNemar. p-value via chi-squared(df=1) = erfc(sqrt(stat/2))."""
    n01 = sum(1 for x, y in zip(a, b) if x < 0.5 and y >= 0.5)
    n10 = sum(1 for x, y in zip(a, b) if x >= 0.5 and y < 0.5)
    total = n01 + n10
    if total == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / total
    p = math.erfc(math.sqrt(stat / 2))
    return float(stat), float(p)


def permutation_test(
    a: list[float], b: list[float], rng: np.random.Generator
) -> tuple[float, float]:
    """Paired permutation test for mean(a) - mean(b). Returns (observed_diff, p_value)."""
    diff = np.array(a) - np.array(b)
    observed = float(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(_N_PERM, len(diff)))
    null = (signs * diff).mean(axis=1)
    p = float((np.abs(null) >= abs(observed)).mean())
    return observed, p


def analyze(run_a: str, run_b: str | None, metric_filter: str | None) -> dict:
    rng = np.random.default_rng(42)
    keys = [metric_filter] if metric_filter else _STEP_METRICS
    results: dict = {}

    for key in keys:
        a_scores = fetch_case_scores(run_a, key)
        if not a_scores:
            continue

        binary = _is_binary(a_scores)
        ci_a = wilson_ci(a_scores) if binary else bootstrap_ci(a_scores, rng)
        entry: dict = {
            "binary": binary,
            "n_a": len(a_scores),
            "mean_a": float(np.mean(a_scores)),
            "ci_a": ci_a,
        }

        if run_b:
            b_scores = fetch_case_scores(run_b, key)
            if not b_scores:
                entry["note"] = "metric absent in run_b"
            elif len(b_scores) != len(a_scores):
                entry["note"] = f"length mismatch: a={len(a_scores)}, b={len(b_scores)} — paired tests skipped"
                entry["n_b"] = len(b_scores)
                entry["mean_b"] = float(np.mean(b_scores))
                entry["ci_b"] = wilson_ci(b_scores) if binary else bootstrap_ci(b_scores, rng)
            else:
                entry["n_b"] = len(b_scores)
                entry["mean_b"] = float(np.mean(b_scores))
                entry["ci_b"] = wilson_ci(b_scores) if binary else bootstrap_ci(b_scores, rng)
                if binary:
                    stat, p = mcnemar(a_scores, b_scores)
                    entry.update({"test": "McNemar", "stat": stat, "p_value": p})
                else:
                    diff, p = permutation_test(a_scores, b_scores, rng)
                    diff_ci = bootstrap_diff_ci(a_scores, b_scores, rng)
                    entry.update({"test": "Permutation", "mean_diff": diff, "diff_ci": diff_ci, "p_value": p})

        results[key] = entry

    return results


def print_report(results: dict, run_a: str, run_b: str | None) -> None:
    print(f"\n{'='*70}")
    print("Statistical Analysis")
    print(f"  Run A : {run_a}")
    if run_b:
        print(f"  Run B : {run_b}")
    print(f"  CI: {int((1 - _ALPHA) * 100)}%  bootstrap n={_N_BOOT:,}  permutation n={_N_PERM:,}")
    print(f"{'='*70}")

    for key, r in results.items():
        lo_a, hi_a = r["ci_a"]
        ci_label = "Wilson" if r["binary"] else "Bootstrap"
        print(f"\n  {key}  [{ci_label} CI]")
        print(f"    A: {r['mean_a']:.4f}  [{lo_a:.4f}, {hi_a:.4f}]  n={r['n_a']}")

        if "mean_b" in r:
            lo_b, hi_b = r["ci_b"]
            print(f"    B: {r['mean_b']:.4f}  [{lo_b:.4f}, {hi_b:.4f}]  n={r['n_b']}")

        if "p_value" in r:
            p = r["p_value"]
            sig = " *" if p < 0.05 else " (ns)"
            if "mean_diff" in r:
                dl, dh = r.get("diff_ci", (float("nan"), float("nan")))
                print(f"    diff(A-B): {r['mean_diff']:+.4f}  [{dl:.4f}, {dh:.4f}]  "
                      f"{r['test']} p={p:.4f}{sig}")
            else:
                print(f"    {r['test']} p={r['p_value']:.4f}{sig}")

        if "note" in r:
            print(f"    Note: {r['note']}")


def log_to_mlflow(results: dict, run_a: str, run_b: str | None) -> None:
    _set_uri()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_STATS)
    run_name = f"STATS_{run_a[:8]}" + (f"_vs_{run_b[:8]}" if run_b else "")
    flat: dict[str, float] = {}
    for key, r in results.items():
        flat[f"{key}_mean_a"] = r.get("mean_a", 0.0)
        flat[f"{key}_ci_a_lo"] = r["ci_a"][0]
        flat[f"{key}_ci_a_hi"] = r["ci_a"][1]
        if "mean_b" in r:
            flat[f"{key}_mean_b"] = r["mean_b"]
        if "p_value" in r:
            flat[f"{key}_p"] = r["p_value"]
        if "mean_diff" in r:
            flat[f"{key}_diff"] = r["mean_diff"]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"run_a": run_a, "run_b": run_b or "none", "alpha": _ALPHA})
        mlflow.log_metrics(flat)
    print(f"\nLogged to MLflow: {MLFLOW_EXPERIMENT_STATS}", flush=True)


def run(run_a: str, run_b: str | None, metric: str | None) -> None:
    _set_uri()
    results = analyze(run_a, run_b, metric)
    if not results:
        print("No step-level metrics found. Check that run_a is a valid MLflow run ID.")
        return
    print_report(results, run_a, run_b)
    log_to_mlflow(results, run_a, run_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="EL5: paired statistical significance tests → MLflow")
    parser.add_argument("--run_a", required=True, help="MLflow run ID for condition A")
    parser.add_argument("--run_b", default=None, help="MLflow run ID for condition B (omit for single-run CIs)")
    parser.add_argument("--metric", default=None, help="Single metric key (default: all step-level metrics)")
    args = parser.parse_args()
    run(args.run_a, args.run_b, args.metric)


if __name__ == "__main__":
    main()
