"""EL6 — Systematic hallucination audit with typed severity classification.

Addresses limitation: 15-case single-reviewer non-blinded audit, no inter-annotator
agreement statistics (Cohen's κ).

For each case the audit:
  1. Generates a SOAP note with the specified model
  2. Runs an LLM judge as Annotator 1 (accuracy-focused framing)
  3. Runs the same LLM judge as Annotator 2 (patient-safety framing) when --annotators 2
  4. Saves per-case results to CSV
  5. Computes Cohen's κ on the hallucination-type classification (ann1 vs ann2)
  6. Logs summary statistics to MLflow

Hallucination taxonomy:
  Type       → none | commission | omission | both
  Commission → lab_fabrication | vital_fabrication | demographic | medication_error | unsupported_diagnosis
  Omission   → medication_gap | allergy_gap | objective_gap | plan_omission | symptom_gap
  Severity   → critical | moderate | minor  (null when type=none)

Run:
    python -m source.experiments.hallucination_audit --num_examples 50
    python -m source.experiments.hallucination_audit --num_examples 50 --annotators 2
    python -m source.experiments.hallucination_audit --model Qwen/Qwen2.5-3B-Instruct --num_examples 30
    python -m source.experiments.hallucination_audit --output_csv results/audit.csv --num_examples 50

MLflow experiment: SOAP_Hallucination_Audit
Run name pattern:  HALL_{ModelShortName}_{n}cases
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mlflow

from ..config import (
    MLFLOW_EXPERIMENT_HALLUCINATION,
    MLFLOW_TRACKING_URI,
    OPENAI_API_KEY,
    PHASE2_JUDGE_MODEL,
    MODEL_GROUPS,
)
from ..data.dataset import load_examples
from ..models.llm import manager
from ..prompts.phase1 import format_prompt

_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_DEFAULT_PROMPT = "H3_FewShot"


# ---------------------------------------------------------------------------
# Hallucination judge prompts
# ---------------------------------------------------------------------------

_TAXONOMY_SPEC = """\
Hallucination taxonomy:
  type: "none" | "commission" | "omission" | "both"
  commission_subtype: null | "lab_fabrication" | "vital_fabrication" | "demographic" |
                      "medication_error" | "unsupported_diagnosis"
  omission_subtype:   null | "medication_gap" | "allergy_gap" | "objective_gap" |
                      "plan_omission" | "symptom_gap"
  severity: null | "critical" | "moderate" | "minor"
  evidence: one-sentence description of the specific error (or "none")
  affected_section: null | "subjective" | "objective" | "assessment" | "plan"
"""

_PROMPT_ANN1 = """\
You are a medical accuracy auditor reviewing AI-generated SOAP notes.
Compare the generated SOAP note against the clinical conversation and reference note.
Identify any hallucinations — content that is fabricated (commission) or clinically
necessary content that is absent (omission).

{taxonomy}

Clinical conversation:
{conversation}

Reference SOAP note:
{reference}

Generated SOAP note:
{generated}

Respond ONLY with valid JSON matching the taxonomy above:
{{"type": "...", "commission_subtype": ..., "omission_subtype": ...,
  "severity": ..., "evidence": "...", "affected_section": ...}}"""

_PROMPT_ANN2 = """\
You are a patient safety specialist reviewing AI-generated clinical documentation.
From a patient safety perspective, identify any clinical information that is incorrect
(commission hallucination) or dangerously absent (omission hallucination) in the
generated SOAP note compared to the conversation and reference.

{taxonomy}

Clinical conversation:
{conversation}

Reference SOAP note:
{reference}

Generated SOAP note:
{generated}

Respond ONLY with valid JSON matching the taxonomy above:
{{"type": "...", "commission_subtype": ..., "omission_subtype": ...,
  "severity": ..., "evidence": "...", "affected_section": ...}}"""


def _run_judge(prompt_template: str, conversation: str, reference: str, generated: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = prompt_template.format(
        taxonomy=_TAXONOMY_SPEC,
        conversation=conversation[:2000],
        reference=reference[:1500],
        generated=generated[:1500],
    )
    try:
        resp = client.chat.completions.create(
            model=PHASE2_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip().lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "type": "error", "commission_subtype": None, "omission_subtype": None,
            "severity": None, "evidence": str(e), "affected_section": None,
        }


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    classes = sorted(set(labels_a) | set(labels_b))
    conf: dict[tuple, int] = {}
    for c1 in classes:
        for c2 in classes:
            conf[(c1, c2)] = 0
    for a, b in zip(labels_a, labels_b):
        conf[(a, b)] += 1
    p_o = sum(conf[(c, c)] for c in classes) / n
    p_e = sum(
        (sum(conf[(c, x)] for x in classes) / n) * (sum(conf[(x, c)] for x in classes) / n)
        for c in classes
    )
    return (p_o - p_e) / (1 - p_e) if p_e < 1.0 else 0.0


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------

def run(
    model_name: str,
    num_examples: int,
    annotators: int,
    output_csv: str,
) -> None:
    examples = load_examples(num_examples)
    print(f"Loaded {len(examples)} examples.", flush=True)

    manager.load(model_name)

    rows: list[dict] = []
    ann1_types: list[str] = []
    ann2_types: list[str] = []

    for i, ex in enumerate(examples):
        conversation = ex["inputs"].get("conversation", "")
        reference = ex["outputs"].get("reference_soap", "")

        prompt = format_prompt(_DEFAULT_PROMPT, conversation)
        generated = manager.generate([{"role": "user", "content": prompt}])

        print(f"  [{i+1}/{len(examples)}] auditing...", end=" ", flush=True)

        ann1 = _run_judge(_PROMPT_ANN1, conversation, reference, generated)
        ann1_types.append(ann1.get("type", "error"))

        row: dict = {
            "case_id": i,
            "conversation": conversation[:200],
            "generated_soap": generated[:300],
            "reference_soap": reference[:300],
            "ann1_type": ann1.get("type"),
            "ann1_commission_subtype": ann1.get("commission_subtype"),
            "ann1_omission_subtype": ann1.get("omission_subtype"),
            "ann1_severity": ann1.get("severity"),
            "ann1_evidence": ann1.get("evidence"),
            "ann1_affected_section": ann1.get("affected_section"),
        }

        if annotators >= 2:
            ann2 = _run_judge(_PROMPT_ANN2, conversation, reference, generated)
            ann2_types.append(ann2.get("type", "error"))
            row.update({
                "ann2_type": ann2.get("type"),
                "ann2_commission_subtype": ann2.get("commission_subtype"),
                "ann2_omission_subtype": ann2.get("omission_subtype"),
                "ann2_severity": ann2.get("severity"),
                "ann2_evidence": ann2.get("evidence"),
                "ann2_affected_section": ann2.get("affected_section"),
            })

        rows.append(row)
        print(f"type={ann1.get('type')}  severity={ann1.get('severity')}", flush=True)

    manager.unload()

    # --- Save CSV ---
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAudit CSV saved: {output_csv}", flush=True)

    # --- Compute summary stats ---
    n = len(rows)
    valid_types = [r["ann1_type"] for r in rows if r["ann1_type"] not in (None, "error")]
    hall_rate = sum(1 for t in valid_types if t != "none") / len(valid_types) if valid_types else 0.0
    commission_rate = sum(1 for t in valid_types if t in ("commission", "both")) / len(valid_types) if valid_types else 0.0
    omission_rate = sum(1 for t in valid_types if t in ("omission", "both")) / len(valid_types) if valid_types else 0.0
    critical_rate = sum(1 for r in rows if r.get("ann1_severity") == "critical") / n if n else 0.0

    kappa = None
    if annotators >= 2 and len(ann1_types) == len(ann2_types) and ann2_types:
        kappa = cohens_kappa(ann1_types, ann2_types)
        print(f"Cohen's κ (type classification): {kappa:.4f}", flush=True)

    print(f"\nSummary (n={n}):")
    print(f"  Hallucination rate:  {hall_rate:.3f}")
    print(f"  Commission rate:     {commission_rate:.3f}")
    print(f"  Omission rate:       {omission_rate:.3f}")
    print(f"  Critical severity:   {critical_rate:.3f}")
    if kappa is not None:
        print(f"  Cohen's κ:           {kappa:.4f}")

    # --- Log to MLflow ---
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_HALLUCINATION)
    run_name = f"HALL_{model_name.split('/')[-1]}_{n}cases"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name,
            "prompt": _DEFAULT_PROMPT,
            "annotators": annotators,
            "n_cases": n,
            "output_csv": output_csv,
        })
        metrics = {
            "hallucination_rate": hall_rate,
            "commission_rate": commission_rate,
            "omission_rate": omission_rate,
            "critical_rate": critical_rate,
        }
        if kappa is not None:
            metrics["cohens_kappa"] = kappa
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(output_csv)
    print(f"Logged to MLflow: {MLFLOW_EXPERIMENT_HALLUCINATION}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EL6: systematic hallucination audit → CSV + MLflow")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Generator model HF ID")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument(
        "--annotators", type=int, choices=[1, 2], default=1,
        help="1 = single pass, 2 = two annotator passes for Cohen's κ",
    )
    parser.add_argument("--output_csv", default="results/hallucination_audit.csv")
    args = parser.parse_args()
    run(args.model, args.num_examples, args.annotators, args.output_csv)


if __name__ == "__main__":
    main()
