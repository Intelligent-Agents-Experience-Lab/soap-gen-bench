"""Evaluators shared across all three phases.

Regular evaluators : (run, example)            → {"key", "score"}
Judge evaluators   : (run, example, trace_id)  → {"key", "score"}  — also call mlflow.log_feedback
"""
import json
from typing import Any

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

try:
    import evaluate as _hf_evaluate
    _HAVE_EVALUATE = True
except ImportError:
    _HAVE_EVALUATE = False

# Lazy metric cache — evaluate.load() is called once per metric per process
_HF: dict[str, Any] = {}

def _metric(name: str) -> Any | None:
    if not _HAVE_EVALUATE:
        return None
    if name not in _HF:
        _HF[name] = _hf_evaluate.load(name)
    return _HF[name]


# ---------------------------------------------------------------------------
# Shared OpenAI judge helper
# ---------------------------------------------------------------------------

def _openai_judge(
    metric_name: str,
    user_prompt: str,
    model_key: str = "phase1",  # "phase1" → OPENAI_JUDGE_MODEL, "phase2" → PHASE2_JUDGE_MODEL
    trace_id: str | None = None,
    max_tokens: int = 80,
) -> dict:
    """Call OpenAI, parse {"score": float, "reason": "..."}, log feedback, return result dict."""
    import openai
    from ..config import OPENAI_API_KEY, OPENAI_JUDGE_MODEL, PHASE2_JUDGE_MODEL

    model = PHASE2_JUDGE_MODEL if model_key == "phase2" else OPENAI_JUDGE_MODEL
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    score, reason = 0.0, ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip().lstrip("json").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0))
        reason = str(data.get("reason", ""))
    except Exception as e:
        reason = str(e)

    if trace_id:
        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=metric_name,
                value=score,
                rationale=reason or None,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE,
                    source_id=model,
                ),
            )
        except Exception:
            pass

    return {"key": metric_name, "score": score, "comment": reason}


# ---------------------------------------------------------------------------
# 1. JSON validity
# ---------------------------------------------------------------------------

def json_valid(run: Any, example: Any) -> dict:
    raw = run.outputs.get("output", "")
    try:
        text = raw.strip().removeprefix("```json").removesuffix("```").strip() if isinstance(raw, str) else ""
        json.loads(text)
        return {"key": "json_valid", "score": 1}
    except Exception:
        return {"key": "json_valid", "score": 0}


# ---------------------------------------------------------------------------
# 2. Section completeness (heuristic)
# ---------------------------------------------------------------------------

def completeness(run: Any, example: Any) -> dict:
    text = str(run.outputs.get("output", "")).lower()
    found = [s for s in ("subjective", "objective", "assessment", "plan") if s in text]
    return {"key": "completeness", "score": len(found) / 4.0, "comment": f"Found: {found}"}


# ---------------------------------------------------------------------------
# 3. Correctness — Phase 1 LLM judge (continuous [0,1])
# ---------------------------------------------------------------------------

def correctness(run: Any, example: Any, trace_id: str | None = None) -> dict:
    convo = example.inputs.get("conversation", "")
    soap = run.outputs.get("output", "")
    return _openai_judge(
        metric_name="correctness",
        user_prompt=(
            "You are a medical evaluator. Rate the factual correctness of this SOAP note "
            "against the clinical conversation (0.0 = major errors, 1.0 = fully correct).\n\n"
            f"Conversation:\n{convo}\n\nSOAP Note:\n{soap}\n\n"
            'Respond ONLY with JSON: {"score": 0.8, "reason": "..."}'
        ),
        model_key="phase1",
        trace_id=trace_id,
        max_tokens=150,
    )


# ---------------------------------------------------------------------------
# 4–8. RAGAS-aligned binary judges (Phase 2 / Phase 3)
#
#  Each returns 0.0 or 1.0 per case; mean over cases is the reported metric.
#  Context-dependent judges (precision, recall, relevancy) return {"score": None}
#  and are skipped when no retrieved context is available.
# ---------------------------------------------------------------------------

def answer_relevancy(run: Any, example: Any, trace_id: str | None = None) -> dict:
    """Does the SOAP note directly answer the patient's clinical concerns?"""
    convo = example.inputs.get("conversation", "")
    soap = run.outputs.get("output", "")
    return _openai_judge(
        metric_name="answer_relevancy",
        user_prompt=(
            "Evaluate whether this SOAP note directly addresses the patient's clinical "
            "concerns from the conversation. Score 1 if it clearly answers the patient's "
            "stated concerns and clinical questions, 0 if it is off-topic or misses key concerns.\n\n"
            f"Conversation:\n{convo}\n\nSOAP Note:\n{soap}\n\n"
            'Respond ONLY with JSON: {"score": 1, "reason": "..."}'
        ),
        model_key="phase2",
        trace_id=trace_id,
    )


def faithfulness(run: Any, example: Any, trace_id: str | None = None) -> dict:
    """Are all clinical claims in the SOAP note grounded in the source evidence?"""
    convo = example.inputs.get("conversation", "")
    soap = run.outputs.get("output", "")
    context = run.outputs.get("context", "")
    grounding = context if context else convo  # fall back to conversation when no RAG context
    return _openai_judge(
        metric_name="faithfulness",
        user_prompt=(
            "Evaluate whether every clinical claim in this SOAP note is supported by the "
            "provided source evidence. Score 1 if all claims are grounded (no fabrications, "
            "no unsupported numbers or diagnoses), 0 if any claim lacks source support.\n\n"
            f"Source evidence:\n{grounding}\n\nSOAP Note:\n{soap}\n\n"
            'Respond ONLY with JSON: {"score": 1, "reason": "..."}'
        ),
        model_key="phase2",
        trace_id=trace_id,
    )


def contextual_precision(run: Any, example: Any, trace_id: str | None = None) -> dict:
    """Are the retrieved chunks precise — minimal irrelevant content?"""
    context = run.outputs.get("context", "")
    if not context:
        return {"key": "contextual_precision", "score": None, "comment": "no context"}
    convo = example.inputs.get("conversation", "")
    return _openai_judge(
        metric_name="contextual_precision",
        user_prompt=(
            "Evaluate whether the retrieved context chunks are precise and relevant for "
            "generating a SOAP note for this patient. Score 1 if most retrieved chunks are "
            "clinically relevant to this case, 0 if many chunks are irrelevant or noisy.\n\n"
            f"Patient case:\n{convo}\n\nRetrieved context:\n{context}\n\n"
            'Respond ONLY with JSON: {"score": 1, "reason": "..."}'
        ),
        model_key="phase2",
        trace_id=trace_id,
    )


def contextual_recall(run: Any, example: Any, trace_id: str | None = None) -> dict:
    """Does the retrieved context cover all facts needed for a complete SOAP note?"""
    context = run.outputs.get("context", "")
    if not context:
        return {"key": "contextual_recall", "score": None, "comment": "no context"}
    ref = example.outputs.get("reference_soap", "")
    return _openai_judge(
        metric_name="contextual_recall",
        user_prompt=(
            "Evaluate whether the retrieved context contains all the clinical information "
            "needed to support the reference SOAP note. Score 1 if all key facts in the "
            "reference note are present in the retrieved context, 0 if important facts are missing.\n\n"
            f"Reference SOAP note:\n{ref}\n\nRetrieved context:\n{context}\n\n"
            'Respond ONLY with JSON: {"score": 1, "reason": "..."}'
        ),
        model_key="phase2",
        trace_id=trace_id,
    )


def contextual_relevancy(run: Any, example: Any, trace_id: str | None = None) -> dict:
    """Is the retrieved context relevant to this specific patient case?"""
    context = run.outputs.get("context", "")
    if not context:
        return {"key": "contextual_relevancy", "score": None, "comment": "no context"}
    convo = example.inputs.get("conversation", "")
    return _openai_judge(
        metric_name="contextual_relevancy",
        user_prompt=(
            "Evaluate whether the retrieved context is relevant to this patient's clinical "
            "case. Score 1 if the retrieved information is directly applicable to generating "
            "a SOAP note for this patient, 0 if it is mostly unrelated.\n\n"
            f"Patient case:\n{convo}\n\nRetrieved context:\n{context}\n\n"
            'Respond ONLY with JSON: {"score": 1, "reason": "..."}'
        ),
        model_key="phase2",
        trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# 9. BLEU
# ---------------------------------------------------------------------------

def bleu(run: Any, example: Any) -> dict:
    m = _metric("bleu")
    if m is None:
        return {"key": "bleu", "score": 0, "comment": "evaluate not installed"}
    ref = example.outputs.get("reference_soap") or ""
    hyp = run.outputs.get("output") or ""
    if not ref.strip():
        return {"key": "bleu", "score": 0}
    try:
        result = m.compute(predictions=[hyp], references=[[ref]])
        return {"key": "bleu", "score": float(result["bleu"])}
    except Exception as e:
        return {"key": "bleu", "score": 0, "comment": str(e)}


# ---------------------------------------------------------------------------
# 10. ROUGE-L
# ---------------------------------------------------------------------------

def rouge(run: Any, example: Any) -> dict:
    m = _metric("rouge")
    if m is None:
        return {"key": "rouge", "score": 0, "comment": "evaluate not installed"}
    ref = example.outputs.get("reference_soap") or ""
    hyp = run.outputs.get("output") or ""
    try:
        result = m.compute(predictions=[hyp], references=[ref])
        return {"key": "rouge", "score": float(result["rougeL"])}
    except Exception as e:
        return {"key": "rouge", "score": 0, "comment": str(e)}


# ---------------------------------------------------------------------------
# 11. METEOR
# ---------------------------------------------------------------------------

def meteor(run: Any, example: Any) -> dict:
    m = _metric("meteor")
    if m is None:
        return {"key": "meteor", "score": 0, "comment": "evaluate not installed"}
    ref = example.outputs.get("reference_soap") or ""
    hyp = run.outputs.get("output") or ""
    if not ref.strip():
        return {"key": "meteor", "score": 0}
    try:
        result = m.compute(predictions=[hyp], references=[ref])
        return {"key": "meteor", "score": float(result["meteor"])}
    except Exception as e:
        return {"key": "meteor", "score": 0, "comment": str(e)}


# ---------------------------------------------------------------------------
# 12. BERTScore (semantic similarity — F1 of contextual token overlap)
# ---------------------------------------------------------------------------

def bertscore(run: Any, example: Any) -> dict:
    m = _metric("bertscore")
    if m is None:
        return {"key": "bertscore", "score": 0, "comment": "evaluate not installed"}
    ref = example.outputs.get("reference_soap") or ""
    hyp = run.outputs.get("output") or ""
    if not ref.strip():
        return {"key": "bertscore", "score": 0}
    try:
        result = m.compute(predictions=[hyp], references=[ref], lang="en",
                           model_type="distilbert-base-uncased")
        return {"key": "bertscore", "score": float(result["f1"][0])}
    except Exception as e:
        return {"key": "bertscore", "score": 0, "comment": str(e)}


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

# Deterministic surface metrics (no API required)
STANDARD_SUITE = [completeness, bleu, rouge, meteor, bertscore]

# Phase 1: continuous correctness judge via OpenAI
PHASE1_JUDGES = [correctness]

# Phase 2 / 3: RAGAS-aligned binary judges via OpenAI (PHASE2_JUDGE_MODEL)
RAGAS_JUDGES = [
    answer_relevancy,
    faithfulness,
    contextual_precision,
    contextual_recall,
    contextual_relevancy,
]

# Legacy alias kept for any callers still importing JUDGE_SUITE
JUDGE_SUITE = PHASE1_JUDGES
