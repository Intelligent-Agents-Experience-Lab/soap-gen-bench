"""Phase 2: RAG ablation studies (E1–E4 + SLM–LLM showdown) logged to MLflow.

Run:
    python -m source.experiments.phase2 --experiment e2_hybrid --num_examples 30
    python -m source.experiments.phase2 --all --num_examples 30
    python -m source.experiments.phase2 --experiment e_showdown_slm --num_examples 30
    python -m source.experiments.phase2 --experiment e_showdown_llm --num_examples 30

Results in MLflow experiment "SOAP_Phase2_RAG_Ablation".
Start the UI: mlflow ui  →  http://127.0.0.1:5000
"""
import argparse
from typing import List, Dict

from ..config import (
    MLFLOW_EXPERIMENT_P2,
    RAG_MODEL,
    SHOWDOWN_LLM_MODEL,
    SHOWDOWN_SLM_MODEL,
)
from ..data.dataset import load_examples
from ..evaluation.metrics import RAGAS_JUDGES, STANDARD_SUITE
from ..models.llm import manager
from ..prompts.rag import build_prompt
from ..rag.query import QueryEngine
from ..rag.retriever import RetrievalEngine
from ..tracking.mlflow_logger import evaluate_and_log

# ---------------------------------------------------------------------------
# Named experiment configurations
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    # E1: RAG vs NoRAG sanity check (dense, section-level, k=4)
    "e1":             {"retriever_type": "dense",   "index_name": "index_b_section",  "query_strategy": "raw",     "k": 4},
    # E2: Retriever type ablation (section-level index fixed)
    "e2_sparse":      {"retriever_type": "sparse",  "index_name": "index_b_section",  "query_strategy": "raw",     "k": 8},
    "e2_dense":       {"retriever_type": "dense",   "index_name": "index_b_section",  "query_strategy": "raw",     "k": 8},
    "e2_hybrid":      {"retriever_type": "hybrid",  "index_name": "index_b_section",  "query_strategy": "raw",     "k": 8},
    # E3: Granularity ablation (hybrid RRF fixed)
    "e3_note":        {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "raw",    "k": 8},
    "e3_section":     {"retriever_type": "hybrid",  "index_name": "index_b_section",   "query_strategy": "raw",    "k": 8},
    "e3_fixed":       {"retriever_type": "hybrid",  "index_name": "index_fixed",       "query_strategy": "raw",    "k": 8},
    "e3_struct":      {"retriever_type": "hybrid",  "index_name": "index_struct_aware","query_strategy": "raw",    "k": 8},
    # E4: Query strategy ablation (hybrid RRF + note-level fixed — champion config)
    "e4_raw":         {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "raw",     "k": 8},
    "e4_rewrite":     {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "rewrite", "k": 8},
    "e4_multi":       {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "multi",   "k": 8},
    "e4_section":     {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "section", "k": 8},
    # SLM–LLM showdown — champion config (hybrid + note-level + section-wise, k=8)
    # Both models run locally via ModelManager. Override SHOWDOWN_SLM_MODEL /
    # SHOWDOWN_LLM_MODEL env vars to change the models.
    "e_showdown_slm": {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "section", "k": 8,
                       "model_override": SHOWDOWN_SLM_MODEL},
    "e_showdown_llm": {"retriever_type": "hybrid",  "index_name": "index_a_note",      "query_strategy": "section", "k": 8,
                       "model_override": SHOWDOWN_LLM_MODEL},
}


# ---------------------------------------------------------------------------
# Unified RAG pipeline
# ---------------------------------------------------------------------------

class RagPipeline:
    def __init__(
        self,
        retriever_type: str = "hybrid",
        index_name: str = "index_b_section",
        query_strategy: str = "raw",
        k: int = 8,
        model_override: str | None = None,  # if set, load this model instead of RAG_MODEL
    ):
        self.retriever = RetrievalEngine(index_name=index_name)
        self.query_engine = QueryEngine()
        self.retriever_type = retriever_type
        self.query_strategy = query_strategy
        self.k = k
        self.model_override = model_override

    def retrieve(self, conversation: str) -> List[Dict]:
        queries = self._queries(conversation)
        raw: List[Dict] = []
        for q in queries:
            if self.retriever_type == "sparse":
                raw.extend(self.retriever.search_sparse(q, k=self.k))
            elif self.retriever_type == "dense":
                raw.extend(self.retriever.search_dense(q, k=self.k))
            else:
                raw.extend(self.retriever.search_hybrid(q, k=self.k))
        return self._dedup(raw)

    def _queries(self, conversation: str) -> List[str]:
        if self.query_strategy == "rewrite":
            return [self.query_engine.rewrite(conversation)]
        if self.query_strategy == "multi":
            return self.query_engine.multi_query(conversation)
        if self.query_strategy == "section":
            return self.query_engine.section_queries(conversation)
        return [conversation]

    def _dedup(self, results: List[Dict]) -> List[Dict]:
        seen: dict = {}
        for r in results:
            uid = (r["metadata"]["source_id"], r["metadata"]["content"][:50])
            if uid not in seen or r["score"] > seen[uid]["score"]:
                seen[uid] = r
        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[: self.k]

    def __call__(self, inputs: dict) -> dict:
        conversation = inputs.get("conversation", "")
        chunks = self.retrieve(conversation)
        prompt = build_prompt(conversation, chunks)
        output = manager.generate([{"role": "user", "content": prompt}])
        # Pack retrieved text for RAGAS judges
        context = "\n\n".join(c["metadata"].get("content", "") for c in chunks)
        return {"output": output, "context": context}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(experiment_key: str, num_examples: int) -> None:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment_key!r}. Choose from: {list(EXPERIMENTS)}")

    cfg = dict(EXPERIMENTS[experiment_key])
    model_override = cfg.pop("model_override", None)
    gen_model = model_override or RAG_MODEL

    print(f"\nPhase 2 — {experiment_key}: {cfg}  model={gen_model}", flush=True)
    examples = load_examples(num_examples)

    manager.load(gen_model)
    pipeline = RagPipeline(**cfg, model_override=model_override)

    try:
        evaluate_and_log(
            experiment_name=MLFLOW_EXPERIMENT_P2,
            run_name=f"P2_{experiment_key}",
            params={"model": gen_model, "experiment": experiment_key, **cfg, "n_examples": len(examples)},
            target_fn=pipeline,
            examples=examples,
            evaluators=STANDARD_SUITE,
            judges=RAGAS_JUDGES,
        )
    finally:
        manager.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: RAG ablation → MLflow")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", choices=list(EXPERIMENTS), help="Single named experiment")
    group.add_argument("--all", action="store_true", help="Run all experiments sequentially")
    parser.add_argument("--num_examples", type=int, default=30)
    args = parser.parse_args()

    keys = list(EXPERIMENTS) if args.all else [args.experiment]
    for key in keys:
        run(key, args.num_examples)

    print("\nPhase 2 complete.", flush=True)


if __name__ == "__main__":
    main()
