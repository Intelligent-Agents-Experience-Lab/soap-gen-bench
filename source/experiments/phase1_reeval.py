"""EL3 — Unified-judge re-evaluation of Phase 1 using RAGAS judges.

Addresses limitation: cross-phase judge inconsistency.
Phase 1 originally uses a local Mistral-7B correctness judge; Phase 2/3 use
Llama-3.3-70b-versatile via API (RAGAS-aligned binary judges). This makes the
three-phase trajectory table unreliable. This experiment re-runs Phase 1 generation
and scores it with the same RAGAS judges used in Phase 2/3, enabling direct
comparison of answer_relevancy and faithfulness across all phases.

Self-evaluation bias is also removed: Mistral-7B is no longer grading its own output.

Run:
    python -m source.experiments.phase1_reeval --num_examples 50
    python -m source.experiments.phase1_reeval --group slm --num_examples 50
    python -m source.experiments.phase1_reeval --models Qwen/Qwen2.5-3B-Instruct --num_examples 50

MLflow experiment: SOAP_Phase1_UnifiedJudge
Run name pattern:  P1R_{ModelShortName}__{PromptName}
"""
import argparse

from ..config import MLFLOW_EXPERIMENT_REEVAL, MODEL_GROUPS
from ..data.dataset import load_examples
from ..evaluation.metrics import RAGAS_JUDGES, STANDARD_SUITE
from ..models.llm import manager
from ..prompts.phase1 import PROMPTS, format_prompt
from ..tracking.mlflow_logger import evaluate_and_log


def run(
    model_names: list[str],
    prompt_filter: str | None,
    num_examples: int,
    start_prompt: str | None = None,
) -> None:
    examples = load_examples(num_examples)
    print(f"Loaded {len(examples)} examples.", flush=True)

    prompts = dict(PROMPTS)
    if prompt_filter:
        prompts = {k: v for k, v in prompts.items() if k == prompt_filter}
    if start_prompt and start_prompt in prompts:
        keys = list(prompts)
        prompts = {k: prompts[k] for k in keys[keys.index(start_prompt):]}

    for model_name in model_names:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}", flush=True)

        try:
            manager.load(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}", flush=True)
            continue

        for prompt_name in prompts:
            print(f"\n  Prompt: {prompt_name}", flush=True)

            def target(inputs: dict, _pname=prompt_name) -> dict:
                prompt = format_prompt(_pname, inputs["conversation"])
                # context="" so contextual RAGAS judges (precision/recall/relevancy) return None
                return {
                    "output": manager.generate([{"role": "user", "content": prompt}]),
                    "context": "",
                }

            evaluate_and_log(
                experiment_name=MLFLOW_EXPERIMENT_REEVAL,
                run_name=f"P1R_{model_name.split('/')[-1]}__{prompt_name}",
                params={
                    "model": model_name,
                    "prompt": prompt_name,
                    "judge": "RAGAS (unified, same as Phase 2/3)",
                    "n_examples": len(examples),
                },
                target_fn=target,
                examples=examples,
                evaluators=STANDARD_SUITE,
                judges=RAGAS_JUDGES,
            )

        manager.unload()

    print("\nPhase 1 unified-judge re-evaluation complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EL3: Phase 1 re-evaluation with unified RAGAS judge → MLflow"
    )
    parser.add_argument("--models", nargs="+", help="Specific model IDs")
    parser.add_argument("--group", choices=["slm", "llm", "all"], default="all")
    parser.add_argument("--prompt", help="Run only this prompt (e.g. H5_CoT)")
    parser.add_argument("--start_prompt", help="Resume from this prompt")
    parser.add_argument("--num_examples", type=int, default=50)
    args = parser.parse_args()

    if args.models:
        models = args.models
    elif args.group == "all":
        models = MODEL_GROUPS["slm"] + MODEL_GROUPS["llm"]
    else:
        models = MODEL_GROUPS[args.group]

    run(models, args.prompt, args.num_examples, args.start_prompt)


if __name__ == "__main__":
    main()
