"""Phase 1: prompt-engineering ablation (H1–H5) across SLMs and LLMs.

Run:
    python -m source.experiments.phase1 --group slm --num_examples 50
    python -m source.experiments.phase1 --models Qwen/Qwen2.5-3B-Instruct --prompt H5_CoT

Results are logged to MLflow (traces + judge scores). Start the UI first:
    mlflow ui  →  http://127.0.0.1:5000
"""
import argparse

from ..config import MLFLOW_EXPERIMENT_P1, MODEL_GROUPS
from ..data.dataset import load_examples
from ..evaluation.metrics import PHASE1_JUDGES, STANDARD_SUITE
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
    print(f"Loaded {len(examples)} examples from HuggingFace.", flush=True)

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
                return {"output": manager.generate([{"role": "user", "content": prompt}])}

            evaluate_and_log(
                experiment_name=MLFLOW_EXPERIMENT_P1,
                run_name=f"{model_name.split('/')[-1]}__{prompt_name}",
                params={"model": model_name, "prompt": prompt_name, "n_examples": len(examples)},
                target_fn=target,
                examples=examples,
                evaluators=STANDARD_SUITE,
                judges=PHASE1_JUDGES,
            )

        manager.unload()

    print("\nPhase 1 complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: prompt ablation → MLflow")
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
