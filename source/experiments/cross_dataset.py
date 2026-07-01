"""EL1 — Cross-dataset generalization: run Phase 1 prompts on external SOAP datasets.

Addresses limitation: single-corpus evaluation (adesouza1/soap_notes only).

Run:
    python -m source.experiments.cross_dataset --num_examples 50
    python -m source.experiments.cross_dataset --dataset omi_health --num_examples 50
    python -m source.experiments.cross_dataset --prompts H3_FewShot H5_CoT --num_examples 50

Datasets (from config.CROSS_DATASETS):
    omi_health        — omi-health/medical-dialogue-to-soap-summary  (10K, test split)
    augmented_clinical — AGBonnet/augmented-clinical-notes            (30K, train split)
    subash_soap       — SubashNeupane/dataset_SOAP_summary            (1.5K, train split)
    rhyliieee_soap    — rhyliieee/soap-convo-v2                      (1K, train split)

MLflow experiment: SOAP_CrossDataset_Generalization
Run name pattern:  XD_{dataset_key}__{ModelShortName}__{PromptName}
"""
import argparse

from ..config import CROSS_DATASETS, MLFLOW_EXPERIMENT_CROSS, MODEL_GROUPS
from ..data.dataset import load_examples_from_hf
from ..evaluation.metrics import PHASE1_JUDGES, STANDARD_SUITE
from ..models.llm import manager
from ..prompts.phase1 import PROMPTS, format_prompt
from ..tracking.mlflow_logger import evaluate_and_log

# omi-health has a predefined test split; others use train
_DATASET_SPLITS: dict[str, str] = {
    "omi_health": "test",
}


def run(
    dataset_keys: list[str],
    model_names: list[str],
    prompt_keys: list[str],
    num_examples: int,
) -> None:
    for dataset_key in dataset_keys:
        dataset_id = CROSS_DATASETS[dataset_key]
        split = _DATASET_SPLITS.get(dataset_key, "train")
        print(f"\n{'='*60}\nDataset: {dataset_key} ({dataset_id}, split={split})\n{'='*60}", flush=True)

        try:
            examples = load_examples_from_hf(dataset_id, n=num_examples, split=split)
        except Exception as e:
            print(f"Failed to load {dataset_id}: {e}", flush=True)
            continue

        print(f"Loaded {len(examples)} examples.", flush=True)

        for model_name in model_names:
            print(f"\n  Model: {model_name}", flush=True)
            try:
                manager.load(model_name)
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}", flush=True)
                continue

            for prompt_name in prompt_keys:
                print(f"    Prompt: {prompt_name}", flush=True)

                def target(inputs: dict, _pname=prompt_name) -> dict:
                    prompt = format_prompt(_pname, inputs["conversation"])
                    return {"output": manager.generate([{"role": "user", "content": prompt}])}

                evaluate_and_log(
                    experiment_name=MLFLOW_EXPERIMENT_CROSS,
                    run_name=f"XD_{dataset_key}__{model_name.split('/')[-1]}__{prompt_name}",
                    params={
                        "dataset": dataset_id,
                        "dataset_key": dataset_key,
                        "split": split,
                        "model": model_name,
                        "prompt": prompt_name,
                        "n_examples": len(examples),
                    },
                    target_fn=target,
                    examples=examples,
                    evaluators=STANDARD_SUITE,
                    judges=PHASE1_JUDGES,
                )

            manager.unload()

    print("\nCross-dataset experiment complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EL1: cross-dataset generalization → MLflow")
    parser.add_argument(
        "--dataset",
        choices=list(CROSS_DATASETS),
        default=None,
        help="Single dataset key (default: all)",
    )
    parser.add_argument("--models", nargs="+", help="Specific model IDs (default: all reported models)")
    parser.add_argument("--group", choices=["slm", "llm", "all"], default="all")
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=list(PROMPTS),
        default=["H3_FewShot", "H5_CoT"],
        help="Prompt keys to run (default: H3_FewShot H5_CoT)",
    )
    parser.add_argument("--num_examples", type=int, default=50)
    args = parser.parse_args()

    dataset_keys = [args.dataset] if args.dataset else list(CROSS_DATASETS)

    if args.models:
        models = args.models
    elif args.group == "all":
        models = MODEL_GROUPS["slm"] + MODEL_GROUPS["llm"]
    else:
        models = MODEL_GROUPS[args.group]

    run(dataset_keys, models, args.prompts, args.num_examples)


if __name__ == "__main__":
    main()
