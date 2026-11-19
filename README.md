# `source/` — code for the SOAP note generation study

This package contains all code for the article *Retrieval and Fine-Tuning for Structured SOAP Note Generation: A Held-out Evaluation of Reference-Note Imitation and LLM-Judge Leniency*.

It has two parts:

1. **Held-out evaluation and ACI-BENCH replication.** These produce every result in the main article and in Supplemental Article S1, Parts A and C. They live in `experiments/revision_*.py`, `experiments/aci_*.py` and `experiments/jev_*.py`.
2. **Exploratory in-sample phases.** These produced the historical results in Supplemental Article S1, Part B: the Phase 1 prompting ablation, the Phase 2 retrieval components, Phase 3 LoRA, and extension analyses. The code is retained for transparency. These analyses evaluated rows that were also in their retrieval corpus and fine-tuning data, so they are not evidence of generalization.

All commands are run from the repository root (the folder that contains `source/` and `revision/`) as Python modules, for example `python -m source.experiments.revision_runner preflight`.

---

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r source/requirements.txt
```

Create a `.env` file in the repository root (it is git-ignored):

```
OPENAI_API_KEY=...        # gpt-4o-mini-2024-07-18 and gpt-5.6-luna judges
TYPESAFE_API_KEY=...      # TypeSafe Jev judge
```

**Hardware:** one CUDA GPU with 12 GB is enough for 4-bit generation and for LoRA training on the main dataset. ACI-BENCH LoRA training needs more memory; we used a Colab T4 (`experiments/aci_colab.py`).

**Reported environment:** Python 3.11, CUDA 12.4. The pinned package versions are in `requirements.txt`. Every generated note also records its package versions.

**Settings:** the held-out code reads its frozen settings from `revision/protocol/config.json` (main dataset) and `revision/protocol/config_aci.json` (ACI-BENCH). `config.py` in this folder is used only by the exploratory modules.

---

## 2. Held-out evaluation (main dataset)

**Data:** `adesouza1/soap_notes` at a pinned revision. Retrieval and training use the 558-row train split; evaluation uses the 62-row test split.

**Model:** `mistralai/Mistral-7B-Instruct-v0.3` (fixed commit), 4-bit NF4, greedy decoding.

**Arms:**

| Arm | Retrieval | LoRA adapter |
|---|---|---|
| `base` | none | none |
| `rag` | top-8 training notes | none |
| `lora` | none | yes |
| `lora_rag` | top-8 training notes | yes |
| `random_rag` | 8 random training notes | none |
| `schema` | none | none (JSON-schema constrained decoding) |

### Modules

| Module | What it does |
|---|---|
| `experiments/revision_core.py` | Dependency-light primitives: hashing, five-gram Jaccard similarity, SOAP header splitting, JSON validation, structural scores |
| `experiments/revision_prepare.py` | Downloads the pinned splits, builds the SOAP JSON targets, and writes the train/test separation (contamination) audit |
| `experiments/revision_runner.py` | Main driver. Actions: `preflight`, `smoke`, `train-smoke`, `retrieve-smoke`, `train` (LoRA r=16, 60 steps, seed 3407/3408/3409), `retrieve` (hybrid BM25 + MiniLM, reciprocal-rank fusion, contexts cached so RAG-only and RAG+LoRA see identical contexts), `generate --condition <arm>`, `judge` (binary source support and relevance, gpt-4o-mini or gpt-5.6-luna), `annotate` |
| `experiments/revision_rescore.py` | Rescoring with the frozen primary judge, and the reference-note control (each dataset reference note scored against its own conversation) |
| `experiments/jev_judge.py` | TypeSafe Jev scoring. Question design v1 (two questions) and v2 (single-scope support and relevance questions plus six defect checks). Probabilities are banded at 0.7 / 0.3 |
| `experiments/jev_statistics.py` | Merges and summarizes the Jev judgments for the six arms |
| `experiments/revision_measurement_audit.py`, `revision_measurement_audit_v2.py` | Judge-free numeric audit: measurements in a note that never appear in its conversation. v2 normalizes spoken numbers and excludes dates and ratios |
| `experiments/revision_statistics.py` | Paired inference: exact McNemar tests, case-bootstrap intervals, Holm adjustment within each measure, and the RAG×LoRA interaction |
| `experiments/revision_judge_comparison.py` | Combines all judges and the audit; computes Cohen's κ between measures and the Jev v2 defect counts |
| `experiments/revision_seed_sensitivity.py` | Trains seeds 3408 and 3409, regenerates both LoRA arms, scores them with every measure, and summarizes |
| `experiments/revision_reference_overlap.py` | ROUGE-1/2/L against the reference notes, and source-support contrasts restricted to structurally valid note pairs |
| `experiments/revision_human_validation.py` | Imports the two rater workbooks, checks them against the blinded packet, and computes inter-rater κ, the consensus reference, and judge sensitivity and specificity |
| `prompts/rag.py` | The generation prompt with and without retrieved notes (shared with Phase 2) |

### Order of execution

```bash
python -m source.experiments.revision_prepare
python -m source.experiments.revision_runner preflight
python -m source.experiments.revision_runner train --seed 3407
python -m source.experiments.revision_runner retrieve
python -m source.experiments.revision_runner generate --condition base        # repeat for each arm
python -m source.experiments.revision_rescore                                  # primary judge + reference control
python -m source.experiments.revision_runner judge --input <generations> --output <scores>   # gpt-5.6-luna (config judge)
python -m source.experiments.jev_judge score --input <generations> --output <scores>
python -m source.experiments.revision_measurement_audit_v2
python -m source.experiments.revision_statistics
python -m source.experiments.revision_judge_comparison
python -m source.experiments.revision_seed_sensitivity
python -m source.experiments.revision_reference_overlap
python -m source.experiments.revision_human_validation --rater A=<sheet.xlsx> --rater B=<sheet.xlsx>
```

Run any module with `--help` for its options. Outputs go to `revision/results/`. Steps skip or refuse to overwrite existing outputs, so finished work is not silently replaced.

---

## 3. ACI-BENCH replication

**Data:** ACI-BENCH (CC BY 4.0) at commit `b909b2bb`. The 87 train and validation encounters are used for retrieval and training; the 120 official test encounters (test1–test3) are used for evaluation.

**Design:** the same six arms and judges, with recorded adaptations for longer encounters: k = 3, larger token budgets, and batch size 1 × 8. The analysis plan and decision rule are in `revision/protocol/aci_analysis_plan.md`.

| Module | What it does |
|---|---|
| `experiments/aci_prepare.py` | Converts ACI-BENCH notes to the four SOAP fields using a deterministic header mapping, and writes the conversion audit |
| `experiments/aci_replication.py` | Resumable driver: freezes token budgets from the training split, trains, retrieves, generates and judges. `--skip-judges` and `--judge-only` run part of the pipeline |
| `experiments/aci_colab.py` | Runs only the GPU-heavy steps (LoRA training, LoRA-only and RAG+LoRA generation) on a cloud GPU |
| `experiments/aci_report.py` | Report with all arms, paired contrasts (Holm over the three prespecified comparisons; interaction exploratory), the locked decision rule, and the five-gram overlap with retrieved notes |

```bash
python -m source.experiments.aci_prepare
python -m source.experiments.aci_replication
python -m source.experiments.aci_report
```

The Windows launchers `run_aci_replication.bat`, `run_aci_judge_only.bat` and `run_seed_sensitivity.bat` in the repository root call the project virtual environment directly.

---

## 4. Exploratory in-sample phases (Supplemental Article S1, Part B)

These modules are kept as a record of the earlier analyses. Their evaluation rows came from the training split and were in both the retrieval index and the fine-tuning data. The judges also differed by phase, as stated in Supplemental Article S1, Section B.1:
- **Phase 1:** `gpt-4o-mini`, set in `config.py`.
- **Phases 2 and 3:** scored outside this package, in Colab and LangSmith notebooks.

| Module | Purpose |
|---|---|
| `config.py` | Model catalogue, MLflow experiment names, and judge and index settings for the modules below |
| `data/dataset.py` | Loads the Hugging Face dataset and exports SFT JSONL |
| `prompts/phase1.py` | Prompt strategies H1–H5 |
| `models/llm.py` | Loads one model at a time, generates, and unloads |
| `models/finetune.py` | Phase 3 LoRA fine-tuning of Mistral-7B with Unsloth (Colab) |
| `rag/indexer.py`, `rag/retriever.py`, `rag/query.py` | Phase 2 indices (four granularities); sparse, dense and hybrid RRF retrieval; query rewriting |
| `evaluation/metrics.py` | BLEU, ROUGE, METEOR, BERTScore, completeness and LLM-judge evaluators |
| `tracking/mlflow_logger.py` | MLflow logging with GenAI tracing |
| `experiments/phase1.py`, `phase2.py`, `phase3.py` | Phase 1 prompting ablation, Phase 2 retrieval ablations, Phase 3 LoRA evaluation |
| `experiments/phase1_reeval.py`, `cross_dataset.py`, `latency_benchmark.py`, `statistical_analysis.py`, `hallucination_audit.py` | Extension analyses (unified-judge re-evaluation, cross-dataset prompts, latency, pairwise tests, selected hallucination audit) |

---

## 5. Tests

```bash
python -m pytest tests
```

The tests cover the revision primitives, runner, statistics (including the ACI three-test Holm family), Jev parsing and the ACI conversion.

## 6. Data and privacy

No dataset text, generated notes, judge outputs or rater files are stored in this folder. They are written to `revision/results/`, which is excluded from version control apart from text-free summaries. API keys are read only from `.env`.
