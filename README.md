# From Prompting to Retrieval: A Benchmark for Structured Clinical SOAP Note Generation

A reproducible benchmark and experiment suite that systematically evaluates Small Language Models (SLMs) and Large Language Models (LLMs) for generating structured clinical **SOAP notes** (Subjective, Objective, Assessment, Plan) from doctor–patient dialogue. The study progresses from prompt engineering, through retrieval-augmented generation (RAG), to fine-tuning, and includes six supplementary experiments (EL1–EL6) that address dataset scope, judge consistency, deployment cost, statistical power, and hallucination safety.

All generator models run **locally**; an OpenAI model is used only as an LLM-as-judge. Every experiment is tracked with **MLflow** (Runs, GenAI Traces, and Judges tabs).

---

## Table of contents

1. [Description](#description)
2. [Requirements](#requirements)
3. [Dataset information](#dataset-information)
4. [Code information](#code-information)
5. [Usage instructions](#usage-instructions)
6. [Methodology](#methodology)
7. [Metrics reference](#metrics-reference)
8. [MLflow experiment index](#mlflow-experiment-index)
9. [Citations](#citations)
10. [License and contribution guidelines](#license-and-contribution-guidelines)

---

## Description

Clinical documentation is time-consuming and a leading contributor to clinician burnout. This project asks a practical question: **what is the most effective and deployable way to have a language model draft a structured SOAP note from a consultation transcript?** Rather than reporting a single model score, it runs a controlled, three-phase progression and measures how much each intervention actually helps.

- **Phase 1 — Prompt engineering ablation.** Five prompt strategies (H1–H5) across up to 10 open-weight models (1B–8B parameters).
- **Phase 2 — RAG ablation.** A factorial sweep over retriever type × index granularity × query strategy, followed by an SLM-vs-LLM showdown using the champion configuration.
- **Phase 3 — Fine-tuning evaluation.** An untuned base model vs. a QLoRA fine-tuned model vs. fine-tuning combined with RAG.
- **EL1–EL6 — Supplementary experiments.** Cross-dataset generalization, unified-judge re-evaluation, latency/VRAM benchmarking, statistical significance testing, and a systematic hallucination audit.

The repository contains only source code and configuration. Datasets and model weights are downloaded automatically from the Hugging Face Hub on first run (see [Dataset information](#dataset-information)); no protected health information is stored in this repository.

---

## Requirements

### Software

| Requirement | Version / notes |
| ----------- | --------------- |
| Python | 3.10 – 3.13 (developed and tested on 3.13) |
| OS | Windows, macOS, or Linux. Local fine-tuning (Unsloth) is recommended on Linux/Colab; Windows support is limited. |
| GPU | Optional but recommended. A CUDA GPU with ≥ 16 GB VRAM comfortably runs the 7B–8B models; smaller SLMs run on 8 GB. On low-VRAM machines, `airllm` loads large models layer-by-layer (much slower). |
| OpenAI API key | Required — used **only** for LLM-as-judge scoring, not for generation. |

### Python dependencies

Declared in [source/requirements.txt](source/requirements.txt):

```
torch                    # model inference
transformers             # model loading / generation
peft                     # LoRA adapters (Phase 3)
accelerate               # device placement
sentence-transformers    # dense retrieval (S-BERT)
rank-bm25                # sparse retrieval (BM25)
mlflow>=2.14.0           # experiment tracking, traces, judges
datasets                 # Hugging Face dataset loading
tqdm                     # progress bars
evaluate                 # BLEU / ROUGE / METEOR / BERTScore
bert-score               # BERTScore F1 (downloads ~270 MB on first run)
```

Additional packages used at runtime that you may need to install explicitly:

```
python-dotenv            # reads the .env file (imported as dotenv)
openai                   # LLM-as-judge API client
airllm                   # optional: layer-by-layer loader for models exceeding VRAM
```

### Installation

```bash
# 1. Clone and enter the repository
git clone <REPO_URL> soap-gen-bench
cd soap-gen-bench

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\activate            # Windows PowerShell

# 3. Install dependencies
pip install -r source/requirements.txt
pip install python-dotenv openai
pip install airllm                # optional — only if a model exceeds your VRAM
```

### Configuration

Create a `.env` file at the repository root:

```
OPENAI_API_KEY=sk-...                                    # Required — all LLM-as-judge calls
OPENAI_JUDGE_MODEL=gpt-4o-mini                           # Phase 1 correctness judge (default: gpt-4o-mini)
PHASE2_JUDGE_MODEL=gpt-4o-mini                           # Phase 2/3/EL RAGAS judges (default: same)
MLFLOW_TRACKING_URI=sqlite:///mlflow_experiments.db      # Recommended for MLflow 3 (file store is deprecated)
```

Optional overrides:

```
SHOWDOWN_SLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3    # Phase 2 SLM showdown generator
SHOWDOWN_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct   # Phase 2 LLM showdown generator
LORA_ADAPTER_PATH=./mistral_soap_lora                    # Phase 3 local adapter (overrides HF model)
```

Gated models (several Llama and Mistral checkpoints) require accepting the licence on Hugging Face and authenticating once with `huggingface-cli login`.

---

## Dataset information

No clinical data is redistributed in this repository. All datasets are public and are downloaded on demand from the Hugging Face Hub.

### Primary dataset

| Field | Value |
| ----- | ----- |
| Name | `adesouza1/soap_notes` |
| Source | [huggingface.co/datasets/adesouza1/soap_notes](https://huggingface.co/datasets/adesouza1/soap_notes) |
| Content | Doctor–patient dialogue transcripts paired with reference SOAP notes |
| Used by | Phases 1–3, EL3–EL6 |
| Access | Downloaded automatically on first run via the `datasets` library |

### External datasets (EL1 — cross-dataset generalization)

Configured in [source/config.py](source/config.py) (`CROSS_DATASETS`):

| Key | Hugging Face ID | Size | Split used |
| --- | --------------- | ---- | ---------- |
| `omi_health` | omi-health/medical-dialogue-to-soap-summary | 10K | test (250) |
| `augmented_clinical` | AGBonnet/augmented-clinical-notes | 30K | train (sample) |
| `subash_soap` | SubashNeupane/dataset_SOAP_summary | 1.5K | train |
| `rhyliieee_soap` | rhyliieee/soap-convo-v2 | 1K | train |

Each external dataset is subject to its own upstream licence and terms of use; consult the corresponding Hugging Face dataset card before redistribution. These corpora are used here strictly for research evaluation.

---

## Code information

The project is a single importable Python package, `source/`, with one module per responsibility. Run every command as a module from the repository root (e.g. `python -m source.experiments.phase1`).

```
source/
├── config.py                   # All constants — model names, dataset IDs, MLflow experiment names, env vars
├── requirements.txt            # Python dependencies
├── models/
│   ├── llm.py                  # ModelManager — load (fp16 → AirLLM fallback) / @mlflow.trace generate / unload
│   └── finetune.py             # LoRA fine-tuning via Unsloth
├── evaluation/
│   └── metrics.py              # Evaluators + STANDARD_SUITE / PHASE1_JUDGES / RAGAS_JUDGES
├── rag/
│   ├── retriever.py            # RetrievalEngine: sparse (BM25), dense (S-BERT), hybrid (RRF)
│   ├── query.py                # QueryEngine: rewrite, multi-query, section-wise queries
│   └── indexer.py              # SOAPIndexer: builds the 4 granularity indices
├── prompts/
│   ├── phase1.py               # H1–H5 string templates
│   └── rag.py                  # pack_evidence() + build_prompt() for RAG
├── data/
│   └── dataset.py              # load_examples(), load_examples_from_hf(), prepare_sft_jsonl()
├── tracking/
│   └── mlflow_logger.py        # evaluate_and_log() — metrics + judge feedback loop
└── experiments/
    ├── phase1.py               # Phase 1 runner (single-pass, OpenAI judge inline)
    ├── phase2.py               # Phase 2 runner — 14 experiments + RagPipeline
    ├── phase3.py               # Phase 3 runner — base / lora / lora_rag conditions
    ├── cross_dataset.py        # EL1: cross-dataset generalization (4 external datasets)
    ├── phase1_reeval.py        # EL3: Phase 1 re-scored with unified RAGAS judge
    ├── latency_benchmark.py    # EL4: load time + inference latency + VRAM per model
    ├── statistical_analysis.py # EL5: paired significance tests from MLflow run IDs
    └── hallucination_audit.py  # EL6: typed severity audit + Cohen's κ → CSV + MLflow
```

### Models

**Dataset:** [adesouza1/soap_notes](https://huggingface.co/datasets/adesouza1/soap_notes) — downloaded automatically on first run.

#### Phase 1 — SLMs

| Model | Size | Notes |
| ----- | ---- | ----- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1 B | |
| microsoft/phi-2 | 2.7 B | |
| Qwen/Qwen2.5-3B-Instruct | 3 B | |
| meta-llama/Llama-3.2-1B-Instruct | 1 B | Requires HF access approval |
| meta-llama/Llama-3.2-3B-Instruct | 3 B | Requires HF access approval |
| ibm-granite/granite-3.1-2b-instruct | 2 B | IBM enterprise/healthcare model; no gating |

#### Phase 1 — LLMs

| Model | Size | Notes |
| ----- | ---- | ----- |
| mistralai/Mistral-7B-Instruct-v0.3 | 7 B | |
| meta-llama/Meta-Llama-3-8B-Instruct | 8 B | Requires HF access approval |
| Qwen/Qwen2.5-7B-Instruct | 7 B | Pairs with Qwen2.5-3B SLM; no gating |
| ibm-granite/granite-3.1-8b-instruct | 8 B | Pairs with Granite-3.1-2B SLM; no gating |

For gated models, accept the licence at huggingface.co then run `huggingface-cli login`.

#### Phase 2 & 3

| Role | Model | Size |
| ---- | ----- | ---- |
| Phase 2 RAG generator | mistralai/Mistral-7B-Instruct-v0.3 | 7 B |
| Phase 2 showdown SLM | `SHOWDOWN_SLM_MODEL` (default: Mistral-7B) | 7 B |
| Phase 2 showdown LLM | `SHOWDOWN_LLM_MODEL` (default: Llama-3-8B) | 8 B |
| Phase 3 base | mistralai/Mistral-7B-v0.1 | 7 B |
| Phase 3 fine-tuned | SaberaBanu/mistral-soap-notes | 7 B |
| All phases judge | OpenAI `OPENAI_JUDGE_MODEL` / `PHASE2_JUDGE_MODEL` | — (API) |

All generator models run **locally**. Only the judge calls the OpenAI API.

---

## Usage instructions

Activate the virtual environment and start the MLflow UI, then run any experiment as a Python module from the repository root.

### Start the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_experiments.db
# Open http://127.0.0.1:5000
```

After running experiments, results appear in three tabs:

- **Runs** — aggregated metrics per condition
- **Traces** (`/#/experiments/{id}/traces`) — one trace per LLM generation call
- **Judges** (`/#/experiments/{id}/judges`) — LLM-as-judge feedback per trace

### Phase 1 — Prompt ablation

Evaluates H1–H5 prompt strategies across SLMs and LLMs. Each run logs BLEU, ROUGE-L, METEOR, completeness (deterministic) and correctness (OpenAI judge).

```bash
# All SLMs, all prompts
python -m source.experiments.phase1 --group slm --num_examples 50

# All LLMs, all prompts
python -m source.experiments.phase1 --group llm --num_examples 50

# All models, all prompts
python -m source.experiments.phase1 --group all --num_examples 50

# Single model
python -m source.experiments.phase1 --models Qwen/Qwen2.5-3B-Instruct --num_examples 50

# Multiple specific models
python -m source.experiments.phase1 --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct --num_examples 50

# Single prompt only
python -m source.experiments.phase1 --group slm --prompt H5_CoT --num_examples 50

# Resume from a specific prompt (skips earlier ones)
python -m source.experiments.phase1 --group llm --start_prompt H3_FewShot --num_examples 50
```

**Prompt strategies**

| ID | Strategy | Description |
| -- | -------- | ----------- |
| `H1_Baseline` | Zero-shot baseline | Bare instruction, no examples |
| `H2_Structured` | Structured | Explicit JSON schema in prompt |
| `H3_FewShot` | Few-shot | 1 in-context example |
| `H4_Dynamic` | Dynamic few-shot | Clinician persona + RAG placeholder |
| `H5_CoT` | Chain-of-thought | Step-by-step reasoning before JSON |

MLflow experiment: **SOAP_Phase1_Prompt_Ablation**. Run names: `{ModelShortName}__{PromptName}` (e.g. `Qwen2.5-3B-Instruct__H5_CoT`).

### Phase 2 — RAG ablation

Each run logs BLEU, ROUGE-L, METEOR, completeness (deterministic) and five RAGAS-aligned binary judges: `answer_relevancy`, `faithfulness`, `contextual_precision`, `contextual_recall`, `contextual_relevancy`.

```bash
# Single experiment
python -m source.experiments.phase2 --experiment e2_hybrid --num_examples 30

# All 14 configurations sequentially
python -m source.experiments.phase2 --all --num_examples 30
```

**Experiment catalogue**

| Key | Axis | What varies |
| --- | ---- | ----------- |
| `e1` | E1 RAG vs NoRAG | Dense retrieval, section-level, k=4 (sanity check) |
| `e2_sparse` | E2 Retriever | BM25 sparse |
| `e2_dense` | E2 Retriever | S-BERT dense |
| `e2_hybrid` | E2 Retriever | Hybrid RRF ← winner |
| `e3_note` | E3 Granularity | Note-level index ← winner |
| `e3_section` | E3 Granularity | Section-level index |
| `e3_fixed` | E3 Granularity | Fixed 256-token windows |
| `e3_struct` | E3 Granularity | Structure-aware chunks |
| `e4_raw` | E4 Query strategy | Raw transcript |
| `e4_rewrite` | E4 Query strategy | Rewritten symptom list |
| `e4_multi` | E4 Query strategy | Multi-query merged |
| `e4_section` | E4 Query strategy | Section-wise SOAP queries ← winner |
| `e_showdown_slm` | SLM–LLM showdown | Champion config + `SHOWDOWN_SLM_MODEL` |
| `e_showdown_llm` | SLM–LLM showdown | Champion config + `SHOWDOWN_LLM_MODEL` |

Champion config (used for E4 and showdown): **Hybrid RRF + note-level index + section-wise queries, k=8**. MLflow experiment: **SOAP_Phase2_RAG_Ablation**.

### Phase 3 — Fine-tuning evaluation

Compares three conditions using RAGAS judges. Contextual judges are skipped when no RAG context is present.

| Condition | Model | RAG |
| --------- | ----- | --- |
| `base` | mistralai/Mistral-7B-v0.1 (untuned) | No |
| `lora` | SaberaBanu/mistral-soap-notes (QLoRA fine-tuned, merged) | No |
| `lora_rag` | SaberaBanu/mistral-soap-notes + champion RAG config | Yes (EL2) |

If `LORA_ADAPTER_PATH` in `.env` points at an existing local adapter directory, the `lora` condition uses `Mistral-7B-v0.1 + local adapter` instead of the HF merged model.

```bash
python -m source.experiments.phase3 --num_examples 50              # base + lora
python -m source.experiments.phase3 --condition base --num_examples 50
python -m source.experiments.phase3 --condition lora --num_examples 50
python -m source.experiments.phase3 --condition lora_rag --num_examples 30   # EL2
```

MLflow experiment: **SOAP_Phase3_Finetuning**. Run names: `P3_base_Mistral-7B-v0.1` · `P3_lora_mistral-soap-notes` · `P3_lora_rag_mistral-soap-notes`.

### EL1 — Cross-dataset generalization

Tests whether Phase 1 results hold on external SOAP datasets. Addresses the single-corpus limitation.

```bash
# All 4 external datasets, champion prompts (H3_FewShot + H5_CoT), all models
python -m source.experiments.cross_dataset --num_examples 50

# Single dataset
python -m source.experiments.cross_dataset --dataset omi_health --num_examples 50

# Custom prompts
python -m source.experiments.cross_dataset --prompts H3_FewShot H5_CoT --group slm
```

MLflow experiment: **SOAP_CrossDataset_Generalization**.

### EL3 — Unified-judge re-evaluation

Phase 1 uses a continuous correctness judge (`OPENAI_JUDGE_MODEL`); Phase 2/3 use five binary RAGAS judges (`PHASE2_JUDGE_MODEL`). This makes the three-phase metric trajectory incomparable and introduces self-evaluation bias when the generator is also the judge. EL3 re-runs Phase 1 generation scored with the same RAGAS judges as Phase 2/3, enabling direct cross-phase comparison.

```bash
python -m source.experiments.phase1_reeval --num_examples 50
python -m source.experiments.phase1_reeval --group slm --num_examples 50
```

MLflow experiment: **SOAP_Phase1_UnifiedJudge**.

### EL4 — Latency & efficiency benchmark

Measures inference time and VRAM usage across all models for deployment evidence.

```bash
python -m source.experiments.latency_benchmark           # all models
python -m source.experiments.latency_benchmark --group slm
```

Logs `load_time_s`, `avg_latency_ms`, `p95_latency_ms`, `peak_vram_mb` per model. MLflow experiment: **SOAP_Latency_Benchmark**.

### EL5 — Statistical significance testing

Addresses the no-paired-tests limitation. Fetches per-case metrics from any two MLflow runs (logged at `step=0..n-1` by `evaluate_and_log`) and applies:

- **Binary metrics** (RAGAS judges 0/1): Wilson 95% CI per run, McNemar paired test (Yates-corrected)
- **Continuous metrics** (BLEU, ROUGE-L, METEOR, correctness): Bootstrap 95% CI per run, paired permutation test

```bash
# Compare two runs (copy run IDs from MLflow UI)
python -m source.experiments.statistical_analysis --run_a RUN_ID_A --run_b RUN_ID_B

# Single-run CIs only (no paired test)
python -m source.experiments.statistical_analysis --run_a RUN_ID_A

# Single metric only
python -m source.experiments.statistical_analysis --run_a ID_A --run_b ID_B --metric rouge
```

Run IDs are visible in the MLflow UI Runs table or in the console output of any experiment (`mlflow run_id: ...`). MLflow experiment: **SOAP_Statistical_Analysis**.

### EL6 — Systematic hallucination audit

Addresses the 15-case single-reviewer non-blinded audit limitation. For each example: generates a SOAP note, runs an LLM judge as Annotator 1 (accuracy framing) and optionally Annotator 2 (patient-safety framing), then computes Cohen's κ on the hallucination-type classification.

**Hallucination taxonomy:**

- Type: `none` | `commission` | `omission` | `both`
- Commission subtypes: `lab_fabrication`, `vital_fabrication`, `demographic`, `medication_error`, `unsupported_diagnosis`
- Omission subtypes: `medication_gap`, `allergy_gap`, `objective_gap`, `plan_omission`, `symptom_gap`
- Severity: `critical` | `moderate` | `minor`

```bash
# Single annotator pass (hallucination rates only)
python -m source.experiments.hallucination_audit --num_examples 50

# Two annotator passes → also computes Cohen's κ
python -m source.experiments.hallucination_audit --num_examples 50 --annotators 2

# Custom model or output path
python -m source.experiments.hallucination_audit --model Qwen/Qwen2.5-3B-Instruct --num_examples 30 --output_csv results/audit.csv
```

Outputs a per-case CSV with `ann1_*` (and `ann2_*`) columns. Summary metrics logged to MLflow: `hallucination_rate`, `commission_rate`, `omission_rate`, `critical_rate`, `cohens_kappa` (if `--annotators 2`). MLflow experiment: **SOAP_Hallucination_Audit**.

### Optional: build RAG indices and export the SFT dataset

```bash
# Build the 4 granularity indices (only if the pre-built index files are absent)
python -m source.rag.indexer

# Export Alpaca-format JSONL for Phase 3 fine-tuning
python -m source.data.dataset
```

### Optional: fine-tune locally

Requires Unsloth (Linux/Colab recommended — Windows support is limited):

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

python -m source.models.finetune                        # 60 steps, full dataset
python -m source.models.finetune --max_steps 120        # more steps
python -m source.models.finetune --num_examples 500     # limit training set
python -m source.models.finetune --save_dir ./my_lora   # custom save path
```

Adapters are saved to `LORA_ADAPTER_PATH` (default: `./mistral_soap_lora`). Once saved, Phase 3 evaluation uses them automatically instead of `SaberaBanu/mistral-soap-notes`.

---

## Methodology

The benchmark isolates one variable at a time so that each observed improvement can be attributed to a specific intervention. All phases share the same data loader, prompt-to-JSON extraction, evaluation suite, and MLflow logging path, so scores are directly comparable within a judging regime.

**1. Data preparation.** Transcripts and reference SOAP notes are loaded from the Hugging Face dataset via `source/data/dataset.py`. The number of evaluation examples is controlled with `--num_examples`.

**2. Generation.** `ModelManager` (`source/models/llm.py`) loads each open-weight model locally in fp16, falling back to AirLLM's layer-by-layer loader when a model exceeds available VRAM. Generation calls are wrapped with `@mlflow.trace`, so every call is captured in the MLflow Traces tab.

**3. Prompting (Phase 1).** Five strategies (H1–H5) escalate from a bare zero-shot instruction to an explicit JSON schema, a few-shot example, a clinician-persona dynamic prompt, and finally chain-of-thought reasoning before the JSON output.

**4. Retrieval (Phase 2).** The RAG ablation varies three axes independently — retriever type (BM25 sparse, S-BERT dense, hybrid RRF), index granularity (note-level, section-level, fixed windows, structure-aware chunks), and query strategy (raw, rewritten, multi-query, section-wise) — carrying the winner of each axis forward. The champion configuration is then used for the SLM-vs-LLM showdown.

**5. Fine-tuning (Phase 3).** A QLoRA-fine-tuned Mistral-7B is compared against its untuned base and against fine-tuning combined with the champion RAG configuration.

**6. Evaluation.** Every generation is scored with deterministic surface metrics (completeness, BLEU, ROUGE-L, METEOR, BERTScore F1) plus LLM-as-judge metrics — a continuous correctness judge in Phase 1 and five binary RAGAS-aligned judges in Phases 2–3. Because the two judging regimes are not directly comparable, EL3 re-scores Phase 1 under the RAGAS judges to place all phases on one axis. See [Metrics reference](#metrics-reference).

**7. Supplementary validation (EL1–EL6).** Cross-dataset generalization (EL1), the EL2 RAG+LoRA condition, unified-judge re-evaluation (EL3), latency/VRAM benchmarking (EL4), paired statistical significance testing with Wilson/McNemar and bootstrap/permutation tests (EL5), and a systematic, taxonomy-based hallucination audit with inter-annotator agreement (EL6) each address a specific threat to validity.

**8. Tracking.** `tracking/mlflow_logger.py` (`evaluate_and_log`) records per-case metrics at `step=0..n-1` and attaches judge feedback to each trace, which is what makes the EL5 paired tests reproducible from run IDs alone.

---

## Metrics reference

Surface metrics use the [Hugging Face `evaluate`](https://huggingface.co/docs/evaluate) library. BERTScore uses `distilbert-base-uncased` and downloads ~270 MB on first run.

| Suite | Metrics | Phases | Judge |
| ----- | ------- | ------ | ----- |
| `STANDARD_SUITE` | completeness, BLEU, ROUGE-L, METEOR, BERTScore F1 | All | — (deterministic) |
| `PHASE1_JUDGES` | correctness (continuous 0–1) | 1 | `OPENAI_JUDGE_MODEL` |
| `RAGAS_JUDGES` | answer_relevancy, faithfulness, contextual_precision, contextual_recall, contextual_relevancy (binary 0/1) | 2, 3, EL | `PHASE2_JUDGE_MODEL` |

Context-dependent RAGAS judges return `None` and are skipped when no retrieved context is available.

---

## MLflow experiment index

| Experiment name | Runner | What it measures |
| --------------- | ------ | ---------------- |
| `SOAP_Phase1_Prompt_Ablation` | `phase1.py` | Prompt strategy × model capacity |
| `SOAP_Phase2_RAG_Ablation` | `phase2.py` | RAG component ablation + showdown |
| `SOAP_Phase3_Finetuning` | `phase3.py` | Base vs. fine-tuned vs. RAG+LoRA |
| `SOAP_CrossDataset_Generalization` | `cross_dataset.py` | EL1: domain shift across 4 datasets |
| `SOAP_Phase1_UnifiedJudge` | `phase1_reeval.py` | EL3: Phase 1 re-scored with RAGAS judge |
| `SOAP_Latency_Benchmark` | `latency_benchmark.py` | EL4: load time, latency, VRAM |
| `SOAP_Statistical_Analysis` | `statistical_analysis.py` | EL5: Wilson CI, McNemar, permutation tests |
| `SOAP_Hallucination_Audit` | `hallucination_audit.py` | EL6: typed severity + Cohen's κ |

---

## Citations

If you use this code or the accompanying results, please cite the associated manuscript. Replace the placeholder fields below with the final publication details once available.

```bibtex
@article{soap_gen_bench,
  title   = {From Prompting to Retrieval: A Benchmark for Structured Clinical SOAP Note Generation},
  author  = {<AUTHORS>},
  journal = {<JOURNAL / VENUE>},
  year    = {<YEAR>},
  doi     = {<DOI>},
  note    = {Code: <REPOSITORY URL>}
}
```

This work builds on the following resources, which should also be cited where appropriate:

- **Primary dataset:** adesouza1/soap_notes — https://huggingface.co/datasets/adesouza1/soap_notes
- **Experiment tracking:** MLflow — https://mlflow.org
- **Evaluation metrics:** Hugging Face `evaluate` — https://huggingface.co/docs/evaluate
- **Retrieval-evaluation framework (RAGAS-aligned judges):** RAGAS — https://github.com/explodinggradients/ragas
- **Models:** cite the respective model cards (TinyLlama, Phi-2, Qwen2.5, Llama 3.2 / Llama 3, Mistral-7B, IBM Granite 3.1) as listed under [Code information → Models](#models).

---

## License and contribution guidelines

### License

This project is released under the **MIT License** — see [LICENSE](LICENSE). Copyright © 2026 Intelligent Agents & Experience Lab.

The MIT License covers the source code in this repository only. Datasets and pre-trained model weights are downloaded from third parties and remain subject to their own licences and terms of use; review each dataset and model card before redistribution or clinical use.

### Contribution guidelines

Contributions are welcome. To propose a change:

1. Open an issue describing the bug, experiment, or enhancement.
2. Fork the repository and create a feature branch.
3. Keep changes surgical and follow the existing module layout (one responsibility per module under `source/`).
4. Add or update the relevant experiment runner and, where applicable, log new metrics through `tracking/mlflow_logger.py` so results stay reproducible.
5. Open a pull request that references the issue and summarizes the experimental impact.

### Responsible-use note

This software generates draft clinical documentation and is intended for **research use only**. Generated SOAP notes are not a substitute for clinician judgement and must be reviewed by a qualified professional before any clinical use. Do not process real protected health information without appropriate ethical approval, data-use agreements, and privacy safeguards.
