# From Prompting to Retrieval: SOAP Note Generation Experiment

Systematic evaluation of SLMs and LLMs for structured clinical SOAP note generation across three phases:

- **Phase 1** — Prompt engineering ablation (H1–H5 strategies, up to 8 models)
- **Phase 2** — RAG ablation (retriever type × granularity × query strategy) + SLM–LLM showdown
- **Phase 3** — Fine-tuned model evaluation (base vs. SaberaBanu/mistral-soap-notes vs. RAG+LoRA)
- **EL1–EL6** — Supplementary experiments addressing dataset scope, judge consistency, deployment, statistical power, and hallucination auditing

All experiments are tracked locally with **MLflow**, including GenAI **Traces** and **Judges** tabs.

---

## Setup

### 1. Activate the virtual environment

```powershell
.venv\Scripts\activate          # Windows PowerShell
source .venv/bin/activate        # Mac / Linux
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install airllm          # layer-by-layer loader for models that exceed VRAM
```

### 3. Configure environment

Create a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...                                    # Required — used for all LLM-as-judge calls
OPENAI_JUDGE_MODEL=gpt-4o-mini                           # Phase 1 correctness judge (default: gpt-4o-mini)
PHASE2_JUDGE_MODEL=gpt-4o-mini                           # Phase 2/3/EL RAGAS judges (default: same)
MLFLOW_TRACKING_URI=sqlite:///mlflow_experiments.db      # Required for MLflow 3 (file store is deprecated)
```

Optional overrides:

```
SHOWDOWN_SLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3   # Phase 2 SLM showdown generator
SHOWDOWN_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct  # Phase 2 LLM showdown generator
LORA_ADAPTER_PATH=./mistral_soap_lora                   # Phase 3 local adapter (overrides HF model)
```

### 4. Start the MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow_experiments.db
# Open http://127.0.0.1:5000
```

After running experiments, results appear in three tabs:

- **Runs** — aggregated metrics per condition
- **Traces** (`/#/experiments/{id}/traces`) — one trace per LLM generation call
- **Judges** (`/#/experiments/{id}/judges`) — LLM-as-judge feedback per trace

---

## Running experiments

All runners are Python modules — run from the repo root with the venv active.

### Phase 1 — Prompt ablation

Evaluates H1–H5 prompt strategies across SLMs and LLMs (see model table below).
Each run logs BLEU, ROUGE-L, METEOR, completeness (deterministic) and correctness (OpenAI judge).

```powershell
# All SLMs, all prompts
python -m source.experiments.phase1 --group slm --num_examples 50

# All LLMs, all prompts
python -m source.experiments.phase1 --group llm --num_examples 50

# All models, all prompts
python -m source.experiments.phase1 --group all --num_examples 50

# Single model
python -m source.experiments.phase1 --models Qwen/Qwen2.5-3B-Instruct --num_examples 50

python -m source.experiments.phase1 --models mistralai/Mistral-7B-Instruct-v0.3 --num_examples 50

# Multiple specific models
python -m source.experiments.phase1 --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct --num_examples 50

# Single prompt only
python -m source.experiments.phase1 --group slm --prompt H5_CoT --num_examples 50

# Resume from a specific prompt (skips earlier ones)
python -m source.experiments.phase1 --group llm --start_prompt H3_FewShot --num_examples 50
```

**Prompt strategies**

| ID                | Strategy           | Description                         |
| ----------------- | ------------------ | ----------------------------------- |
| `H1_Baseline`   | Zero-shot baseline | Bare instruction, no examples       |
| `H2_Structured` | Structured         | Explicit JSON schema in prompt      |
| `H3_FewShot`    | Few-shot           | 1 in-context example                |
| `H4_Dynamic`    | Dynamic few-shot   | Clinician persona + RAG placeholder |
| `H5_CoT`        | Chain-of-thought   | Step-by-step reasoning before JSON  |

MLflow experiment: **SOAP_Phase1_Prompt_Ablation**
Run names: `{ModelShortName}__{PromptName}` (e.g. `Qwen2.5-3B-Instruct__H5_CoT`)

---

### Phase 2 — RAG ablation

Each run logs BLEU, ROUGE-L, METEOR, completeness (deterministic) and five RAGAS-aligned binary judges:
`answer_relevancy`, `faithfulness`, `contextual_precision`, `contextual_recall`, `contextual_relevancy`.

```powershell
# Single experiment
python -m source.experiments.phase2 --experiment e2_hybrid --num_examples 30

# All 14 configurations sequentially
python -m source.experiments.phase2 --all --num_examples 30
```

**Experiment catalogue**

| Key                | Axis              | What varies                                        |
| ------------------ | ----------------- | -------------------------------------------------- |
| `e1`             | E1 RAG vs NoRAG   | Dense retrieval, section-level, k=4 (sanity check) |
| `e2_sparse`      | E2 Retriever      | BM25 sparse                                        |
| `e2_dense`       | E2 Retriever      | S-BERT dense                                       |
| `e2_hybrid`      | E2 Retriever      | Hybrid RRF ← winner                               |
| `e3_note`        | E3 Granularity    | Note-level index ← winner                         |
| `e3_section`     | E3 Granularity    | Section-level index                                |
| `e3_fixed`       | E3 Granularity    | Fixed 256-token windows                            |
| `e3_struct`      | E3 Granularity    | Structure-aware chunks                             |
| `e4_raw`         | E4 Query strategy | Raw transcript                                     |
| `e4_rewrite`     | E4 Query strategy | Rewritten symptom list                             |
| `e4_multi`       | E4 Query strategy | Multi-query merged                                 |
| `e4_section`     | E4 Query strategy | Section-wise SOAP queries ← winner                |
| `e_showdown_slm` | SLM–LLM showdown | Champion config +`SHOWDOWN_SLM_MODEL`            |
| `e_showdown_llm` | SLM–LLM showdown | Champion config +`SHOWDOWN_LLM_MODEL`            |

Champion config (used for E4 and showdown): **Hybrid RRF + note-level index + section-wise queries, k=8**.

MLflow experiment: **SOAP_Phase2_RAG_Ablation**

---

### Phase 3 — Fine-tuning evaluation

Compares three conditions using RAGAS judges. Contextual judges are skipped when no RAG context
is present.

| Condition    | Model                                                    | RAG       |
| ------------ | -------------------------------------------------------- | --------- |
| `base`     | mistralai/Mistral-7B-v0.1 (untuned)                      | No        |
| `lora`     | SaberaBanu/mistral-soap-notes (QLoRA fine-tuned, merged) | No        |
| `lora_rag` | SaberaBanu/mistral-soap-notes + champion RAG config      | Yes (EL2) |

If `LORA_ADAPTER_PATH` in `.env` points at an existing local adapter directory, the `lora`
condition uses `Mistral-7B-v0.1 + local adapter` instead of the HF merged model.

```powershell
python -m source.experiments.phase3 --num_examples 50              # base + lora
python -m source.experiments.phase3 --condition base --num_examples 50
python -m source.experiments.phase3 --condition lora --num_examples 50
python -m source.experiments.phase3 --condition lora_rag --num_examples 30   # EL2
```

MLflow experiment: **SOAP_Phase3_Finetuning**
Run names: `P3_base_Mistral-7B-v0.1` · `P3_lora_mistral-soap-notes` · `P3_lora_rag_mistral-soap-notes`

---

### EL1 — Cross-dataset generalization

Tests whether Phase 1 results hold on external SOAP datasets. Addresses the single-corpus limitation.

```powershell
# All 4 external datasets, champion prompts (H3_FewShot + H5_CoT), all models
python -m source.experiments.cross_dataset --num_examples 50

# Single dataset
python -m source.experiments.cross_dataset --dataset omi_health --num_examples 50

# Custom prompts
python -m source.experiments.cross_dataset --prompts H3_FewShot H5_CoT --group slm
```

**Datasets:**

| Key                    | HuggingFace ID                              | Size | Split used     |
| ---------------------- | ------------------------------------------- | ---- | -------------- |
| `omi_health`         | omi-health/medical-dialogue-to-soap-summary | 10K  | test (250)     |
| `augmented_clinical` | AGBonnet/augmented-clinical-notes           | 30K  | train (sample) |
| `subash_soap`        | SubashNeupane/dataset_SOAP_summary          | 1.5K | train          |
| `rhyliieee_soap`     | rhyliieee/soap-convo-v2                     | 1K   | train          |

MLflow experiment: **SOAP_CrossDataset_Generalization**

---

### EL3 — Unified-judge re-evaluation

Phase 1 uses a continuous correctness judge (`OPENAI_JUDGE_MODEL`); Phase 2/3 use
five binary RAGAS judges (`PHASE2_JUDGE_MODEL`). This makes the three-phase metric
trajectory incomparable and introduces self-evaluation bias when the generator is
also the judge. EL3 re-runs Phase 1 generation scored with the same RAGAS judges as
Phase 2/3, enabling direct cross-phase comparison.

```powershell
python -m source.experiments.phase1_reeval --num_examples 50
python -m source.experiments.phase1_reeval --group slm --num_examples 50
```

MLflow experiment: **SOAP_Phase1_UnifiedJudge**

---

### EL4 — Latency & efficiency benchmark

Measures inference time and VRAM usage across all models for deployment evidence.

```powershell
python -m source.experiments.latency_benchmark           # all models
python -m source.experiments.latency_benchmark --group slm
```

Logs `load_time_s`, `avg_latency_ms`, `p95_latency_ms`, `peak_vram_mb` per model.
MLflow experiment: **SOAP_Latency_Benchmark**

---

### EL5 — Statistical significance testing

Addresses the no-paired-tests limitation. Fetches per-case metrics from any two MLflow runs
(logged at `step=0..n-1` by `evaluate_and_log`) and applies:

- **Binary metrics** (RAGAS judges 0/1): Wilson 95% CI per run, McNemar paired test (Yates-corrected)
- **Continuous metrics** (BLEU, ROUGE-L, METEOR, correctness): Bootstrap 95% CI per run, paired permutation test

```powershell
# Compare two runs (copy run IDs from MLflow UI)
python -m source.experiments.statistical_analysis --run_a RUN_ID_A --run_b RUN_ID_B

# Single-run CIs only (no paired test)
python -m source.experiments.statistical_analysis --run_a RUN_ID_A

# Single metric only
python -m source.experiments.statistical_analysis --run_a ID_A --run_b ID_B --metric rouge
```

Run IDs are visible in the MLflow UI Runs table or in the console output of any experiment
(`mlflow run_id: ...`). MLflow experiment: **SOAP_Statistical_Analysis**

---

### EL6 — Systematic hallucination audit

Addresses the 15-case single-reviewer non-blinded audit limitation. For each example: generates
a SOAP note, runs an LLM judge as Annotator 1 (accuracy framing) and optionally Annotator 2
(patient-safety framing), then computes Cohen's κ on the hallucination-type classification.

**Hallucination taxonomy:**

- Type: `none` | `commission` | `omission` | `both`
- Commission subtypes: `lab_fabrication`, `vital_fabrication`, `demographic`, `medication_error`, `unsupported_diagnosis`
- Omission subtypes: `medication_gap`, `allergy_gap`, `objective_gap`, `plan_omission`, `symptom_gap`
- Severity: `critical` | `moderate` | `minor`

```powershell
# Single annotator pass (hallucination rates only)
python -m source.experiments.hallucination_audit --num_examples 50

# Two annotator passes → also computes Cohen's κ
python -m source.experiments.hallucination_audit --num_examples 50 --annotators 2

# Custom model or output path
python -m source.experiments.hallucination_audit --model Qwen/Qwen2.5-3B-Instruct --num_examples 30 --output_csv results/audit.csv
```

Outputs a per-case CSV with `ann1_*` (and `ann2_*`) columns. Summary metrics logged to
MLflow: `hallucination_rate`, `commission_rate`, `omission_rate`, `critical_rate`,
`cohens_kappa` (if `--annotators 2`). MLflow experiment: **SOAP_Hallucination_Audit**

---

### Fine-tune locally (optional)

Mirrors `src/level3/stage_4_finetuning.ipynb`. Requires unsloth (Linux/Colab recommended — Windows support is limited):

```powershell
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

python -m source.models.finetune                        # 60 steps, full dataset
python -m source.models.finetune --max_steps 120        # more steps
python -m source.models.finetune --num_examples 500     # limit training set
python -m source.models.finetune --save_dir ./my_lora   # custom save path
```

Adapters are saved to `LORA_ADAPTER_PATH` (default: `./mistral_soap_lora`). Once saved, Phase 3 evaluation will use them automatically instead of `SaberaBanu/mistral-soap-notes`.

---

### Build RAG indices (one-time)

Only needed if the pre-built files in `src/level2/data/` are absent:

```powershell
python -m source.rag.indexer
```

### Export SFT training dataset

Writes Alpaca-format JSONL for Phase 3 fine-tuning:

```powershell
python -m source.data.dataset
```

---

## Metrics reference

Surface metrics use the [HuggingFace `evaluate`](https://huggingface.co/docs/evaluate) library.
BERTScore uses `distilbert-base-uncased` and downloads ~270 MB on first run.

| Suite              | Metrics                                                                                                    | Phases   | Judge                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | -------- | ---------------------- |
| `STANDARD_SUITE` | completeness, BLEU, ROUGE-L, METEOR, BERTScore F1                                                          | All      | — (deterministic)     |
| `PHASE1_JUDGES`  | correctness (continuous 0–1)                                                                              | 1        | `OPENAI_JUDGE_MODEL` |
| `RAGAS_JUDGES`   | answer_relevancy, faithfulness, contextual_precision, contextual_recall, contextual_relevancy (binary 0/1) | 2, 3, EL | `PHASE2_JUDGE_MODEL` |

Context-dependent RAGAS judges return `None` and are skipped when no retrieved context is available.

---

## MLflow experiment index

| Experiment name                      | Runner                      | What it measures                           |
| ------------------------------------ | --------------------------- | ------------------------------------------ |
| `SOAP_Phase1_Prompt_Ablation`      | `phase1.py`               | Prompt strategy × model capacity          |
| `SOAP_Phase2_RAG_Ablation`         | `phase2.py`               | RAG component ablation + showdown          |
| `SOAP_Phase3_Finetuning`           | `phase3.py`               | Base vs. fine-tuned vs. RAG+LoRA           |
| `SOAP_CrossDataset_Generalization` | `cross_dataset.py`        | EL1: domain shift across 4 datasets        |
| `SOAP_Phase1_UnifiedJudge`         | `phase1_reeval.py`        | EL3: Phase 1 re-scored with RAGAS judge    |
| `SOAP_Latency_Benchmark`           | `latency_benchmark.py`    | EL4: load time, latency, VRAM              |
| `SOAP_Statistical_Analysis`        | `statistical_analysis.py` | EL5: Wilson CI, McNemar, permutation tests |
| `SOAP_Hallucination_Audit`         | `hallucination_audit.py`  | EL6: typed severity + Cohen's κ           |

---

## Source package layout

```
source/
├── config.py                   # All constants — model names, dataset IDs, MLflow experiment names
├── models/
│   ├── llm.py                  # ModelManager — load (fp16 → AirLLM fallback) / @mlflow.trace generate / unload
│   └── finetune.py             # LoRA fine-tuning via Unsloth (mirrors stage_4_finetuning.ipynb)
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

---

## Dataset and models

**Dataset:** [adesouza1/soap_notes](https://huggingface.co/datasets/adesouza1/soap_notes) — downloaded automatically on first run.

### Phase 1 — SLMs

| Model                               | Size  | Notes                                      |
| ----------------------------------- | ----- | ------------------------------------------ |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0  | 1.1 B |                                            |
| microsoft/phi-2                     | 2.7 B |                                            |
| Qwen/Qwen2.5-3B-Instruct            | 3 B   |                                            |
| meta-llama/Llama-3.2-1B-Instruct    | 1 B   | Requires HF access approval                |
| meta-llama/Llama-3.2-3B-Instruct    | 3 B   | Requires HF access approval                |
| ibm-granite/granite-3.1-2b-instruct | 2 B   | IBM enterprise/healthcare model; no gating |

### Phase 1 — LLMs

| Model                               | Size | Notes                                    |
| ----------------------------------- | ---- | ---------------------------------------- |
| mistralai/Mistral-7B-Instruct-v0.3  | 7 B  |                                          |
| meta-llama/Meta-Llama-3-8B-Instruct | 8 B  | Requires HF access approval              |
| Qwen/Qwen2.5-7B-Instruct            | 7 B  | Pairs with Qwen2.5-3B SLM; no gating     |
| ibm-granite/granite-3.1-8b-instruct | 8 B  | Pairs with Granite-3.1-2B SLM; no gating |

For gated models, accept the licence at huggingface.co then run `huggingface-cli login`.

### Phase 2 & 3

| Role                  | Model                                                 | Size     |
| --------------------- | ----------------------------------------------------- | -------- |
| Phase 2 RAG generator | mistralai/Mistral-7B-Instruct-v0.3                    | 7 B      |
| Phase 2 showdown SLM  | `SHOWDOWN_SLM_MODEL` (default: Mistral-7B)          | 7 B      |
| Phase 2 showdown LLM  | `SHOWDOWN_LLM_MODEL` (default: Llama-3-8B)          | 8 B      |
| Phase 3 base          | mistralai/Mistral-7B-v0.1                             | 7 B      |
| Phase 3 fine-tuned    | SaberaBanu/mistral-soap-notes                         | 7 B      |
| All phases judge      | OpenAI`OPENAI_JUDGE_MODEL` / `PHASE2_JUDGE_MODEL` | — (API) |

All generator models run **locally**. Only the judge calls the OpenAI API.

