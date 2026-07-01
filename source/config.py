import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

# Pre-built BM25/dense index files (built once by rag/indexer.py)
INDEX_DIR = _REPO_ROOT / "src" / "level2" / "data"

# HuggingFace dataset
DATASET_HF = "adesouza1/soap_notes"

# MLflow — defaults to ./mlruns (local).
# Start the UI with: mlflow ui  →  http://127.0.0.1:5000
# Override with the MLFLOW_TRACKING_URI env var for a remote server.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_P1     = "SOAP_Phase1_Prompt_Ablation"
MLFLOW_EXPERIMENT_P2     = "SOAP_Phase2_RAG_Ablation"
MLFLOW_EXPERIMENT_P3     = "SOAP_Phase3_Finetuning"
MLFLOW_EXPERIMENT_CROSS        = "SOAP_CrossDataset_Generalization"
MLFLOW_EXPERIMENT_REEVAL       = "SOAP_Phase1_UnifiedJudge"
MLFLOW_EXPERIMENT_LAT          = "SOAP_Latency_Benchmark"
MLFLOW_EXPERIMENT_STATS        = "SOAP_Statistical_Analysis"
MLFLOW_EXPERIMENT_HALLUCINATION = "SOAP_Hallucination_Audit"

# External datasets for cross-dataset generalization (EL1)
CROSS_DATASETS: dict[str, str] = {
    "omi_health":        "omi-health/medical-dialogue-to-soap-summary",
    "augmented_clinical": "AGBonnet/augmented-clinical-notes",
    "subash_soap":       "SubashNeupane/dataset_SOAP_summary",
    "rhyliieee_soap":    "rhyliieee/soap-convo-v2",
}

# OpenAI judge (used for all LLM-as-judge calls)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")   # Phase 1 correctness
PHASE2_JUDGE_MODEL = os.environ.get("PHASE2_JUDGE_MODEL", OPENAI_JUDGE_MODEL)  # Phase 2/3 RAGAS

# Phase 2 SLM–LLM showdown — both run locally via ModelManager
SHOWDOWN_SLM_MODEL = os.environ.get("SHOWDOWN_SLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
SHOWDOWN_LLM_MODEL = os.environ.get("SHOWDOWN_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Model catalogue
MODEL_GROUPS = {
    "slm": [
        # Original SLMs
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "Qwen/Qwen2.5-3B-Instruct",
        # Added: modern SLMs for broader comparison
        "meta-llama/Llama-3.2-1B-Instruct",        # Llama 3.2 @ 1B — modern arch vs TinyLlama
        "meta-llama/Llama-3.2-3B-Instruct",        # Llama 3.2 @ 3B — vs Qwen2.5-3B
        "ibm-granite/granite-3.1-2b-instruct",     # IBM enterprise/healthcare SLM @ 2B
    ],
    "llm": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",           # same family as Qwen2.5-3B SLM
        "ibm-granite/granite-3.1-8b-instruct", # same family as Granite-3.1-2B SLM
    ],
}

RAG_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FT_BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# Merged fine-tuned model used as Phase 3 "lora" condition by default.
# If LORA_ADAPTER_PATH points at a local adapter directory, that takes precedence.
FT_FINETUNED_MODEL = "SaberaBanu/mistral-soap-notes"

# LoRA adapters — set LORA_ADAPTER_PATH env var or place adapters at repo root
LORA_ADAPTER_PATH = os.environ.get(
    "LORA_ADAPTER_PATH",
    str(_REPO_ROOT / "mistral_soap_lora"),
)
