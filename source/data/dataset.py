"""Dataset utilities: load from HuggingFace, export SFT JSONL."""
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from ..config import DATASET_HF


def _row_fields(row: dict) -> tuple[str, str]:
    """Normalise column-name variations across dataset versions."""
    conv = (
        row.get("patient_convo")
        or row.get("dialogue")
        or row.get("conversation")
        or row.get("text")
        or ""
    )
    soap = row.get("soap_notes") or row.get("soap_note") or ""
    return conv, soap


def load_examples(n: int | None = None) -> list[dict]:
    """Return up to n examples as plain dicts ready for the tracking loop.

    Each entry: {"inputs": {"conversation": str}, "outputs": {"reference_soap": str}}
    """
    ds = load_dataset(DATASET_HF, split="train")
    if n:
        ds = ds.select(range(min(len(ds), n)))

    examples = []
    for row in ds:
        conv, soap = _row_fields(row)
        if conv:
            examples.append({"inputs": {"conversation": conv}, "outputs": {"reference_soap": soap}})
    return examples


def load_examples_from_hf(
    dataset_id: str,
    n: int | None = None,
    split: str = "train",
) -> list[dict]:
    """Load examples from any HuggingFace SOAP-style dataset.

    Normalises column names across different dataset schemas so the result
    matches the format expected by evaluate_and_log():
        [{"inputs": {"conversation": str}, "outputs": {"reference_soap": str}}, ...]

    Conversation field tried in order:
        dialogue, conversation, input, context, patient_convo, text
    Reference SOAP field tried in order:
        soap, note, output, response, summary, soap_notes, soap_note
    """
    ds = load_dataset(dataset_id, split=split)
    if n:
        ds = ds.select(range(min(len(ds), n)))

    _CONV_KEYS = ("dialogue", "conversation", "input", "context", "patient_convo", "text")
    _SOAP_KEYS = ("soap", "note", "output", "response", "summary", "soap_notes", "soap_note")

    examples = []
    for row in ds:
        conv = next((row[k] for k in _CONV_KEYS if k in row and row[k]), None)
        soap = next((row[k] for k in _SOAP_KEYS if k in row and row[k]), "")
        if conv:
            examples.append({"inputs": {"conversation": conv}, "outputs": {"reference_soap": soap}})
    return examples


def prepare_sft_jsonl(output_path: str | Path) -> None:
    """Format the dataset as Alpaca-style JSONL for supervised fine-tuning."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instruction = (
        "Generate a SOAP note from the clinical conversation. "
        "Output MUST be a valid JSON object with detailed content for the following keys: "
        '"subjective", "objective", "assessment", "plan".'
    )

    ds = load_dataset(DATASET_HF, split="train")
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Formatting SFT dataset"):
            conv, soap = _row_fields(row)
            if not conv or not soap:
                continue
            f.write(json.dumps({"instruction": instruction, "input": conv, "output": soap}) + "\n")
            written += 1

    print(f"Saved {written} examples to {output_path}")


if __name__ == "__main__":
    prepare_sft_jsonl(Path(__file__).parent.parent.parent / "src" / "level2" / "data" / "sft_train.jsonl")
