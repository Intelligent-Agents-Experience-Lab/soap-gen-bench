"""Build and persist all four granularity indices from the HuggingFace dataset.

Run once:
    python -m source.rag.indexer
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import DATASET_HF, INDEX_DIR

CHUNK_TOKENS = 256  # target size for fixed-size chunks


class SOAPIndexer:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._embed_model = None

    @property
    def _embed(self) -> SentenceTransformer:
        if self._embed_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}…")
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_sections(self, text: str) -> dict:
        """Split plain-text SOAP note into {Subjective, Objective, Assessment, Plan}."""
        sections = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}
        parts = re.split(r"(Subjective|Objective|Assessment|Plan):", text, flags=re.IGNORECASE)
        current = None
        for part in parts:
            if not part:
                continue
            match = re.fullmatch(r"(Subjective|Objective|Assessment|Plan)", part.strip(), re.IGNORECASE)
            if match:
                current = match.group(1).capitalize()
            elif current:
                sections[current] += part.strip()
        return sections

    def _fixed_chunks(self, text: str, max_tokens: int = CHUNK_TOKENS) -> list:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]

    # ------------------------------------------------------------------
    # Build all four corpora and save
    # ------------------------------------------------------------------

    def build(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Loading dataset {DATASET_HF}…")
        ds = load_dataset(DATASET_HF, split="train")

        a_corpus, b_corpus, c_corpus, d_corpus = [], [], [], []

        for i, row in enumerate(ds):
            soap = row.get("soap_notes") or row.get("soap_note") or ""
            conv = (
                row.get("patient_convo")
                or row.get("dialogue")
                or row.get("conversation")
                or row.get("text")
                or ""
            )
            if not soap:
                continue

            # Index A — full note
            a_corpus.append({"source_id": i, "content": soap, "type": "note", "section_type": "Full"})

            # Index B — section-level
            for section, content in self._parse_sections(soap).items():
                if content.strip():
                    b_corpus.append({"source_id": i, "content": content, "type": "section", "section_type": section})

            # Index C — fixed 256-token chunks
            for chunk in self._fixed_chunks(soap):
                c_corpus.append({"source_id": i, "content": chunk, "type": "chunk", "section_type": "Fixed"})

            # Index D — structure-aware: sections chunked if long
            for section, content in self._parse_sections(soap).items():
                if not content.strip():
                    continue
                words = content.split()
                if len(words) <= CHUNK_TOKENS:
                    d_corpus.append({"source_id": i, "content": content, "type": "struct_chunk", "section_type": section})
                else:
                    for chunk in self._fixed_chunks(content):
                        d_corpus.append({"source_id": i, "content": chunk, "type": "struct_chunk", "section_type": section})

        for name, corpus in [
            ("index_a_note", a_corpus),
            ("index_b_section", b_corpus),
            ("index_fixed", c_corpus),
            ("index_struct_aware", d_corpus),
        ]:
            self._save(name, corpus)

    def _save(self, name: str, corpus: list) -> None:
        print(f"Encoding {len(corpus)} docs for {name}…")
        texts = [d["content"] for d in corpus]

        # Dense vectors
        vecs = self._embed.encode(texts, show_progress_bar=True, batch_size=64)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        np.save(INDEX_DIR / f"{name}_vectors.npy", vecs)
        with open(INDEX_DIR / f"{name}_metadata.pkl", "wb") as f:
            pickle.dump(corpus, f)

        # BM25
        tokenised = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenised)
        with open(INDEX_DIR / f"{name}_bm25.pkl", "wb") as f:
            pickle.dump({"bm25": bm25, "metadata": corpus}, f)

        print(f"  Saved {name} ({len(corpus)} docs)")


if __name__ == "__main__":
    SOAPIndexer().build()
