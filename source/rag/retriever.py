"""Sparse, dense, and hybrid (RRF) retrieval over pre-built SOAP note indices."""
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import INDEX_DIR

# Valid index names — match the filenames in INDEX_DIR
INDEX_NAMES = ("index_a_note", "index_b_section", "index_fixed", "index_struct_aware")


class RetrievalEngine:
    """Loads dense vectors and BM25 index for one granularity level."""

    def __init__(self, index_name: str = "index_b_section", embedding_model: str = "all-MiniLM-L6-v2"):
        if index_name not in INDEX_NAMES:
            raise ValueError(f"index_name must be one of {INDEX_NAMES}, got {index_name!r}")
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        self._embed_model = None

        # Dense
        self._dense_vectors: np.ndarray | None = None
        self._dense_meta: list | None = None

        # Sparse
        self._bm25 = None
        self._sparse_meta: list | None = None

        self._load()

    def _load(self) -> None:
        v_path = INDEX_DIR / f"{self.index_name}_vectors.npy"
        m_path = INDEX_DIR / f"{self.index_name}_metadata.pkl"
        if v_path.exists() and m_path.exists():
            self._dense_vectors = np.load(v_path)
            with open(m_path, "rb") as f:
                self._dense_meta = pickle.load(f)

        s_path = INDEX_DIR / f"{self.index_name}_bm25.pkl"
        if s_path.exists():
            with open(s_path, "rb") as f:
                data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._sparse_meta = data["metadata"]

    @property
    def _embed(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def search_dense(self, query: str, k: int = 8) -> List[Dict]:
        if self._dense_vectors is None:
            return []
        vec = self._embed.encode([query])[0]
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        scores = np.dot(self._dense_vectors, vec)
        top = np.argsort(scores)[::-1][:k]
        return [{"metadata": self._dense_meta[i], "score": float(scores[i])} for i in top]

    def search_sparse(self, query: str, k: int = 8) -> List[Dict]:
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k]
        return [{"metadata": self._sparse_meta[i], "score": float(scores[i])} for i in top]

    def search_hybrid(self, query: str, k: int = 8, rrf_k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion of dense + sparse results."""
        dense = self.search_dense(query, k=rrf_k)
        sparse = self.search_sparse(query, k=rrf_k)

        def _uid(meta: dict) -> tuple:
            return (meta["source_id"], meta["content"][:50])

        rrf: dict = {}
        for rank, r in enumerate(dense):
            uid = _uid(r["metadata"])
            rrf[uid] = rrf.get(uid, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, r in enumerate(sparse):
            uid = _uid(r["metadata"])
            rrf[uid] = rrf.get(uid, 0) + 1.0 / (rrf_k + rank + 1)

        uid_to_meta = {_uid(r["metadata"]): r["metadata"] for r in dense + sparse}
        top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"metadata": uid_to_meta[uid], "score": score} for uid, score in top]
