"""Query rewriting strategies for Phase 2 Experiment 4."""
import re
from typing import List


class QueryEngine:
    """Uses the loaded ModelManager to generate alternative retrieval queries."""

    # ------------------------------------------------------------------
    # Q1: Rewrite to clinical summary
    # ------------------------------------------------------------------

    def rewrite(self, conversation: str) -> str:
        from ..models.llm import manager
        if not manager.is_ready:
            return conversation[:500]
        messages = [{"role": "user", "content": (
            "Rewrite the following clinical conversation into a concise list of "
            "medical symptoms, findings, and history suitable for document retrieval. "
            "Respond ONLY with the list.\n\n"
            f"CONVERSATION:\n{conversation[-1000:]}"
        )}]
        result = manager.generate(messages, max_new_tokens=60)
        return result or conversation[:500]

    # ------------------------------------------------------------------
    # Q2: Generate multiple queries
    # ------------------------------------------------------------------

    def multi_query(self, conversation: str, n: int = 3) -> List[str]:
        from ..models.llm import manager
        if not manager.is_ready:
            return [conversation[:300]]
        messages = [{"role": "user", "content": (
            f"Generate {n} different search queries to find clinical evidence for the "
            "following conversation. Focus on: 1. History, 2. Physical Findings, 3. Plan. "
            "Respond as a numbered list.\n\n"
            f"CONVERSATION:\n{conversation[-1000:]}"
        )}]
        result = manager.generate(messages, max_new_tokens=100)
        queries = re.findall(r"\d+\.\s*(.*)", result) or result.split("\n")[:n]
        return [q for q in queries if q.strip()][:n] or [conversation[:300]]

    # ------------------------------------------------------------------
    # Q3: One query per SOAP section
    # ------------------------------------------------------------------

    def section_queries(self, conversation: str) -> List[str]:
        from ..models.llm import manager
        if not manager.is_ready:
            return [
                "Subjective symptoms",
                "Physical examination findings",
                "Diagnosis and assessment",
                "Treatment plan",
            ]
        messages = [{"role": "user", "content": (
            "Based on this transcript, generate 4 targeted search queries — one for "
            "each SOAP section: 1. Subjective (symptoms), 2. Objective (exam/tests), "
            "3. Assessment (diagnosis), 4. Plan (treatment). "
            "Respond ONLY with the 4 numbered queries.\n\n"
            f"CONVERSATION:\n{conversation[-1000:]}"
        )}]
        result = manager.generate(messages, max_new_tokens=120)
        queries = re.findall(r"\d+\.\s*(.*)", result) or result.split("\n")[:4]
        return [q for q in queries if q.strip()][:4]
