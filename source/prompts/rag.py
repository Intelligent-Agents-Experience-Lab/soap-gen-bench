"""RAG prompt builder shared by all Phase 2 experiments."""
from typing import List, Dict


def pack_evidence(chunks: List[Dict]) -> str:
    """Group retrieved chunks by SOAP section into structured evidence slots."""
    slots: dict[str, list] = {"Subjective": [], "Objective": [], "Assessment": [], "Plan": []}
    for chunk in chunks:
        meta = chunk.get("metadata", chunk)  # accept raw metadata or wrapped dict
        section = meta.get("section_type", "Subjective")
        content = meta.get("content", "")
        if section in slots:
            slots[section].append(content)
        else:
            slots.setdefault(section, []).append(content)

    lines = []
    for section, contents in slots.items():
        lines.append(f"### Evidence-{section} ###")
        if contents:
            lines.extend(f"- {c}" for c in contents)
        else:
            lines.append("- No specific evidence found.")
        lines.append("")
    return "\n".join(lines)


def build_prompt(conversation: str, chunks: List[Dict]) -> str:
    evidence = pack_evidence(chunks)
    return (
        "You are an expert medical scribe. Generate a SOAP note in JSON format.\n\n"
        f"### RETRIEVED EVIDENCE ###\n{evidence}\n"
        "### TASK ###\n"
        "Using the conversation and the evidence above, output a valid JSON object with keys:\n"
        '- "subjective": Patient reports, symptoms, history.\n'
        '- "objective": Vitals, physical exam, lab results.\n'
        '- "assessment": Diagnosis, clinical reasoning.\n'
        '- "plan": Treatment, medications, follow-up.\n\n'
        f"### CONVERSATION ###\n{conversation}\n\n"
        "### OUTPUT (JSON) ###"
    )
