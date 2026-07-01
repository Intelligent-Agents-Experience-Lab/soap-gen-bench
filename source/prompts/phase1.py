"""Phase 1 prompt strategies H1–H5 as plain format strings."""

PROMPTS: dict[str, str] = {
    # H1: Zero-shot baseline
    "H1_Baseline": (
        "Summarize the following clinical conversation into a SOAP note.\n\n"
        "Conversation:\n{conversation}"
    ),

    # H2: Explicit JSON schema
    "H2_Structured": (
        "Generate a SOAP note from the clinical conversation.\n"
        "Output MUST be a valid JSON object with detailed content for the following keys: \n"
        '"subjective", "objective", "assessment", "plan".\n\n'
        "Conversation:\n{conversation}"
    ),

    # H3: One-shot example
    "H3_FewShot": (
        "Example Input:\n"
        "Doctor: How are you?\n"
        "Patient: My head hurts and I have a runny nose.\n"
        "Doctor: Fever?\n"
        "Patient: Yes, 100.4.\n"
        "Doctor: Looks like a viral URI. Rest and fluids.\n\n"
        "Example Output:\n"
        '{{\n'
        '  "subjective": "Patient reports headache and rhinorrhea.",\n'
        '  "objective": "Temp 100.4F.",\n'
        '  "assessment": "Viral URI.",\n'
        '  "plan": "Rest, hydration."\n'
        "}}\n\n"
        "Task: Generate a SOAP note for the conversation below in JSON format.\n\n"
        "Conversation:\n{conversation}"
    ),

    # H4: Dynamic / RAG placeholder (static stand-in — real retrieval in Phase 2)
    "H4_Dynamic": (
        "Preamble: You are an expert medical scribe. Here are similar past examples to guide you.\n\n"
        "[Inserted Similar Examples]\n"
        "Subjective: Cp/sob...\n"
        "...\n"
        "[End Examples]\n\n"
        "Now processing current case:\n"
        "Conversation:\n{conversation}"
    ),

    # H5: Chain-of-Thought decomposition
    "H5_CoT": (
        "Task: Generate a SOAP note.\n\n"
        "Strategy:\n"
        "1. First, extract the subjective symptoms and patient history.\n"
        "2. Second, list the objective findings and vitals.\n"
        "3. Third, reason about the assessment/diagnosis.\n"
        "4. Fourth, formulate the treatment plan.\n"
        "5. Finally, output the distinct SOAP sections.\n\n"
        "Conversation:\n{conversation}"
    ),
}


def format_prompt(prompt_name: str, conversation: str) -> str:
    template = PROMPTS[prompt_name]
    return template.format(conversation=conversation)
