# models.py — provider pools + model registry
# Add more keys by appending _2, _3, etc. to each list below.

# ── Per-provider key pools ────────────────────────────────────────────────────
# Each string is an env-var name. Add as many accounts as you have.
GROQ_KEYS = ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
CEREBRAS_KEYS = ["CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY_2"]
MISTRAL_KEYS = ["MISTRAL_API_KEY_1"]
SAMBANOVA_KEYS = ["SAMBANOVA_API_KEY_1"]

# ── Global fallback chain (order = priority when primary is exhausted) ────────
# Every (model_str, key_pool) pair is tried in sequence.
FALLBACK_CHAIN: list[tuple[str, list[str]]] = [
    ("groq/llama-3.1-8b-instant", GROQ_KEYS),
    ("cerebras/llama3.1-8b", CEREBRAS_KEYS),
    ("mistral/mistral-small-latest", MISTRAL_KEYS),
    ("sambanova/Meta-Llama-3.3-70B-Instruct", SAMBANOVA_KEYS),
]

# ── Extraction council — each model has a preferred primary ───────────────────
# Value: (primary_model_str, primary_key_pool)
# Falls back to full FALLBACK_CHAIN automatically (see utils/llm.py).
EXTRACTION_MODELS: dict[str, tuple[str, list[str]]] = {
    "Llama-Groq": ("groq/llama-3.1-8b-instant", GROQ_KEYS),
    "Llama-Cerebras": ("cerebras/llama3.1-8b", CEREBRAS_KEYS),
    "Llama-Mistral": ("mistral/mistral-small-latest", MISTRAL_KEYS),
}

# ── Consolidation chain (70B models for better synthesis quality) ─────────────
CONSOLIDATION_CHAIN: list[tuple[str, list[str]]] = [
    ("groq/llama-3.3-70b-versatile", GROQ_KEYS),
    ("cerebras/llama3.1-70b", CEREBRAS_KEYS),
    ("mistral/mistral-large-latest", MISTRAL_KEYS),
]

# ── Display label used in app.py UI ──────────────────────────────────────────
CONSOLIDATION_MODEL = "Llama-3.3-70B (Groq → Cerebras → Mistral)"

# ── Inference settings ────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 2048
MAX_PAPER_CHARS = 12_000  # ~3k tokens — fits every model's context window
MAX_SHEETS_CHARS = 4_000  # per-sheet truncation for consolidation prompt
