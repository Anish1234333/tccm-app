EXTRACTION_MODELS = {
    "Llama-Groq":     ("groq/llama-3.1-8b-instant",                 "GROQ_API_KEY"),
    "Llama-Cerebras": ("cerebras/llama3.1-8b",                      "CEREBRAS_API_KEY"),
    "Qwen-OR":        ("openrouter/qwen/qwen-2.5-7b-instruct:free",  "OPENROUTER_API_KEY"),
}
CONSOLIDATION_MODEL = ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY")

MAX_NEW_TOKENS = 2048
MAX_PAPER_CHARS = 12_000
MAX_SHEETS_CHARS = 4_000