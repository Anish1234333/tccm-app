# models.py — HuggingFace Serverless Inference model registry
# All models are free-tier accessible with a HF_TOKEN

EXTRACTION_MODELS = {
    "MiniLM": "microsoft/MiniLM-L12-H384-uncased",
    "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

CONSOLIDATION_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Shared inference settings
MAX_NEW_TOKENS = 2048
MAX_PAPER_CHARS = 12_000  # ~3 k tokens — fits every model's context window
MAX_SHEETS_CHARS = 4_000  # per-sheet truncation for consolidation prompt
