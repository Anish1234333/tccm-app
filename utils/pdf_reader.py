# utils/pdf_reader.py
import os
import fitz  # PyMuPDF
from models import MAX_PAPER_CHARS


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text[:MAX_PAPER_CHARS]


def load_papers(pdf_paths: list[str]) -> list[dict]:
    return [
        {"paper_id": os.path.splitext(os.path.basename(p))[0], "text": extract_text(p)}
        for p in pdf_paths
    ]


def load_inventory(inventory_path: str) -> str:
    if inventory_path.endswith(".xlsx"):
        import pandas as pd

        df = pd.read_excel(inventory_path)
        return df.to_string(index=False)[:3_000]
    with open(inventory_path, "r", encoding="utf-8") as f:
        return f.read()[:3_000]
