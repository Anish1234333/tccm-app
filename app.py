# app.py  [WAIVER: none — well within 100-line ceiling]
import os, json, tempfile
import gradio as gr
from prompts import THEORY_PROMPT, CHARACTERISTICS_PROMPT, METHOD_PROMPT
from models import EXTRACTION_MODELS, CONSOLIDATION_MODEL_ID
from agents.tccm_agent import tccm_graph
from agents.council_agent import council_graph
from utils.pdf_reader import load_papers, load_inventory
from utils.excel_writer import export_excel

PROMPT_MAP = {
    "Theory": THEORY_PROMPT,
    "Characteristics": CHARACTERISTICS_PROMPT,
    "Method": METHOD_PROMPT,
}
MODEL_NAMES = list(
    EXTRACTION_MODELS.keys()
)  # ["Mistral-7B", "Zephyr-7B", "Qwen2.5-7B"]


# ── Core orchestration (UI glue — not agent logic) ───────────────────────────


def _extract_one(pdfs, inventory_file, journal, prompt_type, model_name):
    papers = load_papers([p.name for p in pdfs]) if pdfs else []
    inventory = load_inventory(inventory_file.name) if inventory_file else ""
    result = tccm_graph.invoke(
        {
            "papers": papers,
            "prompt_template": PROMPT_MAP[prompt_type],
            "model_id": EXTRACTION_MODELS[model_name],
            "journal": journal,
            "inventory": inventory,
            "results": [],
        }  # ty:ignore[invalid-argument-type]
    )
    return result["results"]


def run_council(pdfs, inventory_file, journal, prompt_type, progress=gr.Progress()):
    progress(0.05, desc=f"Extracting with {MODEL_NAMES[0]}…")
    s1 = _extract_one(pdfs, inventory_file, journal, prompt_type, MODEL_NAMES[0])
    progress(0.38, desc=f"Extracting with {MODEL_NAMES[1]}…")
    s2 = _extract_one(pdfs, inventory_file, journal, prompt_type, MODEL_NAMES[1])
    progress(0.68, desc=f"Extracting with {MODEL_NAMES[2]}…")
    s3 = _extract_one(pdfs, inventory_file, journal, prompt_type, MODEL_NAMES[2])
    progress(0.85, desc=f"Consolidating with {CONSOLIDATION_MODEL_ID}…")
    s4 = council_graph.invoke(
        {"sheet1": s1, "sheet2": s2, "sheet3": s3, "consolidated": []}  # ty:ignore[invalid-argument-type]
    )["consolidated"]
    progress(0.95, desc="Writing Excel…")
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, prefix="tccm_") as f:
        xlsx = export_excel(s1, s2, s3, s4, f.name)
    progress(1.0, desc="Done ✓")
    return (
        json.dumps(s1[:5], indent=2),
        json.dumps(s2[:5], indent=2),
        json.dumps(s3[:5], indent=2),
        json.dumps(s4[:5], indent=2),
        xlsx,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Agentic TCCM Extractor — LLM Council") as demo:
    gr.Markdown(
        "# 📚 Agentic TCCM Extractor\n"
        f"**Council:** {MODEL_NAMES[0]} · {MODEL_NAMES[1]} · {MODEL_NAMES[2]} → "
        f"**Consolidator:** `{CONSOLIDATION_MODEL_ID}`  ·  Set `HF_TOKEN` in Space secrets."
    )
    with gr.Row():
        with gr.Column(scale=1):
            pdfs = gr.File(
                label="Paper PDFs", file_count="multiple", file_types=[".pdf"]
            )
            inventory_f = gr.File(
                label="IS_Theories (.xlsx / .txt)", file_types=[".xlsx", ".txt", ".csv"]
            )
            journal = gr.Textbox(label="Journal Name", value="MIS Quarterly")
            prompt_type = gr.Radio(
                ["Theory", "Characteristics", "Method"],
                label="Extraction Prompt",
                value="Theory",
            )
            run_btn = gr.Button("▶ Run Full LLM Council", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(f"Sheet 1 — {MODEL_NAMES[0]}"):
                    out1 = gr.JSON(label="Preview (first 5 rows)")
                with gr.Tab(f"Sheet 2 — {MODEL_NAMES[1]}"):
                    out2 = gr.JSON(label="Preview (first 5 rows)")
                with gr.Tab(f"Sheet 3 — {MODEL_NAMES[2]}"):
                    out3 = gr.JSON(label="Preview (first 5 rows)")
                with gr.Tab("Sheet 4 — Consolidated"):
                    out4 = gr.JSON(label="Preview (first 5 rows)")
            xlsx_out = gr.File(label="⬇ Download 4-Sheet Excel")

    run_btn.click(
        fn=run_council,
        inputs=[pdfs, inventory_f, journal, prompt_type],
        outputs=[out1, out2, out3, out4, xlsx_out],
    )

if __name__ == "__main__":
    demo.launch()
