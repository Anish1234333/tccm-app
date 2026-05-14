# agents/council_agent.py  [WAIVER: none — well within 60-line ceiling]
import os, json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from huggingface_hub import InferenceClient
from models import CONSOLIDATION_MODEL_ID, MAX_NEW_TOKENS, MAX_SHEETS_CHARS


# ── State schema ─────────────────────────────────────────────────────────────


class CouncilState(TypedDict):
    sheet1: list
    sheet2: list
    sheet3: list
    consolidated: list


# ── Node ─────────────────────────────────────────────────────────────────────


def consolidate(state: CouncilState) -> dict:
    """Single responsibility: call consolidation LLM, return tagged rows."""
    from prompts import CONSOLIDATION_PROMPT  # imported here to avoid circular

    client = InferenceClient(token=os.environ.get("HF_TOKEN", ""))
    prompt = CONSOLIDATION_PROMPT.format(
        sheet1=json.dumps(state["sheet1"], indent=2)[:MAX_SHEETS_CHARS],
        sheet2=json.dumps(state["sheet2"], indent=2)[:MAX_SHEETS_CHARS],
        sheet3=json.dumps(state["sheet3"], indent=2)[:MAX_SHEETS_CHARS],
    )
    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=CONSOLIDATION_MODEL_ID,
        max_tokens=MAX_NEW_TOKENS * 2,
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [{"error": "consolidation_parse_failed", "raw": raw[:600]}]
    return {"consolidated": parsed if isinstance(parsed, list) else [parsed]}


# ── Compiled graph ───────────────────────────────────────────────────────────

council_graph = (
    StateGraph(CouncilState)
    .add_node("consolidate", consolidate)
    .add_edge(START, "consolidate")
    .add_edge("consolidate", END)
    .compile()
)
