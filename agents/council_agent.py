# agents/council_agent.py
import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from models import CONSOLIDATION_CHAIN, MAX_NEW_TOKENS, MAX_SHEETS_CHARS
from utils.llm import build_pool, call_llm


# ── State schema ──────────────────────────────────────────────────────────────


class CouncilState(TypedDict):
    sheet1: list
    sheet2: list
    sheet3: list
    consolidated: list


# ── Node ──────────────────────────────────────────────────────────────────────


def consolidate(state: CouncilState) -> dict:
    from prompts import CONSOLIDATION_PROMPT

    # Build pool from consolidation chain (first entry is primary)
    primary_model, primary_keys = CONSOLIDATION_CHAIN[0]
    pool = build_pool(primary_model, primary_keys)

    prompt = CONSOLIDATION_PROMPT.format(
        sheet1=json.dumps(state["sheet1"], indent=2)[:MAX_SHEETS_CHARS],
        sheet2=json.dumps(state["sheet2"], indent=2)[:MAX_SHEETS_CHARS],
        sheet3=json.dumps(state["sheet3"], indent=2)[:MAX_SHEETS_CHARS],
    )

    raw = call_llm(
        pool,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
    )

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [{"error": "consolidation_parse_failed", "raw": raw[:600]}]

    return {"consolidated": parsed if isinstance(parsed, list) else [parsed]}


# ── Compiled graph ────────────────────────────────────────────────────────────

council_graph = (
    StateGraph(CouncilState)  # ty:ignore[invalid-argument-type]
    .add_node("consolidate", consolidate)
    .add_edge(START, "consolidate")
    .add_edge("consolidate", END)
    .compile()
)
