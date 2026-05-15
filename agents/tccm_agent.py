# agents/tccm_agent.py
import json
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models import MAX_NEW_TOKENS
from utils.llm import build_pool, call_llm


# ── State schemas ─────────────────────────────────────────────────────────────


class GraphState(TypedDict):
    papers: list[dict]
    prompt_template: str
    model_id: tuple[str, list[str]]  # (model_str, key_env_var_list)
    journal: str
    inventory: str
    results: Annotated[list, operator.add]  # reducer accumulates


class PaperState(TypedDict):
    paper: dict
    prompt_template: str
    model_id: tuple[str, list[str]]
    journal: str
    inventory: str
    results: Annotated[list, operator.add]


# ── Nodes ─────────────────────────────────────────────────────────────────────


def dispatch(state: GraphState) -> list[Send]:
    """Fan-out: one Send per paper, no throttling needed — pool handles retries."""
    return [
        Send("extract", {**state, "paper": p, "results": []}) for p in state["papers"]
    ]


def extract(state: PaperState) -> dict:
    """Single responsibility: call LLM via pool, parse JSON, return result list."""
    primary_model, primary_keys = state["model_id"]
    pool = build_pool(primary_model, primary_keys)

    prompt = (
        state["prompt_template"]
        .replace("{paper_text}", state["paper"]["text"])
        .replace("{inventory}", state.get("inventory", ""))
        .replace("{journal}", state.get("journal", "IS Journal"))
    )

    raw = call_llm(
        pool,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
    )

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [
            {
                "paper_id": state["paper"]["paper_id"],
                "raw_output": raw[:800],
                "parse_error": True,
            }
        ]

    return {"results": parsed if isinstance(parsed, list) else [parsed]}


# ── Compiled graph ────────────────────────────────────────────────────────────

tccm_graph = (
    StateGraph(GraphState)  # ty:ignore[invalid-argument-type]
    .add_node("extract", extract)
    .add_conditional_edges(START, dispatch, ["extract"])
    .add_edge("extract", END)
    .compile()
)
