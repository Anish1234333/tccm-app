# agents/tccm_agent.py  [WAIVER: none — well within 80-line ceiling]
import os, json, operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from huggingface_hub import InferenceClient
from models import MAX_NEW_TOKENS


# ── State schemas ────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    papers:           list[dict]   # [{paper_id, text}, ...]
    prompt_template:  str          # fully-formatted RCTO prompt string (with {placeholders})
    model_id:         str          # HF model repo id
    journal:          str
    inventory:        str          # 159-theory text, empty for Characteristics/Method
    results:          Annotated[list, operator.add]   # reducer accumulates


class PaperState(TypedDict):
    paper:            dict
    prompt_template:  str
    model_id:         str
    journal:          str
    inventory:        str
    results:          Annotated[list, operator.add]


# ── Nodes ────────────────────────────────────────────────────────────────────

def dispatch(state: GraphState) -> list[Send]:
    """Fan-out: one Send per paper — no for-loop in agent logic."""
    return [
        Send("extract", {**state, "paper": p, "results": []})
        for p in state["papers"]
    ]


def extract(state: PaperState) -> dict:
    """Single responsibility: call LLM, parse JSON, return result list."""
    client = InferenceClient(token=os.environ.get("HF_TOKEN", ""))
    prompt = state["prompt_template"].format(
        paper_text = state["paper"]["text"],
        inventory  = state.get("inventory", ""),
        journal    = state.get("journal", "IS Journal"),
    )
    resp = client.chat_completion(
        messages   = [{"role": "user", "content": prompt}],
        model      = state["model_id"],
        max_tokens = MAX_NEW_TOKENS,
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [{"paper_id": state["paper"]["paper_id"],
                   "raw_output": raw[:800], "parse_error": True}]
    return {"results": parsed if isinstance(parsed, list) else [parsed]}


# ── Compiled graph ───────────────────────────────────────────────────────────

tccm_graph = (
    StateGraph(GraphState)
    .add_node("extract", extract)
    .add_conditional_edges(START, dispatch, ["extract"])
    .add_edge("extract", END)
    .compile()
)