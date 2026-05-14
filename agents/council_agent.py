# agents/council_agent.py  [WAIVER: none — well within 60-line ceiling]
import os
import json
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from models import CONSOLIDATION_MODEL, MAX_NEW_TOKENS, MAX_SHEETS_CHARS
import litellm
import operator
from dataclasses import dataclass, field
# ── State schema ─────────────────────────────────────────────────────────────


@dataclass
class CouncilState:
    sheet1: list
    sheet2: list
    sheet3: list
    consolidated: Annotated[list, operator.add] = field(default_factory=list)


# ── Node ─────────────────────────────────────────────────────────────────────


def consolidate(state):
    model_str, api_key_name = CONSOLIDATION_MODEL  # unpack tuple
    from prompts import CONSOLIDATION_PROMPT

    prompt = CONSOLIDATION_PROMPT.format(
        sheet1=json.dumps(state["sheet1"], indent=2)[:MAX_SHEETS_CHARS],
        sheet2=json.dumps(state["sheet2"], indent=2)[:MAX_SHEETS_CHARS],
        sheet3=json.dumps(state["sheet3"], indent=2)[:MAX_SHEETS_CHARS],
    )
    resp = litellm.completion(
        model=model_str,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        api_key=os.environ.get(api_key_name, ""),
        fallbacks=[
            {
                "model": "cerebras/llama3.1-70b",
                "api_key": os.environ.get("CEREBRAS_API_KEY", ""),
            },
        ],
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
