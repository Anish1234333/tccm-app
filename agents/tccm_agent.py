import os
import json
import operator

# from typing import TypedDict, Annotated
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import litellm
from models import MAX_NEW_TOKENS
import time

# tccm_agent.py
from dataclasses import dataclass, field


@dataclass
class GraphState:
    papers: list[dict]
    prompt_template: str
    model_id: tuple[str, str]
    journal: str
    inventory: str
    results: Annotated[list, operator.add] = field(default_factory=list)


@dataclass
class PaperState:
    paper: dict
    prompt_template: str
    model_id: tuple[str, str]
    journal: str
    inventory: str
    results: Annotated[list, operator.add] = field(default_factory=list)


def dispatch(state: GraphState) -> list[Send]:
    sends = []
    for i, p in enumerate(state.papers):  # state.papers not state["papers"]
        if i > 0:
            time.sleep(2)
        sends.append(
            Send(
                "extract",
                PaperState(  # construct PaperState explicitly
                    paper=p,
                    prompt_template=state.prompt_template,
                    model_id=state.model_id,
                    journal=state.journal,
                    inventory=state.inventory,
                    results=[],
                ),
            )
        )
    return sends


def extract(state: PaperState) -> dict:
    model_str, api_key_name = state.model_id

    prompt = (
        state.prompt_template.replace("{paper_text}", state.paper["text"])
        .replace("{inventory}", state.inventory or "")
        .replace("{journal}", state.journal or "IS Journal")
    )

    resp = litellm.completion(
        model=model_str,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        api_key=os.environ.get(api_key_name, ""),
        fallbacks=[
            {
                "model": "cerebras/llama3.1-8b",
                "api_key": os.environ.get("CEREBRAS_API_KEY", ""),
            },
            {
                "model": "openrouter/qwen/qwen-2.5-7b-instruct:free",
                "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            },
        ],
        num_retries=3,
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [
            {
                "paper_id": state.paper["paper_id"],
                "raw_output": raw[:800],
                "parse_error": True,
            }
        ]
    return {"results": parsed if isinstance(parsed, list) else [parsed]}


tccm_graph = (
    StateGraph(GraphState)
    .add_node("extract", extract)
    .add_conditional_edges(START, dispatch, ["extract"])
    .add_edge("extract", END)
    .compile()
)
