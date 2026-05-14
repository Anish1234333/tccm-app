import os, json, operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import litellm
from models import MAX_NEW_TOKENS

class GraphState(TypedDict):
    papers:          list[dict]
    prompt_template: str
    model_id:        tuple[str, str]   # (litellm model string, env key name)
    journal:         str
    inventory:       str
    results:         Annotated[list, operator.add]

class PaperState(TypedDict):
    paper:           dict
    prompt_template: str
    model_id:        tuple[str, str]
    journal:         str
    inventory:       str
    results:         Annotated[list, operator.add]

def dispatch(state: GraphState) -> list[Send]:
    return [Send("extract", {**state, "paper": p, "results": []}) for p in state["papers"]]

def extract(state: PaperState) -> dict:
    model_str, api_key_name = state["model_id"]
    
    prompt = (
        state["prompt_template"]
        .replace("{paper_text}", state["paper"]["text"])
        .replace("{inventory}",  state.get("inventory", ""))
        .replace("{journal}",    state.get("journal", "IS Journal"))
    )

    resp = litellm.completion(
        model=model_str,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        api_key=os.environ.get(api_key_name, ""),
        fallbacks=[
            {"model": "cerebras/llama3.1-8b", "api_key": os.environ.get("CEREBRAS_API_KEY", "")},
            {"model": "openrouter/qwen/qwen-2.5-7b-instruct:free", "api_key": os.environ.get("OPENROUTER_API_KEY", "")},
        ],
        num_retries=3,
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [{"paper_id": state["paper"]["paper_id"], "raw_output": raw[:800], "parse_error": True}]
    return {"results": parsed if isinstance(parsed, list) else [parsed]}

tccm_graph = (
    StateGraph(GraphState)  # ty:ignore[invalid-argument-type]
    .add_node("extract", extract)
    .add_conditional_edges(START, dispatch, ["extract"])
    .add_edge("extract", END)
    .compile()
)