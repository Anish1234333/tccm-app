# utils/llm.py — pool-based LLM calling, shared by tccm_agent and council_agent
"""
build_pool()  → ordered (model_str, api_key) list, primary first then fallbacks
call_llm()    → tries each slot until one succeeds, raises RuntimeError if all fail
"""

import os
import time
import logging

import litellm
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
)

from models import FALLBACK_CHAIN, MAX_NEW_TOKENS

log = logging.getLogger(__name__)

# Suppress litellm's noisy provider-list banners
litellm.suppress_debug_info = True  # ty:ignore[invalid-assignment]


def build_pool(
    primary_model: str,
    primary_keys: list[str],
) -> list[tuple[str, str]]:
    """
    Returns an ordered list of (model_str, resolved_api_key) pairs.

    Layout:
      1. primary_model × each non-empty key in primary_keys
      2. every other entry in FALLBACK_CHAIN × their non-empty keys
         (skips primary_model to avoid duplicates)
    """
    pool: list[tuple[str, str]] = []

    # 1. Primary slots
    for env_var in primary_keys:
        key = os.environ.get(env_var, "").strip()
        if key:
            pool.append((primary_model, key))

    # 2. Fallback slots — everything except the primary
    for model_str, key_env_vars in FALLBACK_CHAIN:
        if model_str == primary_model:
            continue
        for env_var in key_env_vars:
            key = os.environ.get(env_var, "").strip()
            if key:
                pool.append((model_str, key))

    if not pool:
        raise RuntimeError(
            "No API keys found in environment. "
            "Set at least one of: GROQ_API_KEY_1, CEREBRAS_API_KEY_1, "
            "MISTRAL_API_KEY_1, SAMBANOVA_API_KEY_1."
        )

    return pool


# Exceptions that mean "this provider is busy — try the next one"
_RETRIABLE = (RateLimitError, ServiceUnavailableError, APIConnectionError)


def call_llm(
    pool: list[tuple[str, str]],
    messages: list[dict],
    max_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Iterate through pool trying each (model, key) pair.
    Returns the response text on first success.
    Raises RuntimeError if every slot fails.
    """
    last_exc: Exception = RuntimeError("Pool was empty")

    for idx, (model_str, api_key) in enumerate(pool):
        try:
            log.debug("Trying %s (slot %d/%d)", model_str, idx + 1, len(pool))
            resp = litellm.completion(
                model=model_str,
                messages=messages,
                max_tokens=max_tokens,
                api_key=api_key,
                num_retries=1,  # one quick internal retry before we move on
                timeout=120,
            )
            return resp.choices[0].message.content

        except _RETRIABLE as exc:
            log.warning(
                "Rate-limited on %s (slot %d/%d): %s — trying next slot",
                model_str,
                idx + 1,
                len(pool),
                exc,
            )
            last_exc = exc
            if idx < len(pool) - 1:
                time.sleep(0.5)  # tiny breath before next provider
            continue

        except Exception as exc:
            log.warning(
                "Error on %s (slot %d/%d): %s — trying next slot",
                model_str,
                idx + 1,
                len(pool),
                exc,
            )
            last_exc = exc
            continue

    raise RuntimeError(
        f"All {len(pool)} provider slots exhausted. Last error: {last_exc}"
    )
