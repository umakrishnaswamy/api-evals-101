"""BACKEND=mock: a no-API-key stand-in. NOT a model.

Use it to check plumbing (Weave login, datasets, scorers, the compare view)
before the workshop, or if a provider is down. It returns the same scripted
answer for all three "models", so mock results say NOTHING about which model
is better. 02_eval_three_models.py skips the capability suite on mock, because
scripted answers to hard questions measure nothing.
"""

from __future__ import annotations

import json
import re

from llm import LLMResult


def mock_chat(provider: str, model: str, system: str, user: str) -> LLMResult:
    if system.startswith("You are a strict QA reviewer"):
        return LLMResult(text=json.dumps({"verdict": "yes", "reason": "mock judge"}),
                         response_model=f"mock:{model}")

    ids = re.findall(r"^\[([a-z0-9-]+)\]", user, flags=re.MULTILINE)
    if not ids:
        text = ("I'm sorry, I can't help with that here. I can connect you with a "
                "human agent who can take a closer look.\nSources: none")
        return LLMResult(text=text, response_model=f"mock:{model}")

    # Restate the top article verbatim, then cite it.
    first = user.split(f"[{ids[0]}]", 1)[1].split("\n\n[", 1)[0]
    body = " ".join(first.strip().splitlines()[1:]).strip()
    text = f"Here's what our help center says: {body}\nSources: {ids[0]}"
    return LLMResult(text=text, input_tokens=len(user) // 4,
                     output_tokens=len(text) // 4, response_model=f"mock:{model}")


def mock_chat_with_tools(provider: str, model: str, system: str,
                         messages: list[dict], tools: list[dict]):
    """Scripted tool use: search first, look up the account if the customer
    says "my"/"I", then answer from the articles. Same for every "model"."""
    from llm import ToolTurn

    i = max(n for n, m in enumerate(messages) if m["role"] == "user")
    question, after = messages[i]["content"], messages[i + 1:]
    used = {c["name"] for m in after for c in m.get("tool_calls", [])}
    if "search_help_center" not in used:
        call = {"id": f"mock_{len(messages)}", "name": "search_help_center",
                "arguments": {"query": question}}
        return ToolTurn(text="", tool_calls=[call], response_model=f"mock:{model}")
    if "get_account_details" not in used and re.search(r"\b(my|I|I'm)\b", question):
        call = {"id": f"mock_{len(messages)}a", "name": "get_account_details",
                "arguments": {}}
        return ToolTurn(text="", tool_calls=[call], response_model=f"mock:{model}")
    articles = next((m["content"] for m in after if m.get("name") == "search_help_center"), "")
    try:
        docs = json.loads(articles)
    except json.JSONDecodeError:
        docs = []
    user = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in docs)
    r = mock_chat(provider, model, "", user)
    return ToolTurn(text=r.text, tool_calls=[], input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens, response_model=r.response_model)
