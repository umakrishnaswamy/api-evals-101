"""One function to call any of the three providers: `chat(...)`.

Grok (xAI) and OpenAI share the OpenAI chat-completions schema; xAI just
uses a different base_url. Anthropic uses its Messages API. All three return
the same `LLMResult`, so the bot and the judge don't care who answered.

Weave auto-instruments the `openai` and `anthropic` clients once
`weave.init()` has run, so every call shows up in traces with tokens and
latency without any extra code here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import config


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str = ""


@lru_cache(maxsize=None)
def _openai_client(base_url: str | None, key_env: str):
    import openai

    return openai.OpenAI(api_key=os.environ[key_env], base_url=base_url)


@lru_cache(maxsize=None)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


_KEY_ENV = config.KEY_ENV
_BASE_URL = {"xai": "https://api.x.ai/v1", "openai": None}


def chat(provider: str, model: str, system: str, user: str,
         max_tokens: int = 700) -> LLMResult:
    """Single-turn chat call. `provider` is 'xai', 'openai' or 'anthropic'."""
    if config.uses_mock(provider):  # BACKEND=mock, or this provider has no key
        from mock_llm import mock_chat

        return mock_chat(provider, model, system, user)

    if provider in ("xai", "openai"):
        client = _openai_client(_BASE_URL[provider], _KEY_ENV[provider])
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        if config.TEMPERATURE is not None:
            kwargs["temperature"] = float(config.TEMPERATURE)
        resp = client.chat.completions.create(**kwargs)
        usage = resp.usage
        return LLMResult(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            response_model=getattr(resp, "model", model) or model,
        )

    if provider == "anthropic":
        client = _anthropic_client()
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if config.TEMPERATURE is not None:
            kwargs["temperature"] = float(config.TEMPERATURE)
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            response_model=getattr(resp, "model", model) or model,
        )

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Tool calling, used by 01_trace_conversations.py's agent loop.
#
# Messages use one provider-neutral shape:
#   {"role": "user" | "assistant", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}
# Tools: [{"name", "description", "parameters": <JSON schema>}]
# ---------------------------------------------------------------------------


@dataclass
class ToolTurn:
    text: str
    tool_calls: list[dict]          # [{"id", "name", "arguments": dict}]
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str = ""


def chat_with_tools(provider: str, model: str, system: str, messages: list[dict],
                    tools: list[dict], max_tokens: int = 700) -> ToolTurn:
    """One model step: either a final answer (no tool_calls) or tool requests."""
    if config.uses_mock(provider):
        from mock_llm import mock_chat_with_tools

        return mock_chat_with_tools(provider, model, system, messages, tools)

    if provider in ("xai", "openai"):
        import json

        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                             "content": m["content"]})
            elif m.get("tool_calls"):
                msgs.append({"role": "assistant", "content": m.get("content") or None,
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": json.dumps(c["arguments"])}}
                                            for c in m["tool_calls"]]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(model=model, messages=msgs,
                      tools=[{"type": "function", "function": t} for t in tools])
        if config.TEMPERATURE is not None:
            kwargs["temperature"] = float(config.TEMPERATURE)
        client = _openai_client(_BASE_URL[provider], _KEY_ENV[provider])
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [{"id": c.id, "name": c.function.name,
                  "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (msg.tool_calls or [])]
        usage = resp.usage
        return ToolTurn(text=msg.content or "", tool_calls=calls,
                        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        response_model=getattr(resp, "model", model) or model)

    if provider == "anthropic":
        msgs = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m["content"]}
                # Consecutive tool results go in one user message.
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
            elif m.get("tool_calls"):
                content = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"],
                             "input": c["arguments"]} for c in m["tool_calls"]]
                msgs.append({"role": "assistant", "content": content})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=msgs,
                      tools=[{"name": t["name"], "description": t["description"],
                              "input_schema": t["parameters"]} for t in tools])
        if config.TEMPERATURE is not None:
            kwargs["temperature"] = float(config.TEMPERATURE)
        resp = _anthropic_client().messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input)}
                 for b in resp.content if b.type == "tool_use"]
        return ToolTurn(text=text, tool_calls=calls,
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                        response_model=getattr(resp, "model", model) or model)

    raise ValueError(f"Unknown provider: {provider}")
