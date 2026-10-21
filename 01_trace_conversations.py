"""01 — Trace a few support conversations with Weave's agent instrumentation.

Before evaluating, look at what the bot actually does. This sends five short
customer conversations through the bot on each model and records them with the
Weave agent SDK (OTel GenAI conventions).

Here the bot is a small tool-calling agent. The MODEL decides, every turn,
which of two tools to call (if any):

    search_help_center    the help-center articles (policies, prices, rules)
    get_account_details   the signed-in customer's plan, lines, AutoPay, outages

    invoke_agent acme-support-bot          <- one customer turn
    ├── chat <model>                       <- model asks for tools
    ├── execute_tool search_help_center
    ├── execute_tool get_account_details   <- only if the model wants it
    └── chat <model>                       <- model writes the answer

Then open the Agents tab: one agent, three model versions, fifteen
conversations. Compare how often each model reaches for each tool, and open a
turn to see whether the answer matches what the tools returned. A model that
answers "When does my discount start?" without looking up the account is
guessing about this customer.

(02_eval_three_models.py evaluates the simpler fixed-retrieval bot in
support_bot.py, so the comparison there isolates the model's answer quality.)

Usage:
  python 01_trace_conversations.py               # all three models
  python 01_trace_conversations.py anthropic     # one model
"""

from __future__ import annotations

import json
import sys

import weave
from weave.conversation import Message, ToolCallPart, Usage

import config
from llm import chat_with_tools
from support_bot import SYSTEM_PROMPT, search_help_center

AGENT_NAME = "acme-support-bot"
AGENT_DESCRIPTION = "RAG customer-support agent for Acme Mobile (workshop demo)"
MAX_STEPS = 4  # model calls per turn, so a tool loop can't run away

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "- Answer ONLY from the help-center articles provided.",
    "- Use search_help_center to find the help-center articles, and\n"
    "  get_account_details when the answer depends on THIS customer's account.\n"
    "- Answer ONLY from what the tools return.",
)

# The signed-in customer. Made up, like everything else in the workshop.
ACCOUNT = {
    "customer": "Jordan Rivera",
    "plan": "Essentials",
    "lines": 2,
    "autopay": {"enabled": True, "payment_method": "Visa credit card"},
    "international_pass": None,
    "recent_outages": [{"date": "yesterday", "duration_hours": 6}],
    "device": {"model": "Acme Pixel 8", "purchased_days_ago": 9},
}

TOOLS = [
    {
        "name": "search_help_center",
        "description": "Search Acme Mobile's help-center articles. Returns the top "
                       "matching articles (id, title, full text).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "What to look up, in plain words"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_account_details",
        "description": "Get the signed-in customer's account: plan, number of lines, "
                       "AutoPay payment method, travel passes, recent outages, device.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def run_tool(name: str, arguments: dict) -> list | dict | str:
    if name == "search_help_center":
        docs = search_help_center(arguments.get("query", ""), k=3)
        return [{"id": d["id"], "title": d["title"], "content": d["content"]}
                for d in docs] or "no articles found"
    if name == "get_account_details":
        return ACCOUNT
    return {"error": f"unknown tool {name}"}


def to_weave(m: dict) -> Message:
    """Our provider-neutral message -> a Weave Message for the chat view."""
    if m["role"] == "tool":
        return Message.tool_result(m["tool_call_id"], m["content"])
    if m.get("tool_calls"):
        return Message.assistant(m.get("content", ""), tool_calls=[
            ToolCallPart(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in m["tool_calls"]])
    return Message(role=m["role"], content=m["content"])


CONVERSATIONS = [
    ("autopay-credit-card", [
        "Hi, I just turned on AutoPay.",
        "When does the $5 discount kick in?",
    ]),
    ("travel-abroad", [
        "I'm heading to Mexico for 3 days.",
        "Will my data work, and what will it cost me?",
    ]),
    ("frustrated-outage", [
        "My service was out for 6 hours yesterday and I missed a work call.",
        "I want a credit. And honestly, can I sue you for this?",
    ]),
    ("return-new-phone", [
        "I don't love my new phone. Can I still return it?",
        "Is there a fee for that?",
    ]),
    ("general-pricing", [
        "How much is the Unlimited plan?",
        "And does home internet come with it?",
    ]),
]


def run_turn(conversation, name: str, history: list[dict],
             user_message: str) -> tuple[str, list[str]]:
    """One customer turn: let the model call tools until it answers."""
    cand = config.CANDIDATES[name]
    with conversation.start_turn(
        user_message=user_message,
        model=cand["model"],
        agent_name=AGENT_NAME,
        agent_id=name,
        agent_description=AGENT_DESCRIPTION,
        agent_version=cand["model"],          # groups turns by model in the Agents tab
        system_instructions=[AGENT_SYSTEM_PROMPT],
    ) as turn:
        turn.set_attributes({"app.channel": "chat", "app.candidate": name})
        messages = history + [{"role": "user", "content": user_message}]
        start = len(messages)
        answer = "(stopped: too many tool calls)"
        for _ in range(MAX_STEPS):
            with weave.start_llm(model=cand["model"], provider_name=cand["provider"],
                                 system_instructions=[AGENT_SYSTEM_PROMPT]) as llm:
                step = chat_with_tools(cand["provider"], cand["model"],
                                       AGENT_SYSTEM_PROMPT, messages, TOOLS)
                reply = {"role": "assistant", "content": step.text,
                         "tool_calls": step.tool_calls}
                llm.input_messages = [to_weave(m) for m in messages]
                llm.output_messages = [to_weave(reply)]
                llm.record(usage=Usage(input_tokens=step.input_tokens,
                                       output_tokens=step.output_tokens),
                           response_model=step.response_model or cand["model"])
            messages.append(reply)
            if not step.tool_calls:
                answer = step.text
                break
            for call in step.tool_calls:
                with weave.start_tool(name=call["name"], arguments=call["arguments"],
                                      tool_call_id=call["id"]) as tool:
                    result = run_tool(call["name"], call["arguments"])
                    tool.result = result
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "name": call["name"], "content": json.dumps(result)})
        turn.record(output_messages=[to_weave(m) for m in messages[start:]])
        tools_used = [c["name"] for m in messages[start:] for c in m.get("tool_calls") or []]
        history.extend([{"role": "user", "content": user_message},
                        {"role": "assistant", "content": answer}])
        return answer, tools_used


def main(names: list[str]) -> None:
    config.warn_about_mocks(names, judge=False)
    weave.init(config.project())

    for name in names:
        label = f"{name}-mock" if config.uses_mock(config.CANDIDATES[name]["provider"]) else name
        print(f"\n######## {label} ({config.CANDIDATES[name]['model']}) ########")
        for conv_name, turns in CONVERSATIONS:
            print(f"=== {conv_name} ===")
            with weave.start_conversation(agent_name=AGENT_NAME,
                                          conversation_name=f"{conv_name}__{label}") as conversation:
                history: list[dict] = []
                for msg in turns:
                    reply, tools_used = run_turn(conversation, name, history, msg)
                    print(f"  USER : {msg[:80]}")
                    print(f"  TOOLS: {', '.join(tools_used) or '(none)'}")
                    print(f"  BOT  : {reply[:80]!r}\n")

    print("Done. Open the Weave UI -> Agents -> acme-support-bot.")


if __name__ == "__main__":
    main(sys.argv[1:] or list(config.CANDIDATES))
