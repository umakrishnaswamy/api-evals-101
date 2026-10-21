"""The app under test: a RAG customer-support bot for a fictional carrier, Acme Mobile.

Shape:
    SupportBot.predict
    ├── search_help_center   (keyword retriever over 10 help-center articles)
    └── answer               (one LLM call; the provider/model is the variable)

The retriever, prompt and corpus are identical for every model, so when the
three eval runs differ, the difference is the model.

Everything about Acme is made up for this workshop.
"""

from __future__ import annotations

import re

import weave

import config
from llm import LLMResult, chat

# ---------------------------------------------------------------------------
# Help-center corpus. Several articles hide a condition or exclusion in the
# middle of the paragraph. Those are where models slip, and the capability
# suite below goes straight at them.
# ---------------------------------------------------------------------------

DOCUMENTS: list[dict] = [
    {"id": "plans-pricing", "title": "Acme Mobile plans and pricing",
     "content": "Acme offers two plans. Essentials is $40 per line per month with 15GB of "
                "high-speed data. Unlimited is $65 per line per month. On Unlimited, after "
                "50GB of use in a billing cycle, your data may be deprioritized (slowed during "
                "network congestion); it is never cut off. Taxes and fees are extra."},
    {"id": "autopay-discount", "title": "AutoPay discount",
     "content": "Customers enrolled in AutoPay get $5 off per line per month, up to 5 lines. "
                "The discount applies only when AutoPay is linked to a debit card or bank "
                "account; AutoPay with a credit card does not qualify. The discount appears "
                "within two bill cycles of enrolling."},
    {"id": "refund-policy", "title": "Refunds for service charges",
     "content": "You can request a refund for a service charge within 30 days of the bill date. "
                "Refunds go back to the original payment method and take 5 to 7 business days. "
                "Prorated refunds are not given for partial months after you cancel a line."},
    {"id": "device-returns", "title": "Returning a phone or device",
     "content": "Devices can be returned within 14 days of delivery. A $35 restocking fee "
                "applies to opened devices unless the device is defective. Start a return in the "
                "Acme app under Orders, then ship the device with the prepaid label."},
    {"id": "billing-disputes", "title": "Disputing a charge on your bill",
     "content": "To dispute a charge, contact support within 60 days of the bill date on which "
                "the charge first appeared. You do not need to pay the disputed amount while it "
                "is under review, but pay the rest of your bill to avoid late fees."},
    {"id": "international-roaming", "title": "Using your phone abroad",
     "content": "Unlimited includes free texting and 2G-speed data in over 200 countries. "
                "Essentials does not include international roaming; add a Travel Pass for $10 "
                "per day. Calls abroad cost $0.25 per minute on both plans."},
    {"id": "outage-credits", "title": "Service outage credits",
     "content": "If an outage affects your service for more than 24 consecutive hours, you can "
                "request a credit for each affected day. Credits must be requested within 30 days "
                "of the outage. Outages shorter than 24 hours are not eligible."},
    {"id": "sim-swap-security", "title": "Protecting your account from SIM swaps",
     "content": "Set an account PIN in the Acme app under Security. Acme will never ask for your "
                "PIN by text or email. A SIM change requires your PIN or an in-store ID check. "
                "If you get an unexpected 'SIM changed' alert, call support immediately."},
    {"id": "trade-in", "title": "Trading in a device",
     "content": "Trade-in credit is applied over 24 monthly bill credits, not as a lump sum. "
                "If you cancel the line before 24 months, the remaining credits stop. The device "
                "must power on and have no cracked screen to qualify."},
    {"id": "home-internet", "title": "Acme 5G Home Internet",
     "content": "5G Home Internet is $50 per month, or $35 per month when bundled with an "
                "Unlimited line. Availability depends on your address; check it on the website "
                "before ordering. There is no annual contract."},
]

_DOC_BY_ID = {d["id"]: d for d in DOCUMENTS}

# ---------------------------------------------------------------------------
# Two suites, kept separate on purpose (they answer different questions).
#
#   REGRESSION  — behavior a good bot should already get right. Should sit near
#                 100%. The interesting day is the day one goes red.
#   CAPABILITY  — the hard cases: buried exclusions, wrong premises, questions
#                 the help center can't answer. Should start lower. That's the
#                 hill to climb, and where models really separate.
#
# Row fields (every field is used by at least one scorer):
#   expected_doc_ids  articles that must be retrieved and cited
#   must_include      facts the answer must state; each item is a list of
#                     acceptable phrasings ("any of these counts")
#   must_not_include  known wrong claims (traps) that must NOT appear
#   should_answer     False = out of scope or not covered: decline/escalate
# ---------------------------------------------------------------------------

REGRESSION: list[dict] = [
    {"id": "r-plan-price", "question": "How much is the Unlimited plan per line?",
     "expected_doc_ids": ["plans-pricing"], "must_include": [["$65"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-essentials-data", "question": "How much high-speed data do I get on Essentials?",
     "expected_doc_ids": ["plans-pricing"], "must_include": [["15gb", "15 gb"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-refund-window", "question": "How long do I have to ask for a refund on a service charge?",
     "expected_doc_ids": ["refund-policy"], "must_include": [["30 days"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-return-window", "question": "Can I return my new phone? I got it last week.",
     "expected_doc_ids": ["device-returns"], "must_include": [["14 days"], ["$35"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-dispute-steps", "question": "There's a charge on my bill I don't recognize. What do I do?",
     "expected_doc_ids": ["billing-disputes"], "must_include": [["60 days"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-sim-pin", "question": "How do I stop someone from swapping my SIM?",
     "expected_doc_ids": ["sim-swap-security"], "must_include": [["pin"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-home-internet", "question": "What does 5G Home Internet cost if I already have Unlimited?",
     "expected_doc_ids": ["home-internet"], "must_include": [["$35"]],
     "must_not_include": [], "should_answer": True},
    {"id": "r-oos-legal", "question": "Can I sue Acme for the outage last month?",
     "expected_doc_ids": [], "must_include": [], "must_not_include": [],
     "should_answer": False},
]

CAPABILITY: list[dict] = [
    {   # Buried exclusion: $5 off is right, but not with a credit card.
        "id": "c-autopay-credit-card",
        "question": "I set up AutoPay with my Visa credit card. When will I see the $5 discount?",
        "expected_doc_ids": ["autopay-discount"],
        "must_include": [["debit", "bank account"]],
        "must_not_include": ["credit card qualifies", "credit cards qualify", "credit card is eligible"],
        "should_answer": True},
    {   # Wrong premise: the 60 days runs from the bill date, not when you noticed.
        "id": "c-dispute-premise",
        "question": "I just noticed a weird charge from 3 months ago. I have 60 days from today to dispute it, right?",
        "expected_doc_ids": ["billing-disputes"],
        "must_include": [["bill date"]],
        "must_not_include": ["60 days from today", "yes, you have 60 days"],
        "should_answer": True},
    {   # False certainty: "unlimited" data is deprioritized after 50GB.
        "id": "c-unlimited-really",
        "question": "Is the Unlimited plan really unlimited? I stream a lot.",
        "expected_doc_ids": ["plans-pricing"],
        "must_include": [["50gb", "50 gb"], ["deprioritiz", "slow"]],
        "must_not_include": ["no limits", "never slowed"],
        "should_answer": True},
    {   # Not covered: the help center says nothing about early termination fees.
        "id": "c-etf-not-covered",
        "question": "What's the early termination fee if I cancel my contract?",
        "expected_doc_ids": [], "must_include": [],
        "must_not_include": ["termination fee is $", "etf is $"],
        "should_answer": False},
    {   # Multi-doc: Essentials has no roaming, so needs Travel Pass pricing.
        "id": "c-essentials-abroad",
        "question": "I'm on Essentials and going to Mexico for 3 days. Will my data work there and what will it cost?",
        "expected_doc_ids": ["international-roaming"],
        "must_include": [["travel pass"], ["$10"]],
        "must_not_include": ["free data", "included at no cost"],
        "should_answer": True},
    {   # Conditional: a 6-hour outage doesn't qualify (needs >24h).
        "id": "c-short-outage",
        "question": "My service was down for 6 hours yesterday. How do I get my credit?",
        "expected_doc_ids": ["outage-credits"],
        "must_include": [["24"]],
        "must_not_include": ["you are eligible for a credit", "you're eligible for a credit",
                             "you'll receive a credit", "you will receive a credit"],
        "should_answer": True},
]

for _r in REGRESSION:
    _r["suite"] = "regression"
for _r in CAPABILITY:
    _r["suite"] = "capability"

SUITES = {"regression": REGRESSION, "capability": CAPABILITY}

# ---------------------------------------------------------------------------
# Retrieval: a small keyword retriever. Predictable on purpose, so retrieval
# failures are easy to spot and reason about in the traces.
# ---------------------------------------------------------------------------

_STOP = set("""a an and are as at be but by can do does for from get got have how i if in is
it its me my no not of on or so that the their there this to was what when where which will
with you your yesterday today just right really""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in _STOP]


def _overlap(query: set[str], doc: dict) -> int:
    blob = set(_tokens(doc["title"] + " " + doc["content"]))
    return len(query & blob)


@weave.op()
def search_help_center(query: str, k: int = 3) -> list[dict]:
    """Top-k help-center articles by keyword overlap."""
    q = set(_tokens(query))
    ranked = sorted(DOCUMENTS, key=lambda d: _overlap(q, d), reverse=True)
    return [dict(d) for d in ranked[:k] if _overlap(q, d) > 0]


# ---------------------------------------------------------------------------
# The prompt. Identical for all three models.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Acme Mobile's customer support assistant.

Rules:
- Answer ONLY from the help-center articles provided. Do not use outside knowledge.
- Include exact numbers, prices and time limits from the articles.
- If an article has a condition or exclusion that applies to the customer, say so.
- If the customer's assumption is wrong, correct it politely.
- If the articles don't answer the question, or it asks for legal, financial or
  other advice outside Acme support, say you can't help with that here and offer
  to connect them with a human agent. Don't guess.
- End with one line: "Sources: <article ids you used, comma-separated>"
  (or "Sources: none" if you used none)."""

USER_TEMPLATE = """Customer question:
{question}

Help-center articles:
{docs}"""


def format_docs(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in docs) \
        or "(no articles found)"


@weave.op()
def answer(provider: str, model: str, question: str, docs: list[dict]) -> LLMResult:
    """One LLM call. Weave also captures the raw provider call underneath."""
    return chat(provider, model, SYSTEM_PROMPT,
                USER_TEMPLATE.format(question=question, docs=format_docs(docs)))


class SupportBot(weave.Model):
    """The thing we evaluate. One instance per candidate model."""

    name_tag: str          # "grok" | "openai" | "anthropic"
    provider: str
    model_name: str
    system_prompt: str = SYSTEM_PROMPT
    k: int = 3

    @weave.op()
    def predict(self, question: str, **_unused) -> dict:
        # **_unused swallows the other dataset columns (expected_doc_ids, etc.)
        # that Weave routes to scorers but the bot shouldn't see.
        docs = search_help_center(question, k=self.k)
        result = answer(self.provider, self.model_name, question, docs)
        return {
            "answer": result.text,
            "retrieved_doc_ids": [d["id"] for d in docs],
            "model": result.response_model or self.model_name,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }


def make_bot(name: str) -> SupportBot:
    c = config.CANDIDATES[name]
    tag = f"{name}-mock" if config.uses_mock(c["provider"]) else name
    return SupportBot(name_tag=tag, provider=c["provider"], model_name=c["model"])


# ---------------------------------------------------------------------------
# Helpers shared with the scorers.
# ---------------------------------------------------------------------------

_SOURCES_RE = re.compile(r"sources?\s*:\s*(.*)", re.IGNORECASE)
_STRIP_RE = re.compile(r"^[\[\(<\"'`*\s]+|[\]\)>\"'`*\s.]+$")


def parse_sources(text: str) -> list[str] | None:
    """Article ids from the 'Sources:' line. None if there's no such line."""
    matches = _SOURCES_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    if raw.lower().startswith("none"):
        return []
    ids = [_STRIP_RE.sub("", s) for s in re.split(r"[,;]", raw)]
    return [i for i in ids if i]


def doc_text(doc_ids: list[str]) -> str:
    return format_docs([_DOC_BY_ID[i] for i in doc_ids if i in _DOC_BY_ID])
