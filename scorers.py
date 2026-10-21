"""THE scorer list. Every evaluation in this repo uses exactly this list.

Rules we follow (and that make the results actionable):

1. Every scorer checks ONE requirement and returns ONE boolean, named
   `<failure_area>_ok` (e.g. `citation_ok`). Pass rates trend in the UI; you
   never have to interpret a 6.3/10. The distinct names keep the Compare
   view's radar chart readable (it labels each axis with the key name).
2. Every scorer says WHERE the failure lives (`failure_area`) and WHY
   (`why`). A red cell should tell you what to open next:

      failure_area   what to look at when it fails
      ------------   -------------------------------------------------------
      retrieval      the retriever / query: the right article never reached the model
      completeness   the model had the article but left out a required fact
      hallucination  the model stated a known-wrong claim (often a dropped exclusion)
      citation       the model ignored the output format or cited the wrong article
      scope          the model guessed when it should have declined, or refused a fair question
      grounding      the answer contains claims the articles don't support
      ux             the answer is too short to help or too long to read

3. Use the cheapest check that expresses the requirement: string and set
   checks first, an LLM judge only for fuzzy properties, and the judge only
   ever answers ONE yes/no question.

Because retrieval is identical for every model, `retrieval_hit` should be
identical across the three runs. If it isn't, something other than the model
changed, and the comparison isn't fair.
"""

from __future__ import annotations

import json
import re

import weave

import config
from llm import chat
from support_bot import doc_text, parse_sources

FIX_HINTS = {
    "retrieval": "Open the trace: which articles came back? Fix the query or retriever (same for every model).",
    "completeness": "The fact was in the articles. Read the answer: did the model drop it, or bury it?",
    "hallucination": "A known-wrong claim appeared, often a skipped condition. Check the system prompt's exclusion rule.",
    "citation": "Format drift. Compare what the prompt asks for with what this model emits.",
    "scope": "Model guessed instead of declining (or refused a fair question). Tighten the 'don't guess' rule.",
    "grounding": "Read `why` for the unsupported claims. Usually outside knowledge leaking in.",
    "ux": "Length is off. Usually a model verbosity difference; adjust the prompt or max tokens.",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


def _body(output: dict) -> str:
    """The answer without its Sources line, so citations don't satisfy fact checks."""
    return re.sub(r"sources?\s*:.*$", "", output["answer"], flags=re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# 1. Set check on structured output — no judge needed.
# ---------------------------------------------------------------------------

@weave.op()
def retrieval_hit(output: dict, expected_doc_ids: list[str]) -> dict:
    """Did the retriever return every article this question needs?"""
    missing = [d for d in expected_doc_ids if d not in output["retrieved_doc_ids"]]
    return {"retrieval_ok": not missing, "failure_area": "retrieval",
            "why": f"missing {missing}" if missing else ""}


# ---------------------------------------------------------------------------
# 2. Deterministic string checks — cheap, exact, easy to explain.
# ---------------------------------------------------------------------------

@weave.op()
def has_key_facts(output: dict, must_include: list[list[str]]) -> dict:
    """Does the answer state every required fact (any accepted phrasing)?"""
    body = _norm(_body(output))
    missing = [opts[0] for opts in must_include
               if not any(_norm(o) in body for o in opts)]
    return {"completeness_ok": not missing, "failure_area": "completeness",
            "why": f"missing {missing}" if missing else ""}


@weave.op()
def avoids_known_traps(output: dict, must_not_include: list[str]) -> dict:
    """Does the answer avoid the known-wrong claims for this question?"""
    body = _norm(_body(output))
    hits = [t for t in must_not_include if _norm(t) in body]
    return {"hallucination_ok": not hits, "failure_area": "hallucination",
            "why": f"said {hits}" if hits else ""}


# ---------------------------------------------------------------------------
# 3. Structural check — did the model follow the output format?
# ---------------------------------------------------------------------------

@weave.op()
def cites_sources(output: dict, expected_doc_ids: list[str]) -> dict:
    """Is there a Sources line, citing only retrieved articles, incl. an expected one?"""
    cited = parse_sources(output["answer"])
    if cited is None:
        return {"citation_ok": False, "failure_area": "citation", "why": "no 'Sources:' line"}
    invented = [c for c in cited if c not in output["retrieved_doc_ids"]]
    if invented:
        return {"citation_ok": False, "failure_area": "citation",
                "why": f"cited articles it was never given: {invented}"}
    if expected_doc_ids and not set(cited) & set(expected_doc_ids):
        return {"citation_ok": False, "failure_area": "citation",
                "why": f"cited {cited}, expected one of {expected_doc_ids}"}
    return {"citation_ok": True, "failure_area": "citation", "why": ""}


@weave.op()
def readable_length(output: dict) -> dict:
    """Between 15 and 180 words: long enough to help, short enough to read."""
    words = len(re.findall(r"\w+", _body(output)))
    ok = 15 <= words <= 180
    return {"ux_ok": ok, "failure_area": "ux",
            "why": f"{words} words" + ("" if ok else " (outside 15-180)")}


# ---------------------------------------------------------------------------
# 4. Binary LLM judge — only for properties a string check can't express.
#    The judge is a fixed model that is NOT one of the candidates (config.JUDGE).
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = ("You are a strict QA reviewer for a customer-support bot. Answer the one "
                 "question asked. Reply with JSON only: "
                 '{"verdict": "yes" or "no", "reason": "<one short sentence>"}')


def _ask_judge(question: str) -> tuple[bool, str]:
    raw = chat(config.JUDGE["provider"], config.JUDGE["model"], _JUDGE_SYSTEM, question).text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", raw)).strip().lower()
    return verdict.startswith("yes"), str(data.get("reason", raw))[:200]


@weave.op()
def handles_scope(output: dict, question: str, should_answer: bool) -> dict:
    """Answers fair questions; declines or escalates the ones it shouldn't answer."""
    if should_answer:
        q = (f"Customer asked: {question}\n\nBot replied: {_body(output)}\n\n"
             "Does the bot make a real attempt to answer, rather than refusing or only "
             "deflecting to a human?")
    else:
        q = (f"Customer asked: {question}\n\nBot replied: {_body(output)}\n\n"
             "This question is out of scope or not covered by the help center. Does the "
             "bot avoid inventing an answer and instead decline or offer a human agent?")
    ok, reason = _ask_judge(q)
    return {"scope_ok": ok, "failure_area": "scope", "why": "" if ok else reason}


@weave.op()
def grounded_in_articles(output: dict) -> dict:
    """Is every factual claim supported by the articles the bot was given?"""
    articles = doc_text(output["retrieved_doc_ids"]) or "(none)"
    q = (f"Help-center articles:\n{articles}\n\nBot answer:\n{_body(output)}\n\n"
         "Is every factual claim in the answer (prices, limits, time windows, policies) "
         "supported by the articles above? Saying it can't help, or offering a human "
         "agent, counts as supported. If no, name the unsupported claim in the reason.")
    ok, reason = _ask_judge(q)
    return {"grounding_ok": ok, "failure_area": "grounding", "why": "" if ok else reason}


# ---------------------------------------------------------------------------
# THE list. Import it; don't assemble your own.
# ---------------------------------------------------------------------------

SCORERS = [
    retrieval_hit,          # set check       -> retrieval
    has_key_facts,          # string check    -> completeness
    avoids_known_traps,     # string check    -> hallucination
    cites_sources,          # structural      -> citation
    readable_length,        # structural      -> ux
    handles_scope,          # binary judge    -> scope
    grounded_in_articles,   # binary judge    -> grounding
]

SCORER_AREA = {
    "retrieval_hit": "retrieval", "has_key_facts": "completeness",
    "avoids_known_traps": "hallucination", "cites_sources": "citation",
    "readable_length": "ux", "handles_scope": "scope",
    "grounded_in_articles": "grounding",
}

# The boolean each scorer returns, e.g. cites_sources -> "citation_ok".
PASS_KEY = {s: f"{area}_ok" for s, area in SCORER_AREA.items()}
