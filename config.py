"""Everything you might need to change lives here: project, models, judge.

All values come from environment variables (see .env.example), so nothing
secret is ever written into the repo.
"""

from __future__ import annotations

import datetime as _dt
import os

# ---------------------------------------------------------------------------
# Where runs land in Weave:  <WANDB_BASE_URL>/<WANDB_ENTITY>/<WANDB_PROJECT>/weave
# WANDB_BASE_URL is your W&B host (a Dedicated/Self-Managed URL, or
# https://api.wandb.ai for the public cloud); WANDB_ENTITY is your team.
# ---------------------------------------------------------------------------

WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "").rstrip("/")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "api-evals-101")


def _check(name: str, value: str | None, example: str) -> None:
    if not value or "<" in value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"    export {name}={example}\n"
            "    (or copy .env.example to .env, fill it in, and `source .env`)"
        )


def project() -> str:
    _check("WANDB_BASE_URL", WANDB_BASE_URL, "https://<your-instance>.wandb.io")
    _check("WANDB_ENTITY", WANDB_ENTITY, "<your-team>")
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}"


def ui_url() -> str:
    """The Weave UI for this project (the public cloud UI lives at wandb.ai)."""
    host = WANDB_BASE_URL.replace("://api.wandb.ai", "://wandb.ai")
    return f"{host}/{WANDB_ENTITY}/{WANDB_PROJECT}/weave"


# ---------------------------------------------------------------------------
# The three candidate models. Same bot, same prompt, same retriever:
# the model is the ONLY thing that changes between eval runs.
#
# Model IDs change often. Run `python 00_check_setup.py` to see what your
# keys can reach, then override with XAI_MODEL / OPENAI_MODEL / ANTHROPIC_MODEL.
# ---------------------------------------------------------------------------

CANDIDATES: dict[str, dict] = {
    "grok": {
        "provider": "xai",
        "model": os.environ.get("XAI_MODEL", "grok-4.3"),
        "key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",  # OpenAI-compatible endpoint
    },
    "openai": {
        "provider": "openai",
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
        "key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "anthropic": {
        "provider": "anthropic",
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
}

# ---------------------------------------------------------------------------
# The judge. ONE fixed model grades every candidate, and it is deliberately
# NOT one of the candidates: LLM judges tend to prefer their own outputs
# (self-preference bias), so a candidate grading itself skews the comparison.
# ---------------------------------------------------------------------------

JUDGE = {
    "provider": os.environ.get("JUDGE_PROVIDER", "anthropic"),
    "model": os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
}

# Optional: set TEMPERATURE to force a value. Left unset by default because
# some reasoning models reject any temperature other than their default.
TEMPERATURE = os.environ.get("TEMPERATURE")

# How many times to run each row. >1 shows you run-to-run noise.
TRIALS = int(os.environ.get("TRIALS", "1"))

# BACKEND=mock runs the whole pipeline with no API calls (plumbing check only;
# see mock_llm.py). Otherwise each provider is real if its key is set, and
# falls back to the mock if it isn't, so a missing key never blocks the session.
BACKEND = os.environ.get("BACKEND", "real").lower()

KEY_ENV = {"xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY",
           "anthropic": "ANTHROPIC_API_KEY"}


def uses_mock(provider: str) -> bool:
    """True if calls to this provider go to the mock (BACKEND=mock or no key)."""
    return BACKEND == "mock" or not os.environ.get(KEY_ENV[provider])


def mocked_candidates(names: list[str] | None = None) -> list[str]:
    return [n for n in (names or list(CANDIDATES))
            if uses_mock(CANDIDATES[n]["provider"])]


def warn_about_mocks(names: list[str], judge: bool = True) -> None:
    """Say loudly which parts of this run are scripted, not real models."""
    if BACKEND == "mock":
        return
    for n in mocked_candidates(names):
        print(f"NOTE: {KEY_ENV[CANDIDATES[n]['provider']]} not set -> '{n}' runs on the "
              "scripted mock (labelled '-mock'). Its results say nothing about the model.")
    if judge and uses_mock(JUDGE["provider"]):
        print(f"NOTE: {KEY_ENV[JUDGE['provider']]} not set -> the LLM judge is mocked and "
              "always says yes, so grounded_in_articles is not a real check this run.")


def eval_display_name(scope: str) -> str:
    """<YYYYMMDDTHHMM>__<scope> so the Evaluations tab sorts cleanly."""
    return f"{_dt.datetime.now().strftime('%Y%m%dT%H%M')}__{scope}"
