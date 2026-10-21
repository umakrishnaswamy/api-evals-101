"""00 — Pre-flight check. Run this before the workshop starts.

Checks that:
  - WANDB_ENTITY and the API keys are set (values are never printed)
  - each candidate model and the judge answer a one-line ping
It doesn't log anything to Weave.

If a model ID is rejected, set XAI_MODEL / OPENAI_MODEL / ANTHROPIC_MODEL to
one your key can use. For xAI and OpenAI, this script also lists available IDs.
"""

from __future__ import annotations

import os

import config
from llm import chat


def list_openai_compatible(base_url: str | None, key_env: str) -> list[str]:
    import openai

    client = openai.OpenAI(api_key=os.environ[key_env], base_url=base_url)
    return sorted(m.id for m in client.models.list())


def main() -> None:
    print("W&B host:     ", config.WANDB_BASE_URL or "WANDB_BASE_URL NOT SET")
    print("Weave project:", f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}"
          if config.WANDB_ENTITY else "WANDB_ENTITY NOT SET (your team name)")
    for env in ("WANDB_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        status = "set" if os.environ.get(env) else "MISSING"
        if status == "MISSING" and env == "WANDB_API_KEY" and os.path.exists(
                os.path.expanduser("~/.netrc")):
            status = "not set (using `wandb login` from ~/.netrc)"
        print(f"  {env:<18} {status}")
    if config.BACKEND == "mock":
        print("BACKEND=mock: skipping live pings.")
        return

    targets = [(n, c["provider"], c["model"]) for n, c in config.CANDIDATES.items()]
    targets.append(("judge", config.JUDGE["provider"], config.JUDGE["model"]))
    ok = True
    for label, provider, model in targets:
        if config.uses_mock(provider):
            print(f"  MOCK {label:<10} {provider:<10} {model:<28} -> {config.KEY_ENV[provider]} "
                  "not set; will use the scripted mock")
            continue
        try:
            r = chat(provider, model, "Reply with the single word: ready", "ping")
            print(f"  OK   {label:<10} {provider:<10} {model:<28} -> {r.text.strip()[:30]!r}")
        except Exception as e:  # noqa: BLE001 — show any failure plainly
            ok = False
            print(f"  FAIL {label:<10} {provider:<10} {model:<28} -> {type(e).__name__}: {str(e)[:120]}")
            if provider in ("xai", "openai") and os.environ.get(config.CANDIDATES.get(
                    "grok" if provider == "xai" else "openai", {}).get("key_env", "")):
                try:
                    ids = list_openai_compatible(
                        "https://api.x.ai/v1" if provider == "xai" else None,
                        "XAI_API_KEY" if provider == "xai" else "OPENAI_API_KEY")
                    print(f"       available: {', '.join(ids[:25])}")
                except Exception:  # noqa: BLE001
                    pass
    if config.WANDB_BASE_URL and config.WANDB_ENTITY:
        print(f"\nWeave UI: {config.ui_url()}")
    print("\nAll good." if ok else "\nFix the FAIL lines (usually a model ID) before running 01/02.")


if __name__ == "__main__":
    main()
