"""02 — Evaluate the same support bot on three models. THE hands-on.

Same retriever, same prompt, same datasets, same scorers. The ONLY thing that
changes between runs is the model. That's what makes the comparison fair.

Runs two suites x three models = six evaluations:
  REGRESSION  (8 rows)  should sit near 100%; it's the gate
  CAPABILITY  (6 rows)  hard cases; should start lower; where models separate

Then prints, per suite, each scorer's pass rate per model with a 95% interval,
so you can tell a real gap from noise, plus where each model fails most.

In the Weave UI afterwards:
  1. Evaluations tab, sort by name (descending).
     Runs are named <timestamp>__<suite>__<model>.
  2. Select the three runs of ONE suite and click Compare.
     Don't mix suites: different datasets, so the comparison means nothing.
  3. Click any red cell to open the trace and read `why`.

Usage:
  python 02_eval_three_models.py                  # all three models
  python 02_eval_three_models.py grok openai      # a subset
  TRIALS=3 python 02_eval_three_models.py         # repeat each row to see noise
"""

from __future__ import annotations

import asyncio
import math
import sys

import weave

import config
from scorers import FIX_HINTS, PASS_KEY, SCORER_AREA, SCORERS
from support_bot import SUITES, make_bot


def make_dataset(suite: str) -> weave.Dataset:
    rows = [
        {
            "question": r["question"],
            "expected_doc_ids": r["expected_doc_ids"],
            "must_include": r["must_include"],
            "must_not_include": r["must_not_include"],
            "should_answer": r["should_answer"],
            "row_id": r["id"],
            "suite": r["suite"],
        }
        for r in SUITES[suite]
    ]
    return weave.Dataset(name=f"acme-support-{suite}", rows=rows)


async def run_eval(name: str, suite: str, dataset: weave.Dataset) -> dict:
    bot = make_bot(name)
    run_name = config.eval_display_name(f"{suite}__{bot.name_tag}")
    evaluation = weave.Evaluation(
        name=f"acme-support-{suite}-eval",  # must differ from the Dataset name
        evaluation_name=run_name,   # the display name the Evaluations tab sorts on
        dataset=dataset,
        scorers=SCORERS,
        trials=config.TRIALS,
    )
    print(f"\n=== {suite.upper()} · {bot.name_tag} ({bot.model_name}) ===")
    return await evaluation.evaluate(bot)


def pass_rate(result: dict, scorer: str) -> float | None:
    stats = result.get(scorer, {}).get(PASS_KEY[scorer])
    if isinstance(stats, dict) and "true_fraction" in stats:
        return stats["true_fraction"]
    return None


def ci95(p: float, n: int) -> float:
    """Half-width of a 95% interval for a pass rate over n graded rows."""
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def latency(result: dict) -> float | None:
    lat = result.get("model_latency", {})
    return lat.get("mean") if isinstance(lat, dict) else None


def print_report(suite: str, results: dict[str, dict], n: int) -> None:
    names = list(results)
    print("\n" + "=" * 78)
    print(f"{suite.upper()} — pass rate per scorer (± 95% interval, n={n} per model)")
    print("=" * 78)
    w = max(13, *(len(m) + 2 for m in names))  # "anthropic-mock" needs more than 13
    print(f"{'scorer':<24}{'area':<15}" + "".join(f"{m:>{w}}" for m in names))
    for s in SCORER_AREA:
        cells = []
        for m in names:
            p = pass_rate(results[m], s)
            cells.append("     n/a" if p is None else f"{p:>6.0%} ±{ci95(p, n):>3.0%}")
        print(f"{s:<24}{SCORER_AREA[s]:<15}" + "".join(f"{c:>{w}}" for c in cells))
    lat = [latency(results[m]) for m in names]
    print(f"{'latency (s, mean)':<39}" + "".join(
        f"{'n/a':>{w}}" if v is None else f"{v:>{w}.2f}" for v in lat))

    print("\nWhere each model fails most:")
    for m in names:
        rates = {s: pass_rate(results[m], s) for s in SCORER_AREA}
        rates = {s: r for s, r in rates.items() if r is not None}
        if not rates:
            continue
        worst = min(rates, key=rates.get)
        if rates[worst] >= 1.0:
            print(f"  {m:<15} passes everything in this suite")
        else:
            print(f"  {m:<15} {worst} ({rates[worst]:.0%}) -> {FIX_HINTS[SCORER_AREA[worst]]}")
    print("\nTwo models whose intervals overlap are not clearly different on this data.")


async def main(names: list[str]) -> None:
    unknown = [n for n in names if n not in config.CANDIDATES]
    if unknown:
        raise SystemExit(f"Unknown model name(s): {unknown}. Choose from {list(config.CANDIDATES)}")
    config.warn_about_mocks(names)
    weave.init(config.project())
    if config.BACKEND == "mock":
        print("BACKEND=mock: no model calls. Results check plumbing only, and the "
              "capability suite is skipped.")
    mocked = set(config.mocked_candidates(names))

    for suite in ["regression", "capability"]:
        # Scripted answers to hard questions measure nothing, so mocked models
        # only run the regression suite.
        run = names if suite == "regression" else [n for n in names if n not in mocked]
        if not run:
            continue
        dataset = make_dataset(suite)
        results = {}
        for name in run:
            label = f"{name}-mock" if name in mocked else name
            results[label] = await run_eval(name, suite, dataset)
        print_report(suite, results, n=len(SUITES[suite]) * config.TRIALS)

    print("\nNext: Weave UI -> Evaluations -> select the 3 runs of one suite -> Compare.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(config.CANDIDATES)))
