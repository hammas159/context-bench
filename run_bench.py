"""Run the benchmark and print where the crossover is.

uv run python run_bench.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from contextbench import Pricing, default_strategies, load_squad, sweep  # noqa: E402
from contextbench.retrieval import Retriever, recall_at_k  # noqa: E402

SIZES = [10, 25, 50, 100, 200, 400, 800, 1600]
QUESTIONS = 60


def main() -> None:
    pricing = Pricing()
    corpus = load_squad(max_documents=max(SIZES))
    print(f"\ncorpus: {corpus.summary()}")
    print(
        f"pricing: input ${pricing.input_per_million}/M, "
        f"cached ${pricing.cached_input_per_million}/M, "
        f"output ${pricing.output_per_million}/M, window {pricing.context_window:,}\n"
    )

    # Retrieval quality on its own, before any strategy is involved.
    sample = corpus.head(400)
    print("  retrieval recall@k on 400 documents:")
    for mode in ("lexical", "semantic", "hybrid"):
        retriever = Retriever(sample, mode=mode)
        scores = "  ".join(f"@{k}: {recall_at_k(retriever, sample, k):.1%}" for k in (1, 4, 10))
        print(f"    {mode:<9} {scores}")

    print(
        f"\nsweeping {len(SIZES)} corpus sizes x {len(default_strategies())} strategies, "
        f"{QUESTIONS} questions each\n"
    )
    results = sweep(
        default_strategies(), corpus, sizes=SIZES, pricing=pricing, questions_per_size=QUESTIONS
    )

    print(
        f"\n{'strategy':<28}{'docs':>6}{'answer in ctx':>15}{'input tok':>12}"
        f"{'$/1k questions':>16}{'fits':>7}"
    )
    for result in results.results:
        s = result.summary()
        print(
            f"  {s['strategy']:<26}{s['documents']:>6}{s['answer_in_context']:>14.1%}"
            f"{s['mean_input_tokens']:>12,.0f}{s['cost_per_1000_questions']:>16.3f}"
            f"{'yes' if s['fits_in_window'] else 'NO':>7}"
        )

    # --- the finding -------------------------------------------------------
    grouped = results.by_strategy()
    rag = next(k for k in grouped if k.startswith("RAG (top 4"))
    cag = next(k for k in grouped if "cached" in k)
    cag_raw = next(k for k in grouped if "no cache" in k)

    print("\n" + "=" * 78)
    crossover = results.crossover(cag, rag)
    if crossover:
        print(f"  CAG-with-cache is cheaper than RAG up to {crossover} documents, then RAG wins.")
    else:
        print(f"  Over {SIZES[0]}-{SIZES[-1]} documents, the cost order never flips.")

    raw_crossover = results.crossover(cag_raw, rag)
    print(f"  Without prompt caching, CAG loses from {raw_crossover or SIZES[0]} documents on.")

    small = [r for r in results.results if r.documents == SIZES[0]]
    large = [r for r in results.results if r.documents == SIZES[-1]]
    for label, rows in (("smallest", small), ("largest", large)):
        if not rows:
            continue
        best_recall = max(rows, key=lambda r: r.answer_in_context)
        cheapest = min(rows, key=lambda r: r.cost_per_question)
        print(
            f"  {label:>8} corpus: best recall {best_recall.answer_in_context:.0%} "
            f"({best_recall.strategy}); cheapest {cheapest.strategy}"
        )

    out = results.to_json(Path(__file__).parent / "data" / "results.json")
    print(f"\n  wrote {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
