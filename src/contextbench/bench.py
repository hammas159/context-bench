"""The harness: run every strategy over every question, and sweep corpus size.

One rule governs the design. **Every strategy sees exactly the same questions in exactly
the same order.** Comparisons where one method got an easier slice are the most common way
a benchmark lies, and ordering matters here specifically because MAG's whole mechanism is
memory of what came before.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .corpus import Corpus
from .strategies import Answer, Pricing, Strategy


@dataclass
class Result:
    """How one strategy did on one corpus."""

    strategy: str
    documents: int
    corpus_tokens: int
    questions: int
    answer_in_context: float  # share of questions whose gold answer reached the context
    mean_input_tokens: float
    total_cost: float
    cost_per_question: float
    mean_latency_ms: float
    fits_in_window: bool

    @property
    def cost_per_1000_questions(self) -> float:
        return self.cost_per_question * 1000

    def summary(self) -> dict:
        out = asdict(self)
        out["cost_per_1000_questions"] = round(self.cost_per_1000_questions, 4)
        for key in (
            "answer_in_context",
            "mean_input_tokens",
            "total_cost",
            "cost_per_question",
            "mean_latency_ms",
        ):
            out[key] = round(out[key], 6)
        return out


def run_strategy(
    strategy: Strategy, corpus: Corpus, pricing: Pricing, *, limit: int | None = None
) -> tuple[Result, list[Answer]]:
    """Run one strategy over a corpus and score it."""
    strategy.prepare(corpus)
    strategy.reset()

    questions = corpus.questions[:limit] if limit else corpus.questions
    answers: list[Answer] = []
    found = 0

    for question in questions:
        answer = strategy.build(question, corpus, pricing)
        answers.append(answer)
        found += question.answered_by(answer.context)

    n = max(len(answers), 1)
    total_cost = sum(a.cost(pricing) for a in answers)

    return (
        Result(
            strategy=strategy.name,
            documents=len(corpus),
            corpus_tokens=corpus.total_tokens,
            questions=len(answers),
            answer_in_context=found / n,
            mean_input_tokens=sum(a.total_input_tokens for a in answers) / n,
            total_cost=total_cost,
            cost_per_question=total_cost / n,
            mean_latency_ms=sum(a.latency_ms for a in answers) / n,
            fits_in_window=all(a.fits_in_window for a in answers),
        ),
        answers,
    )


@dataclass
class Sweep:
    """The same comparison at several corpus sizes. Where the crossover lives."""

    sizes: list[int]
    pricing: Pricing
    results: list[Result] = field(default_factory=list)

    def by_strategy(self) -> dict[str, list[Result]]:
        out: dict[str, list[Result]] = {}
        for result in self.results:
            out.setdefault(result.strategy, []).append(result)
        for rows in out.values():
            rows.sort(key=lambda r: r.documents)
        return out

    def crossover(self, cheap: str, dear: str) -> int | None:
        """The corpus size at which `cheap` stops being cheaper than `dear`.

        Returned as a document count, or None if the order never changes over the sizes
        tested. Reporting None rather than extrapolating is deliberate: the crossover is
        only knowable inside the range actually measured.
        """
        grouped = self.by_strategy()
        if cheap not in grouped or dear not in grouped:
            return None

        for a, b in zip(grouped[cheap], grouped[dear], strict=False):
            if a.cost_per_question > b.cost_per_question:
                return a.documents
        return None

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pricing": asdict(self.pricing),
                    "sizes": self.sizes,
                    "results": [r.summary() for r in self.results],
                },
                indent=2,
            )
        )
        return path


def sweep(
    strategies: list[Strategy],
    corpus: Corpus,
    *,
    sizes: list[int],
    pricing: Pricing | None = None,
    questions_per_size: int = 60,
    progress: bool = True,
) -> Sweep:
    """Run every strategy at every corpus size.

    `questions_per_size` is capped so the cost figures are comparable across sizes: a larger
    corpus brings more questions with it, and a total cost that grows partly because there
    were more questions would confuse the very thing being measured.
    """
    pricing = pricing or Pricing()
    out = Sweep(sizes=sizes, pricing=pricing)

    for size in sizes:
        slice_ = corpus.head(size)
        if not slice_.questions:
            continue
        if progress:
            print(
                f"  {size:>5} documents  ({slice_.total_tokens:>8,} tokens, "
                f"{len(slice_.questions):>4} questions)",
                flush=True,
            )

        for strategy in strategies:
            result, _ = run_strategy(strategy, slice_, pricing, limit=questions_per_size)
            out.results.append(result)

    return out
