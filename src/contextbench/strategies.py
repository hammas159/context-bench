"""RAG, CAG and MAG as three answers to one question: what goes in the context?

    RAG   retrieve a few relevant chunks, per query
    CAG   put the entire corpus in, once, and rely on the cache
    MAG   carry what was learned in earlier turns

They are usually discussed as rivals. They are not really — RAG and CAG answer *where the
knowledge comes from*, MAG answers *what persists between turns*, and a real system often
wants two of them. What they genuinely do compete on is **cost**, and that is measurable.

The number that decides the argument, and that most comparisons omit:

    **prompt caching.**

Without it, CAG pays full input price for the whole corpus on every single query and is
absurd past a few thousand tokens. With it, the corpus is billed at roughly a tenth after
the first call, and "just put the handbook in the context" stops being a joke and becomes
an engineering choice with a crossover point.

Where that crossover sits is an empirical question. It is what this module exists to answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .corpus import Corpus, Document, Question, estimate_tokens
from .retrieval import Retriever


@dataclass(frozen=True)
class Pricing:
    """Per-million-token prices, and the cache discount that changes everything.

    Defaults are representative of a mid-tier 2026 model rather than any specific vendor;
    they are a parameter of the experiment, not a claim, and the UI exposes them precisely
    so a reader can put their own numbers in and watch the crossover move.
    """

    input_per_million: float = 0.25
    cached_input_per_million: float = 0.025  # the usual order: about a tenth
    output_per_million: float = 1.25
    context_window: int = 128_000

    def cost(
        self, *, fresh_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0
    ) -> float:
        return (
            fresh_tokens * self.input_per_million
            + cached_tokens * self.cached_input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass
class Answer:
    """What one strategy produced for one question, and what it cost to produce."""

    strategy: str
    question_id: str
    context: str
    fresh_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float
    sources: list[str] = field(default_factory=list)
    fits_in_window: bool = True

    @property
    def total_input_tokens(self) -> int:
        return self.fresh_tokens + self.cached_tokens

    def cost(self, pricing: Pricing) -> float:
        return pricing.cost(
            fresh_tokens=self.fresh_tokens,
            cached_tokens=self.cached_tokens,
            output_tokens=self.output_tokens,
        )


class Strategy:
    """One way of assembling a context. Subclasses implement `build`."""

    name = "strategy"

    def prepare(self, corpus: Corpus) -> None:
        """Anything done once, before any question. Indexing, or warming a cache."""

    def build(self, question: Question, corpus: Corpus, pricing: Pricing) -> Answer:
        raise NotImplementedError

    def reset(self) -> None:
        """Forget anything carried between questions. Only MAG has anything to forget."""


class RAG(Strategy):
    """Retrieve the top k documents, per query.

    Pays a small, constant input cost regardless of corpus size — and pays it in accuracy
    whenever the retriever misses. The trade this whole benchmark is about.
    """

    def __init__(self, *, k: int = 4, mode: str = "hybrid") -> None:
        self.k = k
        self.mode = mode
        self.name = f"RAG (top {k}, {mode})"
        self._retriever: Retriever | None = None

    def prepare(self, corpus: Corpus) -> None:
        self._retriever = Retriever(corpus, mode=self.mode)

    def build(  # noqa: ARG002 - `corpus` is part of the Strategy interface; only CAG reads it
        self, question: Question, corpus: Corpus, pricing: Pricing
    ) -> Answer:
        assert self._retriever is not None, "call prepare() first"
        started = time.perf_counter()

        hits = self._retriever.top_k(question.text, self.k)
        context = "\n\n".join(f"[{d.id}] {d.text}" for d, _ in hits)
        elapsed = (time.perf_counter() - started) * 1000

        return Answer(
            strategy=self.name,
            question_id=question.id,
            context=context,
            # Retrieved text differs every query, so none of it can be cached.
            fresh_tokens=estimate_tokens(context) + estimate_tokens(question.text),
            cached_tokens=0,
            output_tokens=48,
            latency_ms=elapsed,
            sources=[d.id for d, _ in hits],
            fits_in_window=estimate_tokens(context) < pricing.context_window,
        )


class CAG(Strategy):
    """Put the whole corpus in the context, once, and let the cache carry it.

    No retriever, so no retrieval error: if the answer is anywhere in the corpus it is in
    the context, by construction. The cost is that you are billed for the corpus on every
    query — at cache rates after the first — and that the corpus must fit in the window.

    When it does not fit, this strategy **says so** rather than silently truncating. A
    quiet truncation is the worst of both worlds: CAG's price with RAG's recall, and no
    signal that anything was dropped.
    """

    def __init__(self, *, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self.name = "CAG (whole corpus" + (", cached)" if use_cache else ", no cache)")
        self._context = ""
        self._tokens = 0
        self._warm = False

    def prepare(self, corpus: Corpus) -> None:
        self._context = "\n\n".join(f"[{d.id}] {d.text}" for d in corpus.documents)
        self._tokens = estimate_tokens(self._context)
        self._warm = False

    def reset(self) -> None:
        self._warm = False

    def build(self, question: Question, corpus: Corpus, pricing: Pricing) -> Answer:
        started = time.perf_counter()
        fits = self._tokens < pricing.context_window

        if self.use_cache:
            # First call writes the cache at full price; every later call reads it.
            fresh = 0 if self._warm else self._tokens
            cached = self._tokens if self._warm else 0
            self._warm = True
        else:
            fresh, cached = self._tokens, 0

        elapsed = (time.perf_counter() - started) * 1000
        return Answer(
            strategy=self.name,
            question_id=question.id,
            context=self._context,
            fresh_tokens=fresh + estimate_tokens(question.text),
            cached_tokens=cached,
            output_tokens=48,
            latency_ms=elapsed,
            sources=[d.id for d in corpus.documents],
            fits_in_window=fits,
        )


class MAG(Strategy):
    """Retrieve, and also carry forward what earlier turns established.

    Memory here is what previous questions pulled in: a running set of documents already
    seen this session, capped by a token budget and evicted least-recently-used.

    Two honest properties fall out, and both are visible in the results:

      it helps    when consecutive questions are about the same material, which is what
                  real conversations look like
      it hurts    when they are not, because the memory is paid for on every call and
                  crowds out the budget for fresh retrieval
    """

    def __init__(self, *, k: int = 3, memory_tokens: int = 2000, mode: str = "hybrid") -> None:
        self.k = k
        self.memory_tokens = memory_tokens
        self.mode = mode
        self.name = f"MAG (top {k} + {memory_tokens}t memory)"
        self._retriever: Retriever | None = None
        self._memory: list[Document] = []

    def prepare(self, corpus: Corpus) -> None:
        self._retriever = Retriever(corpus, mode=self.mode)
        self._memory = []

    def reset(self) -> None:
        self._memory = []

    def _remember(self, documents: list[Document]) -> None:
        for document in documents:
            self._memory = [d for d in self._memory if d.id != document.id]
            self._memory.append(document)

        # Evict from the front (least recently used) until the budget is met.
        while sum(d.tokens for d in self._memory) > self.memory_tokens and self._memory:
            self._memory.pop(0)

    def build(  # noqa: ARG002 - `corpus` is part of the Strategy interface; only CAG reads it
        self, question: Question, corpus: Corpus, pricing: Pricing
    ) -> Answer:
        assert self._retriever is not None, "call prepare() first"
        started = time.perf_counter()

        hits = self._retriever.top_k(question.text, self.k)
        fresh_documents = [d for d, _ in hits]
        remembered = [d for d in self._memory if d.id not in {f.id for f in fresh_documents}]

        memory_text = "\n\n".join(f"[{d.id}] {d.text}" for d in remembered)
        fresh_text = "\n\n".join(f"[{d.id}] {d.text}" for d in fresh_documents)
        context = (memory_text + "\n\n" + fresh_text).strip()

        self._remember(fresh_documents)
        elapsed = (time.perf_counter() - started) * 1000

        return Answer(
            strategy=self.name,
            question_id=question.id,
            context=context,
            fresh_tokens=estimate_tokens(fresh_text) + estimate_tokens(question.text),
            # Memory is stable across turns, so it is exactly what a prompt cache is for.
            cached_tokens=estimate_tokens(memory_text),
            output_tokens=48,
            latency_ms=elapsed,
            sources=[d.id for d in remembered + fresh_documents],
            fits_in_window=estimate_tokens(context) < pricing.context_window,
        )


def default_strategies() -> list[Strategy]:
    """The line-up. CAG appears twice on purpose — the cache is the whole argument."""
    return [
        RAG(k=4, mode="hybrid"),
        RAG(k=1, mode="hybrid"),
        CAG(use_cache=True),
        CAG(use_cache=False),
        MAG(k=3, memory_tokens=2000),
    ]
