"""Retrieval, in pure numpy, because the comparison should run on any machine.

The default retrievers here need no model download and no GPU: TF-IDF cosine for the
semantic-ish half, BM25 for the lexical half, and Reciprocal Rank Fusion to combine them.
A sentence-transformer plugs into the same interface for anyone who wants it.

That is a real limitation and it is worth being exact about **which** conclusions it
touches. A better embedding model raises RAG's hit rate — it does not change:

  - what the whole corpus costs to preload (CAG's side of the trade)
  - how that cost scales with corpus size
  - where the crossover sits, except by moving it in RAG's favour

So the headline finding is, if anything, *conservative* with weak embeddings: better
retrieval makes RAG look better, and the crossover is already the interesting part.

On hybrid retrieval specifically, a bug worth remembering: a keyword index that ANDs every
term in a natural-language question matches nothing, so a "hybrid" system silently becomes
dense-only and nobody notices because the answers still look fine. `lexical_scores` here
ORs terms and the tests assert a multi-word question still matches.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .corpus import Corpus, Document

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of to in is was were are be been being and or but if then for on at by with "
    "from as that this these those it its he she they we you i not no do does did has have "
    "had will would can could should may might must about into over under between "
    # Question words, added after a live failure. "In what country is Normandy located?"
    # retrieved a paragraph about Warsaw, because `what`, `country` and `located` appear
    # everywhere and out-weighted the one term that actually identified the document.
    # Interrogatives carry no information about *which* document holds the answer - they
    # describe the shape of the answer, which is the generator's problem, not the index's.
    "what which when where who whom whose why how".split()  # noqa: SIM905 - a literal
    # list of 90 words is unreadable; the concatenated strings group them by purpose
)


def tokenise(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


@dataclass
class Index:
    """A term-document matrix and the statistics both scorers need."""

    vocabulary: dict[str, int]
    term_frequencies: np.ndarray  # documents x terms, raw counts
    document_frequencies: np.ndarray
    lengths: np.ndarray
    n_documents: int

    @property
    def average_length(self) -> float:
        return float(self.lengths.mean()) if len(self.lengths) else 0.0


def build_index(documents: list[Document]) -> Index:
    counters = [Counter(tokenise(d.text)) for d in documents]
    vocabulary: dict[str, int] = {}
    for counter in counters:
        for term in counter:
            vocabulary.setdefault(term, len(vocabulary))

    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)
    for row, counter in enumerate(counters):
        for term, count in counter.items():
            matrix[row, vocabulary[term]] = count

    return Index(
        vocabulary=vocabulary,
        term_frequencies=matrix,
        document_frequencies=(matrix > 0).sum(axis=0).astype(np.float32),
        lengths=matrix.sum(axis=1),
        n_documents=len(documents),
    )


def _query_vector(query: str, index: Index) -> tuple[np.ndarray, np.ndarray]:
    """Query term ids and counts, dropping terms the corpus has never seen."""
    counter = Counter(tokenise(query))
    ids, counts = [], []
    for term, count in counter.items():
        position = index.vocabulary.get(term)
        if position is not None:
            ids.append(position)
            counts.append(count)
    return np.array(ids, dtype=int), np.array(counts, dtype=np.float32)


def semantic_scores(query: str, index: Index) -> np.ndarray:
    """TF-IDF cosine similarity. The stand-in for a dense retriever.

    Not an embedding model, and it is labelled `semantic` only by the role it plays in the
    fusion. It captures term overlap weighted by rarity, which is most of what a small
    embedding model does on factual text and none of what a good one does on paraphrase.
    """
    ids, counts = _query_vector(query, index)
    if len(ids) == 0:
        return np.zeros(index.n_documents, dtype=np.float32)

    idf = np.log((index.n_documents + 1) / (index.document_frequencies + 1)) + 1.0
    weighted = index.term_frequencies * idf
    norms = np.linalg.norm(weighted, axis=1)
    norms[norms == 0] = 1.0

    query_weights = counts * idf[ids]
    query_norm = np.linalg.norm(query_weights) or 1.0

    return (weighted[:, ids] @ query_weights) / (norms * query_norm)


def lexical_scores(query: str, index: Index, *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25. Terms are **OR**-ed, deliberately.

    The failure this guards against: a keyword index that requires every query term to be
    present returns nothing for "In what country is Normandy located?", because no paragraph
    contains all of those words. A hybrid retriever built on top of it degrades silently to
    dense-only, and the answers still look plausible, so it goes unnoticed until somebody
    checks per-hit provenance.
    """
    ids, counts = _query_vector(query, index)
    if len(ids) == 0:
        return np.zeros(index.n_documents, dtype=np.float32)

    df = index.document_frequencies[ids]
    idf = np.log(1 + (index.n_documents - df + 0.5) / (df + 0.5))

    tf = index.term_frequencies[:, ids]
    norm = k1 * (1 - b + b * index.lengths[:, None] / max(index.average_length, 1e-9))
    return ((tf * (k1 + 1)) / (tf + norm) * idf).sum(axis=1)


def reciprocal_rank_fusion(*score_arrays: np.ndarray, k: int = 60) -> np.ndarray:
    """Combine rankings by rank, not by score.

    Scores from different retrievers live on incomparable scales — a cosine of 0.4 and a
    BM25 of 11.2 cannot be added, and normalising them makes the weighting depend on the
    spread of whichever query happened to be asked. RRF throws the magnitudes away and
    keeps only the order, which is the one thing both retrievers agree on the meaning of.
    """
    fused = np.zeros_like(score_arrays[0], dtype=np.float64)
    for scores in score_arrays:
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


class Retriever:
    """Ranks documents for a query. Three modes, one interface."""

    def __init__(self, corpus: Corpus, *, mode: str = "hybrid") -> None:
        if mode not in {"semantic", "lexical", "hybrid"}:
            raise ValueError(f"unknown retrieval mode: {mode}")
        self.corpus = corpus
        self.mode = mode
        self.index = build_index(corpus.documents)

    def scores(self, query: str) -> np.ndarray:
        if self.mode == "semantic":
            return semantic_scores(query, self.index)
        if self.mode == "lexical":
            return lexical_scores(query, self.index)
        return reciprocal_rank_fusion(
            semantic_scores(query, self.index), lexical_scores(query, self.index)
        )

    def top_k(self, query: str, k: int) -> list[tuple[Document, float]]:
        scores = self.scores(query)
        k = min(k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        best = best[np.argsort(-scores[best])]
        return [(self.corpus.documents[int(i)], float(scores[int(i)])) for i in best]


def recall_at_k(retriever: Retriever, corpus: Corpus, k: int) -> float:
    """Share of questions whose source document appears in the top k.

    The retrieval-only measure, kept separate from whether the answer text survived into
    the context. They differ, and the gap is informative: a chunking scheme can retrieve the
    right document and still cut the answer in half.
    """
    if not corpus.questions:
        return float("nan")

    hits = 0
    for question in corpus.questions:
        retrieved = {d.id for d, _ in retriever.top_k(question.text, k)}
        hits += question.document_id in retrieved
    return hits / len(corpus.questions)
