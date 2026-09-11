"""RAG vs CAG vs MAG, measured."""

from .bench import Result, Sweep, run_strategy, sweep
from .corpus import Corpus, Document, Question, load_directory, load_squad
from .retrieval import Retriever, recall_at_k
from .strategies import CAG, MAG, RAG, Answer, Pricing, Strategy, default_strategies

__all__ = [
    "CAG",
    "MAG",
    "RAG",
    "Answer",
    "Corpus",
    "Document",
    "Pricing",
    "Question",
    "Result",
    "Retriever",
    "Strategy",
    "Sweep",
    "default_strategies",
    "load_directory",
    "load_squad",
    "recall_at_k",
    "run_strategy",
    "sweep",
]
