"""The corpus, the questions, and the ground truth that makes this a measurement.

A benchmark of retrieval strategies needs three things, and most comparisons only have the
first:

  documents   something to retrieve from
  questions   something to ask
  **answers** something to check against

Without the third, "RAG beat CAG" is a claim about vibes. SQuAD supplies all three: real
Wikipedia paragraphs, questions written by humans who could see the paragraph, and the exact
answer span. That means two things can be measured honestly rather than asserted:

  retrieval hit rate   did the paragraph containing the answer make it into the context?
  answer recall        is the answer string present in the context the model was handed?

Neither needs a language model, which is deliberate. **The parts of this comparison that
matter most — what reaches the context, what it costs, how long it takes — are decidable
without generating a single token.** Generation is a separate, pluggable step, and keeping
the line between them visible is the point.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


@dataclass(frozen=True)
class Document:
    """One retrievable unit. In SQuAD terms, one paragraph."""

    id: str
    title: str
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass(frozen=True)
class Question:
    """A question with its ground truth, and which document holds it."""

    id: str
    text: str
    answers: tuple[str, ...]
    document_id: str

    def answered_by(self, context: str) -> bool:
        """Is any accepted answer present in this context, verbatim?

        A deliberately weak test, and the right one for what it is measuring. It does not
        ask whether a model *would* answer correctly — it asks whether the information was
        **in the room**. A strategy that fails here cannot succeed downstream no matter how
        good the model is, and that is a cleaner question than one entangled with
        generation quality.
        """
        haystack = normalise(context)
        return any(normalise(a) in haystack for a in self.answers)


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. The SQuAD convention."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer.

    Roughly four characters per token for English prose, which is the widely used rule of
    thumb and is within a few percent of GPT-family tokenizers on this kind of text. It is
    an estimate and is named as one: a real tokenizer changes these numbers by a constant
    factor, not the conclusions, and demanding one here would mean a 200 MB dependency to
    make a cost curve slightly smoother.

    Ceiling division, so an **empty string costs nothing**. An earlier version floored at 1
    on the reasoning that a prompt always costs something - which is true of a prompt and
    false of a fragment. Contexts here are assembled from parts, and an absent part that
    bills for one token made MAG report a token of memory before it had remembered anything.
    """
    return (len(text) + 3) // 4


@dataclass
class Corpus:
    """Documents plus the questions asked of them."""

    documents: list[Document] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id = {d.id: d for d in self.documents}

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, document_id: str) -> Document | None:
        return self._by_id.get(document_id)

    @property
    def total_tokens(self) -> int:
        """The whole corpus, as a token count. The number that decides whether CAG works."""
        return sum(d.tokens for d in self.documents)

    def head(self, n_documents: int) -> Corpus:
        """The first n documents, and only the questions those documents can answer.

        Used to sweep corpus size. Keeping the questions consistent with the documents
        matters: if a question's source paragraph is not in the corpus, *no* strategy can
        answer it, and leaving it in makes every strategy look worse in lockstep while
        telling you nothing about which is better.
        """
        kept = self.documents[:n_documents]
        ids = {d.id for d in kept}
        return Corpus(
            documents=kept,
            questions=[q for q in self.questions if q.document_id in ids],
        )

    def summary(self) -> dict:
        sizes = [d.tokens for d in self.documents]
        return {
            "documents": len(self.documents),
            "questions": len(self.questions),
            "total_tokens": self.total_tokens,
            "median_document_tokens": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
            "largest_document_tokens": max(sizes) if sizes else 0,
        }


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target

    print(f"  fetching {url} ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "context-bench/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        payload = response.read()
    target.write_bytes(payload)
    print(
        f"  wrote {target.name} ({len(payload):,} bytes, "
        f"sha256 {hashlib.sha256(payload).hexdigest()[:16]})"
    )
    return target


def load_squad(*, max_documents: int | None = None, questions_per_document: int = 2) -> Corpus:
    """Real Wikipedia paragraphs with human-written questions and gold answers.

    `questions_per_document` is capped because SQuAD asks up to five questions of some
    paragraphs and one of others. Left uncapped, the popular paragraphs dominate the
    average and the benchmark quietly measures those few rather than the corpus.
    """
    path = _download(SQUAD_URL, CACHE / "squad-dev-v1.1.json")
    raw = json.loads(path.read_text(encoding="utf-8"))

    documents: list[Document] = []
    questions: list[Question] = []

    for article in raw["data"]:
        title = article["title"].replace("_", " ")
        for i, paragraph in enumerate(article["paragraphs"]):
            document_id = f"{article['title']}#{i}"
            documents.append(
                Document(id=document_id, title=title, text=paragraph["context"].strip())
            )

            for qa in paragraph["qas"][:questions_per_document]:
                answers = tuple(dict.fromkeys(a["text"] for a in qa["answers"]))
                if not answers:
                    continue
                questions.append(
                    Question(
                        id=qa["id"],
                        text=qa["question"].strip(),
                        answers=answers,
                        document_id=document_id,
                    )
                )

            if max_documents and len(documents) >= max_documents:
                ids = {d.id for d in documents}
                return Corpus(documents, [q for q in questions if q.document_id in ids])

    return Corpus(documents, questions)


def load_directory(path: str | Path, *, pattern: str = "**/*.md") -> Corpus:
    """Any folder of text files as a corpus, with no questions.

    Here so the tool points at a real company handbook or policy set, which is the
    situation the RAG-versus-CAG question actually arises in. Without ground-truth
    questions only cost and latency are measurable — and the interface says so by returning
    an empty question list rather than inventing any.
    """
    root = Path(path)
    documents = [
        Document(
            id=str(file.relative_to(root)),
            title=file.stem,
            text=file.read_text(encoding="utf-8", errors="replace"),
        )
        for file in sorted(root.glob(pattern))
        if file.is_file()
    ]
    return Corpus(documents, [])
