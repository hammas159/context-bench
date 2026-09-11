"""The web front end: change the prices, watch the crossover move.

    uv run uvicorn web.app:app --reload --port 8000

The point of putting a UI on this is not decoration. The crossover between RAG and CAG
depends on three numbers a reader has opinions about — the input price, the cache discount,
and the context window — and a static chart forces my assumptions on them. Here they are
inputs, and the answer recomputes.

The corpus is loaded once at startup and sliced per request, because rebuilding a retrieval
index on every keystroke is the kind of thing that makes a demo feel broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contextbench import (  # noqa: E402
    CAG,
    MAG,
    RAG,
    Corpus,
    Pricing,
    load_squad,
    run_strategy,
)
from contextbench.retrieval import Retriever, recall_at_k  # noqa: E402

HERE = Path(__file__).resolve().parent
MAX_DOCUMENTS = 1600
SWEEP_SIZES = [10, 25, 50, 100, 200, 400, 800, 1600]

app = FastAPI(title="context-bench", docs_url="/api")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

CORPUS: Corpus | None = None
_RETRIEVERS: dict[tuple[int, str], Retriever] = {}


def corpus() -> Corpus:
    global CORPUS
    if CORPUS is None:
        CORPUS = load_squad(max_documents=MAX_DOCUMENTS)
    return CORPUS


def retriever_for(size: int, mode: str) -> Retriever:
    """Cache one retriever per (size, mode). Rebuilding an index per request is the
    difference between a tool that feels instant and one that feels broken."""
    key = (size, mode)
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = Retriever(corpus().head(size), mode=mode)
    return _RETRIEVERS[key]


def build_strategies(*, k: int, memory_tokens: int) -> list:
    return [
        RAG(k=k, mode="hybrid"),
        CAG(use_cache=True),
        CAG(use_cache=False),
        MAG(k=max(1, k - 1), memory_tokens=memory_tokens),
    ]


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "summary": corpus().summary(),
            "sizes": SWEEP_SIZES,
            "defaults": Pricing(),
        },
    )


@app.get("/api/sweep")
async def api_sweep(
    input_price: float = 0.25,
    cached_price: float = 0.025,
    output_price: float = 1.25,
    window: int = 128_000,
    k: int = 4,
    memory_tokens: int = 2000,
    questions: int = 40,
) -> JSONResponse:
    """Re-run the whole sweep under the caller's prices."""
    pricing = Pricing(
        input_per_million=input_price,
        cached_input_per_million=cached_price,
        output_per_million=output_price,
        context_window=window,
    )

    rows = []
    for size in SWEEP_SIZES:
        slice_ = corpus().head(size)
        if not slice_.questions:
            continue
        for strategy in build_strategies(k=k, memory_tokens=memory_tokens):
            result, _ = run_strategy(strategy, slice_, pricing, limit=questions)
            rows.append(result.summary())

    # Where the order flips, computed here so the client does not have to agree with the
    # server about what "cheaper" means.
    crossover = None
    by_size: dict[int, dict[str, float]] = {}
    for row in rows:
        by_size.setdefault(row["documents"], {})[row["strategy"]] = row["cost_per_question"]

    rag_name = next((r["strategy"] for r in rows if r["strategy"].startswith("RAG")), None)
    cag_name = next((r["strategy"] for r in rows if "cached" in r["strategy"]), None)
    if rag_name and cag_name:
        for size in sorted(by_size):
            costs = by_size[size]
            if cag_name in costs and rag_name in costs and costs[cag_name] > costs[rag_name]:
                crossover = size
                break

    return JSONResponse({"rows": rows, "crossover": crossover, "pricing": pricing.__dict__})


@app.get("/api/ask")
async def api_ask(q: str = "", size: int = 200, k: int = 4) -> JSONResponse:
    """What each strategy would put in the context for this question.

    The part of the UI that makes the trade concrete: RAG shows four paragraphs and a
    token count, CAG shows the whole corpus and a much larger one, and the reader can see
    for themselves whether the answer was in there.
    """
    slice_ = corpus().head(size)
    if not q.strip():
        return JSONResponse({"error": "ask something"}, status_code=400)

    hits = retriever_for(size, "hybrid").top_k(q, k)
    total = slice_.total_tokens
    rag_context = "\n\n".join(d.text for d, _ in hits)

    # If this is one of the benchmark's own questions, the gold answer is known and the
    # honest thing to show is whether RAG actually got it - not just what it retrieved.
    # This is where the comparison stops being abstract: on a question RAG misses, CAG's
    # extra cost bought the right answer and RAG's saving bought a confident wrong one.
    gold = next((x for x in slice_.questions if x.text.strip().lower() == q.strip().lower()), None)
    verdict = None
    if gold is not None:
        verdict = {
            "answers": list(gold.answers),
            "gold_document": gold.document_id,
            "in_rag_context": gold.answered_by(rag_context),
            "in_cag_context": gold.answered_by("\n\n".join(d.text for d in slice_.documents)),
            "gold_retrieved": gold.document_id in {d.id for d, _ in hits},
        }

    return JSONResponse(
        {
            "question": q,
            "corpus": {"documents": len(slice_), "tokens": total},
            "rag": {
                "documents": [
                    {
                        "id": d.id,
                        "title": d.title,
                        "score": round(s, 4),
                        "tokens": d.tokens,
                        "text": d.text[:600],
                    }
                    for d, s in hits
                ],
                "tokens": sum(d.tokens for d, _ in hits),
            },
            "cag": {"documents": len(slice_), "tokens": total},
            "verdict": verdict,
        }
    )


@app.get("/api/recall")
async def api_recall(size: int = 400) -> JSONResponse:
    """Retrieval quality alone, before any strategy is layered on top."""
    slice_ = corpus().head(size)
    out = {}
    for mode in ("lexical", "semantic", "hybrid"):
        r = retriever_for(size, mode)
        out[mode] = {f"recall@{k}": round(recall_at_k(r, slice_, k), 4) for k in (1, 4, 10)}
    return JSONResponse(
        {"documents": len(slice_), "questions": len(slice_.questions), "modes": out}
    )


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "corpus": corpus().summary()})
