"""Hybrid retrieval: BM25 + dense, combined by reciprocal rank fusion
(spec §5).

BM25 and dense embeddings fail differently — BM25 catches exact terms
(gene names, drug names, PMIDs) that embeddings blur across; dense catches
paraphrase and synonymy that BM25 can't see at all. RRF combines the two
rankings without needing their raw scores to be on comparable scales (they
aren't: FTS5's bm25() is an unbounded cost, Chroma's distance is a bounded
similarity-ish number) — only rank position matters.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.knowledge.index.bm25 import search_bm25
from app.knowledge.index.dense import Embedder, search_dense
from app.knowledge.schema import Chunk

CHUNKS_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "chunks.jsonl"


def load_chunk_lookup(path: Path = CHUNKS_PATH) -> dict[str, Chunk]:
    lookup: dict[str, Chunk] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = Chunk.model_validate_json(line)
            lookup[c.chunk_id] = c
    return lookup


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(
    query: str,
    embedder: Embedder,
    chunk_lookup: dict[str, Chunk],
    *,
    top_k_each: int = 20,
    top_k_fused: int = 20,
    source_types: list[str] | None = None,
    tumour_type: str | None = None,
) -> list[Chunk]:
    """Spec §5: retrieve top 20 fused. Callers passing these to an LLM
    should slice to 6-8 (spec §6) — that trim happens at the call site, not
    here, so hand-inspection during eval can still see the full top 20.
    """
    bm25_hits = search_bm25(
        query, top_k=top_k_each, source_types=source_types, tumour_type=tumour_type
    )
    dense_hits = search_dense(
        query, embedder, top_k=top_k_each, source_types=source_types
    )

    bm25_ranking = [chunk_id for chunk_id, _ in bm25_hits]
    dense_ranking = [chunk_id for chunk_id, _ in dense_hits]

    # tumour_type isn't pushed into the dense query (Chroma has no clean
    # substring-in-string filter on our comma-joined metadata field), so it
    # is enforced here, post-fusion, using the real Chunk.tumour_types list.
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])

    results: list[Chunk] = []
    for chunk_id, _score in fused:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        if tumour_type and tumour_type not in chunk.tumour_types:
            continue
        results.append(chunk)
        if len(results) >= top_k_fused:
            break

    return results
