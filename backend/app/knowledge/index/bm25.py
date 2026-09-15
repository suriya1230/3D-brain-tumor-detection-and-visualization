"""BM25 lexical index over chunks (spec §5), via SQLite FTS5.

Chosen over a server (OpenSearch/Elasticsearch) because this is local-first
by design (session decision, 2026-09-12): one file, no service to run.
FTS5's bm25() ranking function is the same algorithm family a server-based
index would use — swapping backends later is a change behind
`search_bm25()`'s signature, not a rewrite of anything that calls it.

`title` is boosted relative to `text` (spec §5: "title boosted") via FTS5's
per-column bm25() weight argument.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.knowledge.schema import Chunk

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "bm25.sqlite"

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    text,
    source_type UNINDEXED,
    tumour_types UNINDEXED
);
"""

# bm25(table, *column_weights) — higher weight = more influence on rank.
# Column order must match the table definition above.
_TITLE_WEIGHT = 3.0
_TEXT_WEIGHT = 1.0


def build_bm25_index(chunks: list[Chunk], *, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # full rebuild; this index is always derived from chunks.jsonl

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, title, text, source_type, tumour_types) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (c.chunk_id, c.title, c.text, c.source_type, ",".join(c.tumour_types))
                for c in chunks
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _fts_escape(query: str) -> str:
    # Treat the query as a bag of words, not FTS5 query syntax — a user
    # question containing '"', '*' or ':' shouldn't become a syntax error.
    # OR, not the bare space-separated default (which FTS5 treats as AND):
    # a real bag-of-words search should surface the best partial match, not
    # return nothing just because one word in the question (a filler word
    # like "any", or one uncommon term) doesn't appear anywhere in the
    # corpus. bm25() still ranks documents matching more terms higher, so
    # this doesn't sacrifice precision — it only stops all-or-nothing misses.
    terms = [t.replace('"', '') for t in query.split() if t.strip()]
    return " OR ".join(f'"{t}"' for t in terms)


def search_bm25(
    query: str,
    *,
    top_k: int = 20,
    db_path: Path = DB_PATH,
    source_types: list[str] | None = None,
    tumour_type: str | None = None,
) -> list[tuple[str, float]]:
    """Returns [(chunk_id, bm25_score), ...], best (most negative) first.

    FTS5's bm25() returns *lower is better* (it's a cost, not a similarity);
    callers doing rank fusion should sort ascending, or use the rank
    position rather than the raw score.

    source_types/tumour_type are filters (spec §5: "filter, don't rank"),
    applied as a plain SQL WHERE alongside the MATCH — a clinical question
    can demand source_type=nci_pdq and get exactly that, not a re-ranking
    that merely favors it.
    """
    if not db_path.exists():
        return []

    where_extra = ""
    params: list = [_fts_escape(query)]
    if source_types:
        placeholders = ",".join("?" for _ in source_types)
        where_extra += f" AND source_type IN ({placeholders})"
        params.extend(source_types)
    if tumour_type:
        where_extra += " AND (',' || tumour_types || ',') LIKE ?"
        params.append(f"%,{tumour_type},%")
    params.append(top_k)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT chunk_id, bm25(chunks_fts, {_TITLE_WEIGHT}, {_TEXT_WEIGHT}) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ? {where_extra}
            ORDER BY score
            LIMIT ?
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    return [(chunk_id, score) for chunk_id, score in rows]
