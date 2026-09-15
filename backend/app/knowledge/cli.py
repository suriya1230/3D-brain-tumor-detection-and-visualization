"""Command-line entrypoints for the Phase 3 knowledge pipeline.

    python -m app.knowledge.cli build-corpus   # ingest -> chunk -> chunks.jsonl
    python -m app.knowledge.cli build-index     # chunks.jsonl -> BM25 + dense indexes
    python -m app.knowledge.cli eval            # run the 40-question eval set

Each step reads the previous step's output from disk rather than holding
everything in memory across steps — so any one stage can be rerun on its
own once its input file exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.knowledge.chunking import chunk_all
from app.knowledge.ingest.nci_pdq import load_all_pdq_fixtures
from app.knowledge.ingest.pubmed import ingest_pubmed

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# Same fixture-stage PMID list used by `python -m app.knowledge.ingest.pubmed`
# directly (spec §9 step 2) — kept in one place so build-corpus and that
# module's __main__ block don't drift apart.
PUBMED_FIXTURE_PMIDS = [
    "41057317", "40255400", "39911397", "39387957", "38396785",
    "37857607", "33396284", "33261657", "29938005",
    "38724526", "40939590", "32478924", "34504304",
    "28281173", "35135556",
    # Added 2026-09-12 to close two eval-observed gaps: post-treatment
    # imaging interpretation (pseudoprogression) and recurrence-management
    # specifics beyond PDQ's general re-resection statements.
    "26275367", "23151413", "41134378", "25438810", "40528226",
    "39587462", "39759563", "21655151", "32408220", "40991008",
]


def build_corpus() -> None:
    sections = list(load_all_pdq_fixtures())

    _, pubmed_sections = ingest_pubmed(PUBMED_FIXTURE_PMIDS)
    sections.extend(pubmed_sections)

    chunks = chunk_all(sections)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    by_source = {}
    for c in chunks:
        by_source[c.source_type] = by_source.get(c.source_type, 0) + 1

    print(f"{len(sections)} raw sections -> {len(chunks)} chunks -> {CHUNKS_PATH}")
    for source_type, n in sorted(by_source.items()):
        print(f"  {source_type}: {n} chunks")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "build-corpus":
        build_corpus()
    else:
        print(__doc__)
        sys.exit(1)
