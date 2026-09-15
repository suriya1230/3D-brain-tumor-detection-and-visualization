"""Shared pre-chunk document model.

Ingesters (nci_pdq.py, pubmed.py) each produce RawSection objects: one
coherent section of source content, not yet split to the ~900-token chunk
size. chunking.py (spec §4) turns these into final Chunk records. Keeping
the two steps separate means an ingester never has to know about chunk
sizing, and the chunker never has to know about a source's fetch mechanics.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.knowledge.schema import SourceType


class RawSection(BaseModel):
    source_id: str
    source_type: SourceType
    title: str
    section: str | None = None
    text: str
    url: str
    published: date | None = None
    retrieved: date
    licence: str
    evidence_level: str | None = None
