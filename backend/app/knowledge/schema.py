"""Phase 3 data contracts. §2 (Chunk) and §6 (Answer) of the dev spec.

Changing Chunk's fields means reindexing everything downstream, so treat it
as a migration, not an edit.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["nci_pdq", "pubmed", "pmc_oa", "who_ref", "web"]


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    source_type: SourceType
    title: str
    section: str | None = None
    text: str
    url: str
    published: date | None = None
    retrieved: date
    licence: str
    tumour_types: list[str] = Field(default_factory=list)
    evidence_level: str | None = None
    token_count: int


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    source_type: SourceType
    title: str
    section: str | None = None
    url: str


class Claim(BaseModel):
    text: str
    chunk_ids: list[str]
    # Set by app/knowledge/validator.py (spec §10's Evidence Validator),
    # not by answer_question() itself — a claim can have a real citation
    # and still be flagged if that citation doesn't actually contain the
    # claim's specific numbers/entities. Flagged claims are never dropped
    # silently; they're surfaced for human review (spec: "a validator that
    # silently approves is worse than none").
    flagged: bool = False
    flag_reason: str | None = None

    def is_supported(self) -> bool:
        return len(self.chunk_ids) > 0


class Answer(BaseModel):
    claims: list[Claim]
    unsupported: list[str] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
