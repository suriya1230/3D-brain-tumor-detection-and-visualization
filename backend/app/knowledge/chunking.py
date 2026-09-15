"""Section-aware chunking (spec §4).

Ingesters produce one RawSection per PDQ heading / BioC section, which can
be anywhere from a sentence to twenty thousand characters. This module is
what actually produces spec §2's Chunk records: split on section boundaries
first (already done, by construction, upstream), then only split further
inside a section, on paragraph boundaries, once it's over ~900 tokens, with
15% overlap so a fact near a chunk boundary isn't invisible to retrieval
from either side. Never merges across documents — the input unit is already
one section of one document, and chunk_section() never looks past it.
"""

from __future__ import annotations

import hashlib
import re

import tiktoken

from app.knowledge.ingest.common import RawSection
from app.knowledge.normalization.tumor_types import find_tumour_types_in_text
from app.knowledge.schema import Chunk

MAX_TOKENS = 900
OVERLAP_RATIO = 0.15

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_oversized_paragraphs(paragraphs: list[str], max_tokens: int) -> list[str]:
    """Ensures every unit going into the greedy grouper is <= max_tokens, by
    falling back to sentence-boundary splitting for any paragraph that
    alone exceeds the budget (rare, but real for dense review-article text).
    """
    out: list[str] = []
    for p in paragraphs:
        if count_tokens(p) <= max_tokens:
            out.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf: list[str] = []
        buf_tokens = 0
        for s in sentences:
            st = count_tokens(s)
            if buf and buf_tokens + st > max_tokens:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(s)
            buf_tokens += st
        if buf:
            out.append(" ".join(buf))
    return out


def _greedy_group_with_overlap(units: list[str], max_tokens: int, overlap_ratio: float) -> list[str]:
    overlap_budget = int(max_tokens * overlap_ratio)
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0

    i = 0
    while i < len(units):
        unit = units[i]
        ut = count_tokens(unit)

        if current and current_tokens + ut > max_tokens:
            groups.append("\n\n".join(current))

            overlap_units: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                prev_t = count_tokens(prev)
                if overlap_units and overlap_tokens + prev_t > overlap_budget:
                    break
                overlap_units.insert(0, prev)
                overlap_tokens += prev_t

            # Guarantee termination: if the overlap-seeded group still can't
            # fit `unit` (a large tail paragraph plus a large next unit),
            # rebuilding overlap from it next time would select the same
            # paragraphs again and flush forever without `i` ever advancing.
            # Dropping the overlap for this one transition forces progress —
            # the next check sees an empty `current` and always accepts.
            if overlap_tokens + ut > max_tokens:
                current, current_tokens = [], 0
            else:
                current, current_tokens = overlap_units, overlap_tokens
            continue  # retry `unit` against the (possibly now empty) group

        current.append(unit)
        current_tokens += ut
        i += 1

    if current:
        groups.append("\n\n".join(current))
    return groups


def chunk_section(section: RawSection, *, max_tokens: int = MAX_TOKENS, overlap_ratio: float = OVERLAP_RATIO) -> list[Chunk]:
    if count_tokens(section.text) <= max_tokens:
        spans = [section.text]
    else:
        paragraphs = [p for p in section.text.split("\n\n") if p.strip()]
        units = _split_oversized_paragraphs(paragraphs, max_tokens)
        spans = _greedy_group_with_overlap(units, max_tokens, overlap_ratio)

    chunks: list[Chunk] = []
    for i, span_text in enumerate(spans):
        chunk_id = hashlib.sha1(f"{section.source_id}::{i}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_id=section.source_id,
                source_type=section.source_type,
                title=section.title,
                section=section.section,
                text=span_text,
                url=section.url,
                published=section.published,
                retrieved=section.retrieved,
                licence=section.licence,
                tumour_types=find_tumour_types_in_text(span_text),
                evidence_level=section.evidence_level,
                token_count=count_tokens(span_text),
            )
        )
    return chunks


def chunk_all(sections: list[RawSection]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s in sections:
        chunks.extend(chunk_section(s))
    return chunks
