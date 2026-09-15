"""Evidence Validator (spec §10) — first version.

Spec is explicit that claim-level verification against retrieved text is
an open research problem, and that a working first version is narrower:
"check that every claim's cited chunks contain the claim's key entities
and numbers, and flag the rest for human review. That is achievable and
honest. A validator that silently approves is worse than none, because it
manufactures confidence."

So this does exactly that — nothing fancier, and it never removes a claim.
It only sets Claim.flagged/flag_reason. Numbers get their own check because
they're the sneaky failure mode: an LLM restating "42.3 months" as "45
months" still overlaps heavily on ordinary vocabulary, so a pure word-
overlap check would miss it. Word-overlap catches the coarser failure —
a claim that shares almost no vocabulary with what it's supposedly citing.
"""

from __future__ import annotations

import re

from app.knowledge.schema import Answer, Chunk

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

ENTITY_OVERLAP_THRESHOLD = 0.5


def _numbers_in(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _entities_in(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def validate_claim(claim_text: str, chunk_ids: list[str], chunk_lookup: dict[str, Chunk]) -> tuple[bool, str | None]:
    cited_text = " ".join(chunk_lookup[cid].text for cid in chunk_ids if cid in chunk_lookup)
    if not cited_text:
        return True, "no cited passage text available to check against"

    claim_numbers = _numbers_in(claim_text)
    cited_numbers = _numbers_in(cited_text)
    missing_numbers = claim_numbers - cited_numbers
    if missing_numbers:
        return True, f"number(s) {', '.join(sorted(missing_numbers))} not found in the cited passage(s)"

    claim_entities = _entities_in(claim_text)
    cited_entities = _entities_in(cited_text)
    if claim_entities:
        overlap = len(claim_entities & cited_entities) / len(claim_entities)
        if overlap < ENTITY_OVERLAP_THRESHOLD:
            return True, "claim shares little vocabulary with its cited passage(s)"

    return False, None


def apply_evidence_validator(answer: Answer, chunk_lookup: dict[str, Chunk]) -> Answer:
    for claim in answer.claims:
        flagged, reason = validate_claim(claim.text, claim.chunk_ids, chunk_lookup)
        claim.flagged = flagged
        claim.flag_reason = reason
    return answer
