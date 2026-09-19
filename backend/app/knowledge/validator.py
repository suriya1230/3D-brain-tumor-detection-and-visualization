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
        if flagged:
            claim.flag_sources.append("evidence_validator")
    return answer


# A 2026-09 review of a real report found a claim carrying two independent
# flags (this validator's low word-overlap AND hallucination_detector.py's
# LettuceDetect span flag) rendered anyway, with a warning icon — leaving
# it to the reader to notice and overrule two validators that already
# agreed it wasn't supported. Two agreeing signals is categorically
# stronger than one; the policy below acts on that difference instead of
# treating every flagged claim the same:
#
#   1 flag  -> render with the warning (a human can weigh one signal)
#   2 flags -> drop the claim entirely, don't render it at all
#   every surviving claim in the section still flagged -> abstain the
#   whole section (claims=[]) rather than show a page of warned claims,
#   which is the same "a validator that silently approves is worse than
#   none" principle applied one level up: a section that's all caveats
#   isn't actually telling the reader anything either.
TWO_FLAG_DROP_THRESHOLD = 2


def apply_citation_policy(answer: Answer) -> Answer:
    kept = [c for c in answer.claims if len(c.flag_sources) < TWO_FLAG_DROP_THRESHOLD]
    if kept and all(c.flagged for c in kept):
        kept = []
    answer.claims = kept
    cited_ids = {cid for c in kept for cid in c.chunk_ids}
    answer.sources = [s for s in answer.sources if s.chunk_id in cited_ids]
    return answer
