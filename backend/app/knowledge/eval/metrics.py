"""Eval metrics (spec §8).

Recall@20 is a pure retrieval metric — computed here from chunk_id overlap,
no LLM involved, matching build order step 5. Citation precision,
faithfulness, and abstention rate all need an Answer (step 6) to exist, so
their functions take one; run_eval.py only calls them once that step is
wired up.
"""

from __future__ import annotations

import re

from app.knowledge.schema import Answer, Chunk


def recall_at_k(expected_chunk_ids: list[str], retrieved_chunk_ids: list[str], k: int = 20) -> float:
    """1.0 if any expected chunk is in the top k retrieved, else 0.0.

    Binary per-question hit rate (not fraction of expected chunks found) —
    for these questions one supporting chunk is enough to answer correctly,
    so that's what "the retriever didn't lose the answer" means here.
    """
    if not expected_chunk_ids:
        return float("nan")  # not applicable — e.g. unanswerable questions
    top_k = set(retrieved_chunk_ids[:k])
    return 1.0 if any(cid in top_k for cid in expected_chunk_ids) else 0.0


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]*")


def _key_terms(text: str) -> set[str]:
    """Crude entity/number extraction: capitalized words, numbers, and
    hyphenated terms (gene names, drug names, percentages). Good enough for
    the "does the cited text actually contain this claim's content words"
    check spec §10 describes for the Evidence Validator's first version —
    reused here for citation precision rather than invented separately.
    """
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2}


def citation_precision(answer: Answer, chunk_lookup: dict[str, Chunk]) -> float:
    """Fraction of claims whose cited chunk(s) share enough key terms with
    the claim text that the citation plausibly supports it. Not a semantic
    entailment check — see the module docstring's caveat — but it catches
    the cheap failure mode of citing an unrelated chunk.
    """
    if not answer.claims:
        return float("nan")

    supported = 0
    for claim in answer.claims:
        if not claim.chunk_ids:
            continue  # already excluded from claims by the schema contract in practice
        claim_terms = _key_terms(claim.text)
        cited_text = " ".join(
            chunk_lookup[cid].text for cid in claim.chunk_ids if cid in chunk_lookup
        )
        cited_terms = _key_terms(cited_text)
        overlap = len(claim_terms & cited_terms) / max(1, len(claim_terms))
        if overlap >= 0.5:
            supported += 1

    return supported / len(answer.claims)


def faithfulness(answer: Answer, chunk_lookup: dict[str, Chunk]) -> float:
    """Same key-term-overlap heuristic as citation_precision, but pooled
    across ALL retrieved/cited chunks rather than per-claim — approximates
    "does the answer as a whole stick to what's in the passages."
    """
    if not answer.claims:
        return float("nan")

    all_cited_text = " ".join(
        chunk_lookup[cid].text
        for claim in answer.claims
        for cid in claim.chunk_ids
        if cid in chunk_lookup
    )
    cited_terms = _key_terms(all_cited_text)

    faithful = 0
    for claim in answer.claims:
        claim_terms = _key_terms(claim.text)
        overlap = len(claim_terms & cited_terms) / max(1, len(claim_terms))
        if overlap >= 0.5:
            faithful += 1

    return faithful / len(answer.claims)


def abstained(answer: Answer, question_id: str) -> bool:
    """True if the answer correctly produced no supported claims — the
    desired behavior for the 5 unanswerable questions (spec §6, §8).
    """
    return len(answer.claims) == 0 or all(len(c.chunk_ids) == 0 for c in answer.claims)
