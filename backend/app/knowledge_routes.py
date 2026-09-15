"""FastAPI routes connecting a Phase 1 case to the Phase 3 knowledge
pipeline.

    POST /api/cases/{job_id}/explain   proactive, evidence-grounded summary
    POST /api/cases/{job_id}/ask       a follow-up question against the corpus

Every measurement that reaches the LLM goes through phase1_bridge first
(spec §7) — this module never hands case.json fields to a prompt directly.
Every claim in the response already passed through answer_question's
schema enforcement (spec §6) before it got here.

Retrieval indexes and the LLM client are heavy to construct, so they're
lazy singletons built on first request, not at import time — importing
this module doesn't require the corpus/indexes to already exist, and the
main app can still start if this subsystem isn't set up yet.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.knowledge.agents.clinical_agent import run_default_analysis
from app.knowledge.agents.imaging_agent import answer_from_imaging
from app.knowledge.agents.planner import route_question
from app.knowledge.answering import (
    AnthropicClient,
    FallbackClient,
    GroqClient,
    StubClient,
    answer_question,
)
from app.knowledge.index.dense import MiniLMEmbedder
from app.knowledge.index.fuse import hybrid_search, load_chunk_lookup
from app.knowledge.schema import Answer
from app.knowledge.web_search import search_web
from app.knowledge.validator import apply_evidence_validator

log = logging.getLogger("neuroevidence.knowledge")
router = APIRouter(prefix="/api/cases", tags=["knowledge"])

JOBS = Path(os.getenv("JOB_DIR", Path(__file__).resolve().parent.parent / "jobs"))

_embedder = None
_chunk_lookup = None
_llm_client = None


def _embedder_singleton():
    global _embedder
    if _embedder is None:
        _embedder = MiniLMEmbedder()  # spec §5's measured winner on this corpus
    return _embedder


def _chunk_lookup_singleton():
    global _chunk_lookup
    if _chunk_lookup is None:
        try:
            _chunk_lookup = load_chunk_lookup()
        except FileNotFoundError as exc:
            raise HTTPException(
                503,
                "knowledge corpus not built yet — run "
                "`python -m app.knowledge.cli build-corpus` first",
            ) from exc
    return _chunk_lookup


def _llm_client_singleton():
    global _llm_client
    if _llm_client is None:
        # Every available provider, tried in this order until one answers —
        # not just picking the first key found. Groq only (no paid keys
        # configured) is the intended default; Anthropic is used as an
        # automatic fallback only if you've explicitly set that key too.
        candidates: list = []
        if os.getenv("GROQ_API_KEY"):
            candidates.append(GroqClient())
        if os.getenv("ANTHROPIC_API_KEY"):
            candidates.append(AnthropicClient())

        if not candidates:
            log.warning(
                "no LLM API key set (GROQ_API_KEY / ANTHROPIC_API_KEY) — "
                "falling back to StubClient, which does not produce real answers"
            )
            candidates = [StubClient()]

        _llm_client = candidates[0] if len(candidates) == 1 else FallbackClient(candidates)
    return _llm_client


def _gather_passages(
    query: str,
    chunk_lookup: dict,
    *,
    source_types: list[str] | None = None,
    tumour_type: str | None = None,
) -> list:
    """Curated corpus first, then a few live web results (allowlisted
    domains only — see web_search.py) topped up to spec §5's 6-8 passage
    budget. Corpus gets the majority of the slots deliberately: it's the
    vetted source, web search is a freshness/coverage supplement, not a
    replacement.
    """
    corpus_passages = hybrid_search(
        query,
        _embedder_singleton(),
        chunk_lookup,
        top_k_each=20,
        top_k_fused=5,
        source_types=source_types,
        tumour_type=tumour_type,
    )
    web_passages = search_web(query, max_results=3)
    return corpus_passages + web_passages


def _load_case(job_id: str) -> dict:
    if not job_id.isalnum():
        raise HTTPException(400, "bad job id")
    path = JOBS / job_id / "case.json"
    if not path.is_file():
        raise HTTPException(404, f"no case found for job {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _answer_to_dict(answer: Answer) -> dict:
    return {
        "claims": [
            {
                "text": c.text,
                "chunk_ids": c.chunk_ids,
                "flagged": c.flagged,
                "flag_reason": c.flag_reason,
            }
            for c in answer.claims
        ],
        "unsupported": answer.unsupported,
        "sources": [
            {
                "chunk_id": s.chunk_id,
                "title": s.title,
                "section": s.section,
                "url": s.url,
                "source_type": s.source_type,
            }
            for s in answer.sources
        ],
        "routed_to": "corpus",
    }


class AskBody(BaseModel):
    question: str
    # The immediately preceding question in this conversation, if any — so
    # a follow-up like "how do you fix that" has something to resolve
    # "that" against. Retrieval also needs it: searching for the literal
    # string "how do you fix that" finds nothing, since it has no medical
    # content of its own.
    previous_question: str | None = None


@router.post("/{job_id}/explain")
def explain_case(job_id: str):
    """Phase 4 Clinical Analysis Agent — the five default questions,
    generated automatically for every case. No retrieval: see
    app/knowledge/agents/clinical_agent.py for why, and for the two
    questions (growth duration, survival) that are hardcoded rather than
    ever asked of the model.
    """
    case = _load_case(job_id)
    return run_default_analysis(case, _llm_client_singleton())


@router.post("/{job_id}/ask")
def ask_case(job_id: str, body: AskBody):
    case = _load_case(job_id)  # 404s cleanly if the job doesn't exist
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question is empty")

    decision = route_question(question, has_case=True)
    log.info("planner: %r -> %s (%s)", question, decision.route, decision.reasoning)

    if decision.route == "imaging_agent":
        return answer_from_imaging(case)

    prior = (body.previous_question or "").strip() or None
    # Crude but effective: folding the prior question's words into the
    # retrieval query means a contentless follow-up ("how do you fix
    # that") still surfaces whatever the prior question was actually
    # about, instead of retrieving nothing.
    retrieval_query = f"{prior} {question}" if prior else question

    chunk_lookup = _chunk_lookup_singleton()
    passages = _gather_passages(
        retrieval_query,
        chunk_lookup,
        source_types=decision.source_types,
        tumour_type=decision.tumour_type,
    )
    answer = answer_question(question, passages, _llm_client_singleton(), prior_question=prior)
    answer = apply_evidence_validator(answer, chunk_lookup)
    result = _answer_to_dict(answer)
    result["planner_reasoning"] = decision.reasoning
    return result
