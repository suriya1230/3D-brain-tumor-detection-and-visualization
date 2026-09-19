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
import threading
from concurrent.futures import ThreadPoolExecutor
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
from app.knowledge.cache import TTLCache
from app.knowledge.hallucination_detector import apply_hallucination_detector
from app.knowledge.index.dense import MiniLMEmbedder
from app.knowledge.index.fuse import hybrid_search, load_chunk_lookup
from app.knowledge.schema import Answer
from app.knowledge.web_search import search_web
from app.knowledge.validator import apply_citation_policy, apply_evidence_validator

log = logging.getLogger("neuroevidence.knowledge")
router = APIRouter(prefix="/api/cases", tags=["knowledge"])

JOBS = Path(os.getenv("JOB_DIR", Path(__file__).resolve().parent.parent / "jobs"))

_embedder = None
_chunk_lookup = None
_llm_client = None
# clinical_agent.py's run_default_analysis calls retrieve_fn (-> this
# module's singletons) from three threads concurrently - see dense.py's
# _client() docstring for the real race this caused when these were plain
# check-then-set globals with no lock.
_singleton_lock = threading.Lock()

# See cache.py's module docstring for why these are bounded-TTL, not
# unbounded. Separate caches, not one, so a rewrite-cache eviction storm
# can't push out cited-answer entries and vice versa.
_rewrite_cache = TTLCache()
_cited_answer_cache = TTLCache()


def _embedder_singleton():
    global _embedder
    if _embedder is None:
        with _singleton_lock:
            if _embedder is None:
                _embedder = MiniLMEmbedder()  # spec §5's measured winner on this corpus
    return _embedder


def _chunk_lookup_singleton():
    global _chunk_lookup
    if _chunk_lookup is None:
        with _singleton_lock:
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
        with _singleton_lock:
            if _llm_client is None:
                # Every available provider, tried in this order until one
                # answers — not just picking the first key found. Groq only
                # (no paid keys configured) is the intended default;
                # Anthropic is used as an automatic fallback only if you've
                # explicitly set that key too.
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

                _llm_client = (
                    candidates[0] if len(candidates) == 1 else FallbackClient(candidates)
                )
    return _llm_client


_rewrite_client = None


def _rewrite_client_singleton():
    """A separate, deliberately smaller/faster model for query rewriting
    specifically - measured directly: a trivial completion took 5.4s on
    the primary model (gpt-oss-120b) vs 0.5s on the smaller one
    (gpt-oss-20b), and the gap widens further under real retrieval-sized
    prompts. Safe to use the smaller model here even though GroqClient's
    own comment notes it's less reliable at the citation-FORMAT contract
    (that matters for answer_question) - rewriting a question into
    cleaner search text has no citation format to get wrong.

    Falls back to the primary client if Groq isn't configured at all
    (e.g. Anthropic-only setups don't get this fast path).
    """
    global _rewrite_client
    if _rewrite_client is None:
        with _singleton_lock:
            if _rewrite_client is None:
                if os.getenv("GROQ_API_KEY"):
                    _rewrite_client = GroqClient(
                        model=os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")
                    )
                else:
                    _rewrite_client = _llm_client_singleton()
    return _rewrite_client


# Phase 4's model only ever assumes "glioma" broadly (no classification
# head - see clinical_agent.py's _scan_type_assumption), so there's no
# specific WHO subtype id to INCLUDE-filter on - but these three ARE
# definitely not what it's describing. A 2026-09 review caught metastasis-
# specific management advice (corticosteroids/anticonvulsants framed for
# brain metastases, whole-brain radiotherapy) getting cited into a glioma
# report; excluding these categories from Q2/Q5's retrieval is the filter
# that's actually available. Does NOT reach web search results below -
# Tavily has no tumour-type metadata to filter on, so that residual path
# still depends on the query itself being specific enough.
NON_GLIOMA_TUMOUR_TYPES = ["brain_metastasis", "leptomeningeal_metastasis", "meningioma"]


def _gather_passages(
    query: str,
    chunk_lookup: dict,
    *,
    source_types: list[str] | None = None,
    tumour_type: str | None = None,
    exclude_tumour_types: list[str] | None = None,
) -> list:
    """Curated corpus first, then a few live web results (allowlisted
    domains only — see web_search.py) topped up to spec §5's 6-8 passage
    budget. Corpus gets the majority of the slots deliberately: it's the
    vetted source, web search is a freshness/coverage supplement, not a
    replacement.

    The two lookups have no data dependency on each other - one is local
    compute (BM25 + dense embedding), the other a network round-trip to
    Tavily - so they run on separate threads instead of paying both
    latencies back to back.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        corpus_future = pool.submit(
            hybrid_search,
            query,
            _embedder_singleton(),
            chunk_lookup,
            top_k_each=20,
            top_k_fused=5,
            source_types=source_types,
            tumour_type=tumour_type,
            exclude_tumour_types=exclude_tumour_types,
        )
        web_future = pool.submit(search_web, query, max_results=3)
        corpus_passages = corpus_future.result()
        web_passages = web_future.result()
    return corpus_passages + web_passages


def _rewrite_query(question: str, prior_question: str | None) -> str:
    """Cached wrapper — see cache.py. Same exact (question, prior_question)
    pair skips a full LLM round-trip, which matters here specifically
    because this call sits in front of every retrieval, on the critical
    path for both /explain and /ask latency.
    """
    key = f"{prior_question or ''}||{question}"
    return _rewrite_cache.get_or_compute(key, lambda: _rewrite_query_uncached(question, prior_question))


def _rewrite_query_uncached(question: str, prior_question: str | None) -> str:
    """A malformed or informally-phrased question hurts retrieval directly,
    not just readability — e.g. "how affected this tumer" gets matched by
    dense retrieval on "affected" toward treatment-effect literature,
    because there's no clean query for it to match on the actual intent
    ("how does this tumor affect the patient"). Rewriting into clear
    search-oriented English before hybrid_search fixes that at the source
    instead of trying to compensate downstream in ranking.

    Degrades to the original question on any failure — a bad rewrite must
    never be worse than no rewrite, so this never raises and never returns
    an empty string.
    """
    prior_note = (
        f' The previous question in this conversation was: "{prior_question}".'
        if prior_question
        else ""
    )
    prompt = (
        f"Rewrite the following question into a single, clear, "
        f"well-formed English search query about a brain tumor, "
        f"preserving its intent exactly — do not answer it, do not add "
        f"information it doesn't ask for. Prefer standard clinical "
        f"terminology over vague everyday words where it changes what the "
        f"query would match — e.g. a question about how a tumor affects "
        f"the patient should use \"signs and symptoms\" or \"clinical "
        f"presentation\", not just \"affect\", because that's the "
        f"terminology this corpus actually uses.{prior_note}\n\n"
        f"Question: {question!r}\n\n"
        f"Output ONLY the rewritten question, nothing else."
    )
    try:
        raw = _rewrite_client_singleton().complete(
            "You rewrite malformed or informally-phrased questions into "
            "clear search queries using standard clinical terminology. "
            "Output only the rewritten question, nothing else — no "
            "quotes, no preamble.",
            prompt,
        )
        rewritten = raw.strip().strip('"')
        if rewritten:
            log.info("query rewrite: %r -> %r", question, rewritten)
            return rewritten
    except Exception:
        log.exception("query rewrite failed for %r — using original question", question)
    return question


def _cited_answer(
    query: str,
    *,
    source_types: list[str] | None = None,
    tumour_type: str | None = None,
    exclude_tumour_types: list[str] | None = None,
    prior_question: str | None = None,
    prompt_question: str | None = None,
) -> dict:
    """The full Phase 3 pipeline — retrieval, citation-gated generation,
    both validators — for one question, cached (see cache.py) and shared
    by both call sites that need it: clinical_agent.py's run_default_
    analysis (as `retrieve_fn`, for brain_effects/treatment_information —
    see that module's docstring for why those aren't a free-form LLM call
    any more) and /ask below. One implementation, not two copies that can
    quietly drift apart.

    `query` drives retrieval (may be a rewritten/prior-folded version of
    what the user actually typed); `prompt_question` is what the LLM is
    asked to answer (defaults to `query` when the two are the same thing,
    which is always true for clinical_agent.py's callback — it constructs
    already-clean questions, there's nothing to rewrite).
    """
    cache_key = (
        f"{query}||{prompt_question or ''}||{source_types}||{tumour_type}||"
        f"{exclude_tumour_types}||{prior_question or ''}"
    )

    def _compute() -> dict:
        chunk_lookup = _chunk_lookup_singleton()
        passages = _gather_passages(
            query,
            chunk_lookup,
            source_types=source_types,
            tumour_type=tumour_type,
            exclude_tumour_types=exclude_tumour_types,
        )
        answer = answer_question(
            prompt_question or query,
            passages,
            _llm_client_singleton(),
            prior_question=prior_question,
        )
        validator_lookup = _validator_lookup(chunk_lookup, passages)
        answer = apply_evidence_validator(answer, validator_lookup)
        answer = apply_hallucination_detector(answer, validator_lookup, prompt_question or query)
        # Two independent flags -> drop; every surviving claim still
        # flagged -> abstain the section. See validator.py's docstring.
        answer = apply_citation_policy(answer)
        return _answer_to_dict(answer)

    return _cited_answer_cache.get_or_compute(cache_key, _compute)


def _validator_lookup(chunk_lookup: dict, passages: list) -> dict:
    """apply_evidence_validator/apply_hallucination_detector need the text
    of whatever was actually cited — but web-search results (search_web,
    called inside _gather_passages) are real Chunks that were shown to the
    LLM and can legitimately get cited, yet they don't live in the static
    corpus chunk_lookup. Without this, every web-sourced claim gets flagged
    "no cited passage text available to check against" regardless of
    whether it's actually correct — a validator gap, not a signal about
    the claim. Passages take priority on id collision since they're the
    exact text version shown to the model for this call.
    """
    return {**chunk_lookup, **{p.chunk_id: p for p in passages}}


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
    generated automatically for every case. Two (growth duration, survival)
    are hardcoded rather than ever asked of the model; two (brain effects,
    treatment) go through the same citation-gated retrieval pipeline as
    /ask via _cited_answer; one (what the tumor category generally means)
    is a single unsourced LLM call for genuinely general classification-
    level education. See app/knowledge/agents/clinical_agent.py.
    """
    case = _load_case(job_id)
    # Bakes in the glioma-only exclusion (see NON_GLIOMA_TUMOUR_TYPES)
    # before handing this to clinical_agent.py, which doesn't need to know
    # tumour-type filtering exists at all - it just calls retrieve_fn(question).
    retrieve_fn = lambda question: _cited_answer(  # noqa: E731
        question, exclude_tumour_types=NON_GLIOMA_TUMOUR_TYPES
    )
    return run_default_analysis(case, _llm_client_singleton(), retrieve_fn)


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
    # Clean up the question itself before it ever reaches retrieval —
    # malformed/informal phrasing ("how affected this tumer") matches the
    # wrong things in dense retrieval, see _rewrite_query's docstring.
    rewritten = _rewrite_query(question, prior)
    # Crude but effective: folding the prior question's words into the
    # retrieval query means a contentless follow-up ("how do you fix
    # that") still surfaces whatever the prior question was actually
    # about, instead of retrieving nothing.
    retrieval_query = f"{prior} {rewritten}" if prior else rewritten

    result = _cited_answer(
        retrieval_query,
        source_types=decision.source_types,
        tumour_type=decision.tumour_type,
        prior_question=prior,
        prompt_question=question,
    )
    result["planner_reasoning"] = decision.reasoning
    return result
