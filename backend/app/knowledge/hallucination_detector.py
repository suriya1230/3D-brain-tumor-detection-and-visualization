"""Optional second-opinion hallucination check on top of the Evidence
Validator (validator.py, spec §10).

validator.py's check is deliberately narrow and hand-rolled: numbers and
entity overlap only, "achievable and honest" per the spec rather than
claiming to solve claim verification in general. LettuceDetect
(https://github.com/KRLabsOrg/LettuceDetect, MIT) is a fine-tuned ModernBERT
model that does span-level grounding verification properly - it can catch
things the word/number-overlap check can't (a claim can share every number
and every keyword with its cited passage and still misstate the
relationship between them). This module adds it as a SECOND, independent
check, not a replacement - a claim only needs one validator to flag it.

Optional dependency, on purpose: see requirements-hallucination.txt for why
it isn't folded into requirements-knowledge.txt (a real torch version
conflict with Phase 1's pinned torch==2.4.1, not a style choice). Every
function here degrades to a no-op if the package isn't installed, the same
contract every other optional piece of this pipeline follows (Tavily,
NCBI_API_KEY, the LLM clients' fallback chain) - missing an optional
capability is not a request failure.
"""

from __future__ import annotations

import logging
import threading

from app.knowledge.schema import Answer, Chunk

log = logging.getLogger("lumenbrain.knowledge")

try:
    from lettucedetect.models.inference import HallucinationDetector

    _AVAILABLE = True
except ImportError:
    HallucinationDetector = None  # type: ignore[assignment]
    _AVAILABLE = False

# The base ModernBERT model, not "large" - CPU-friendly, matches how the
# rest of this pipeline runs (MiniLM for dense retrieval, not a bigger
# embedder, chosen the same way per spec §5: measured, not assumed).
MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

# LettuceDetect's own reported span-F1 is in the 0.6-0.7 range (see its
# README) - nowhere near calibrated enough to call this an error
# probability. This threshold is a heuristic filter on its confidence
# output, same honesty rule this project applies to every other
# confidence-shaped number (mean_probability, tta_uncertainty): it decides
# what counts as "flag this", not what fraction of flags are correct.
MIN_CONFIDENCE = 0.7

_detector = None
# clinical_agent.py's run_default_analysis calls this module from multiple
# threads concurrently (brain_effects and treatment_information are
# independent retrieve_fn calls run in parallel) - a plain check-then-set
# global here has the same race dense.py's _client() hit for real: see
# that function's docstring.
_detector_lock = threading.Lock()


def is_available() -> bool:
    return _AVAILABLE


def _singleton():
    global _detector
    if _detector is None and _AVAILABLE:
        with _detector_lock:
            if _detector is None:
                log.info("loading LettuceDetect (%s) - first call only", MODEL_PATH)
                _detector = HallucinationDetector(method="transformer", model_path=MODEL_PATH)
    return _detector


def flagged_spans(context: list[str], question: str, answer: str) -> list[dict]:
    """Returns LettuceDetect's unsupported spans for this (context,
    question, answer) triple, filtered to MIN_CONFIDENCE. Never raises -
    a model/runtime failure degrades to "found nothing", same as apply_
    evidence_validator does for a claim with no cited text at all. That's
    a deliberate choice, not a shortcut: this is a second opinion on top of
    a validator that already runs unconditionally, so losing it should
    never take the request down.

    The actual detector.predict() call is serialized process-wide via
    _detector_lock, not just its construction - HuggingFace's fast
    tokenizer has internal mutable state (truncation/padding config) that
    is not safe under concurrent use, and this function gets called from
    more than one call site that can legitimately run on different
    threads at the same time: clinical_agent.py's run_default_analysis
    runs brain_effects and treatment_information concurrently
    (ThreadPoolExecutor), each independently reaching here. A per-call
    lock inside apply_hallucination_detector alone doesn't cover that -
    only serializing the shared resource itself does. This crashed the
    server once already (RuntimeError: Already borrowed, sometimes
    unrecoverable) before this lock was added.
    """
    if not _AVAILABLE:
        return []
    try:
        detector = _singleton()
        with _detector_lock:
            spans = detector.predict(
                context=context, question=question, answer=answer, output_format="spans"
            )
    except Exception:
        log.exception("LettuceDetect call failed")
        return []
    return [s for s in spans if s.get("confidence", 0) >= MIN_CONFIDENCE]


def apply_hallucination_detector(
    answer: Answer, chunk_lookup: dict[str, Chunk], question: str
) -> Answer:
    """Per-claim, mirroring apply_evidence_validator's structure: each
    claim is checked against only its own cited chunks, not the whole
    corpus, because a claim can be perfectly supported by chunk A while
    LettuceDetect-against-chunk-B would look unsupported for reasons that
    have nothing to do with whether the claim itself is correct.

    A no-op (including if the package isn't installed) - claims already
    flagged by validator.py stay flagged, and this never unflags anything.

    Deliberately sequential, NOT parallelized across claims - tried that,
    and it crashed the server. HuggingFace's fast tokenizer (Rust-backed)
    has its own internal mutable state for truncation/padding config that
    is not safe to touch from multiple threads at once: concurrent calls
    hit `RuntimeError: Already borrowed` (a Rust-side runtime borrow-check
    panic) inconsistently - sometimes catchable, sometimes taking the
    whole process down with it, which is exactly what happened here. The
    model weights themselves (pure inference) would have been fine to
    parallelize; the tokenizer they share is what isn't.
    """
    if not _AVAILABLE:
        return answer

    for claim in answer.claims:
        context = [chunk_lookup[cid].text for cid in claim.chunk_ids if cid in chunk_lookup]
        if not context:
            continue  # validator.py already flagged this - nothing to check here

        spans = flagged_spans(context, question, claim.text)
        if not spans:
            continue

        note = "LettuceDetect flagged unsupported span(s) in this claim"
        if claim.flagged and claim.flag_reason:
            claim.flag_reason = f"{claim.flag_reason}; {note}"
        else:
            claim.flagged = True
            claim.flag_reason = note
        claim.flag_sources.append("lettucedetect")

    return answer
