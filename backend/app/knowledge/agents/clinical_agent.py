"""Phase 4 Clinical Analysis Agent.

Two questions get NO model call at all, ever — growth duration and
survival/prognosis. Both are hardcoded, deterministic, and identical for
every case, because no single MRI can answer either one, and that's not a
retrieval gap to paper over — it's a structural fact about what a
single-timepoint segmentation can know. See ASSESS_GROWTH / ASSESS_PROGNOSIS.

One question (what is this tumor category, in general terms) gets a single
unsourced LLM call — genuinely general classification-level education
("what a glioma is"), not a specific medical claim someone could act on.

The other two (how the location can affect the brain, what treatments are
typically used) used to also be free-form, uncited LLM output under a
system prompt that constrained by topic and hedging language instead of by
citation. A 2026-09 review of a real case caught exactly the failure mode
that design invited: an uncited paragraph stated a left occipital lesion
causes loss of the *left* visual field — backwards; left occipital cortex
serves the *right* visual field. The paragraph read fluently and cited
nothing, so nothing caught it before a clinician would have to. Constraining
a prompt by topic doesn't stop it from stating a specific anatomical fact
wrong — only a citation trail does, and only if something checks it.

So brain_effects and treatment_information now go through the same
citation-gated pipeline as Phase 3's /ask (hybrid_search + answer_question
+ apply_evidence_validator + apply_hallucination_detector), via the
`retrieve_fn` callback knowledge_routes.py passes in — this module doesn't
import the retrieval stack directly, to keep it possible to unit-test
run_default_analysis with a fake retriever.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.knowledge.answering import LLMClient, _strip_code_fence  # reuse, don't duplicate
from app.knowledge.cache import TTLCache
from app.knowledge.phase1_bridge import extract_case_facts

log = logging.getLogger("neuroevidence.knowledge")

# category is almost always literally "Glioma" (this model has no other
# training data - see _category_from_model), so this cache is a near-
# guaranteed hit across every case after the first, the same way
# knowledge_routes.py's cache is for the treatment_information question.
_tumor_explanation_cache = TTLCache()

# Genuinely general, classification-level education only ("what is a
# glioma") — not a location- or treatment-specific claim, which is exactly
# the distinction that matters here. See module docstring.
SYSTEM_PROMPT_TUMOR = """\
You are a clinical education assistant. Given a tumor category name, \
explain in 2-4 sentences what that category generally means as a class of \
finding — general, well-established medical knowledge only.

Rules:
1. NEVER state or imply: a specific diagnosis, a WHO grade, a survival \
percentage, a life expectancy, or how long a tumor has existed.
2. Output ONLY a JSON object of this exact shape, nothing else:
{"explanation": "..."}
"""

# ---------------------------------------------------------------------------
# Hardcoded, deterministic, no LLM involved — the two questions no single
# MRI can answer. Identical for every case, on purpose.
# ---------------------------------------------------------------------------

ASSESS_GROWTH = {
    "estimate": "Cannot be reliably determined from a single MRI.",
    "reason": "Growth rate requires comparison with prior imaging at a "
    "different timepoint. This case has one scan, at one point in time.",
    "future_capability": "If prior scans for this patient are provided "
    "(e.g. from an earlier session), volume change, growth rate, and "
    "changes in segmentation characteristics between them could be "
    "computed — not supported yet.",
}

ASSESS_PROGNOSIS = {
    "individual_survival_probability": "Not reliably estimable from this "
    "MRI segmentation alone.",
    "required_factors": [
        "Tumor type",
        "Tumor grade",
        "Molecular characteristics",
        "Tumor location",
        "Patient age and general health",
        "Extent of tumor removal",
        "Residual disease",
        "Treatment response",
    ],
    "note": "These are established factors in prognosis for CNS tumors. "
    "None of them are available from an imaging segmentation by itself; "
    "estimating survival responsibly would need a separately validated "
    "prognostic model built on properly labeled clinical outcome data, "
    "not this pipeline.",
}

STANDING_WARNINGS = [
    "This is an AI-assisted analysis of an imaging segmentation, reviewed "
    "by a clinician — not an autonomous diagnosis.",
    "The segmentation model has no classification head — the scan type "
    "below is an assumption from what the model was trained on, not a "
    "prediction it made about this scan.",
    "Growth duration cannot be determined from a single scan.",
    "Individual survival/prognosis cannot be estimated from this scan alone.",
]

# Heuristic, not calibrated: inference.py's TTA disagreement score has no
# per-case holdout data to fit a threshold against yet (see the go/no-go
# step in app/meshing.py's meta["confidence"]["note"]). 0.08 is a round,
# conservative cutoff on the sigmoid-probability std, chosen so this flags
# only clearly-disagreeing regions rather than routine boundary noise —
# revisit once real calibration data exists.
UNCERTAINTY_FLAG_THRESHOLD = 0.08


def _category_from_model(model_desc: str) -> str:
    model_desc = (model_desc or "").lower()
    if "glioma" in model_desc:
        return "Glioma"
    return "Unknown (model description did not name a tumor category)"


def _scan_type_assumption(category: str) -> str:
    """NOT a "predicted category" - the model has four segmentation output
    channels (TC/WT/ET/RC) and no classification head at all. "Glioma" is
    true of every case because BraTS 2024 post-treatment GLI, the training
    set, is glioma-only - the model would say the same thing about a
    meningioma or a metastasis. Reporting that as a "prediction" is exactly
    the failure mode spec §7 was written about: a property of the training
    data, presented as a finding about this scan."""
    if category == "Glioma":
        return (
            "Glioma. The segmentation model was trained only on "
            "post-treatment glioma and has no tumor-type classification "
            "head — it cannot distinguish glioma from meningioma, "
            "metastasis, or any other tumor type. This is a scan-type "
            "assumption inherited from training data, not a prediction "
            "about this scan."
        )
    return category


def _generate_tumor_explanation(category: str, client: LLMClient) -> str:
    return _tumor_explanation_cache.get_or_compute(
        category, lambda: _generate_tumor_explanation_uncached(category, client)
    )


def _generate_tumor_explanation_uncached(category: str, client: LLMClient) -> str:
    user_prompt = f"Tumor category: {category}\n\nExplain what this category generally means."
    try:
        raw = client.complete(SYSTEM_PROMPT_TUMOR, user_prompt)
        parsed = json.loads(_strip_code_fence(raw))
        explanation = (parsed.get("explanation") or "").strip()
        if explanation:
            return explanation
        log.warning("clinical agent tumor-explanation JSON had no explanation field")
    except Exception:
        log.exception("clinical agent tumor-explanation LLM call failed")

    return (
        f"A general explanation could not be generated right now (answer "
        f"service unavailable) — {category} above is a segmentation-model "
        f"assumption, not a confirmed diagnosis or WHO grade, which require "
        f"pathology and molecular testing."
    )


# meshing.py's _lobe_namer buckets are coarse y/z-coordinate thresholds,
# not real neuroanatomy - "central region" specifically is this project's
# own made-up label for its fallback bucket (roughly the region around the
# central sulcus) and does not appear in medical literature at all, so a
# retrieval query built from it verbatim can't match anything: neither
# BM25 (no literal term to match) nor dense embeddings (too vague to be
# close to anything specific) find the actual content, and the section
# abstains - not because the evidence doesn't exist (motor cortex anatomy
# is not obscure), but because the query never named it. Expanded here for
# RETRIEVAL only; the displayed location (facts["locations"], via
# phase1_bridge's honesty qualifiers) stays the coarse label, unchanged.
_RETRIEVAL_TERM_FOR_REGION = {
    "central region": (
        "central region (precentral gyrus / primary motor cortex, "
        "postcentral gyrus / primary somatosensory cortex)"
    ),
}


def _expand_location_for_retrieval(location: str) -> str:
    for coarse, expanded in _RETRIEVAL_TERM_FOR_REGION.items():
        if location.endswith(coarse):
            side = location[: -len(coarse)].strip()
            return f"{side} {expanded}".strip()
    return location


def _brain_effects_question(locations: list[str]) -> str:
    if not locations:
        return (
            "What brain functions are generally associated with different "
            "brain regions, and what possible effects can a tumor have on "
            "a patient?"
        )
    where = " and ".join(_expand_location_for_retrieval(loc) for loc in locations)
    return (
        f"What brain functions are associated with the {where}, and what "
        f"possible effects can a tumor located there have on a patient?"
    )


def _treatment_question(category: str) -> str:
    return f"What are the typical treatment approaches for {category}?"


def _segmentation_confidence(facts: dict) -> tuple[dict, list[str]]:
    """Per-region TTA disagreement -> (structured summary, hedge notes).

    This is metacognition applied at exactly one place in the pipeline
    (inference.py's TTA pass) and surfaced here, the same way ASSESS_GROWTH
    / ASSESS_PROGNOSIS surface a structural "this can't be known" instead of
    staying silent about it — a region with elevated self-disagreement gets
    named, not averaged away into a single case-level confidence number.
    """
    region_uncertainty = {
        key: r["tta_uncertainty"]
        for key, r in facts["regions"].items()
        if r.get("tta_uncertainty") is not None
    }

    notes = [
        f"{key} boundary showed elevated disagreement across repeated "
        f"passes of this scan (uncertainty {unc:.3f}) — treat this "
        f"region's extent as less reliable than the others in this case."
        for key, unc in region_uncertainty.items()
        if unc >= UNCERTAINTY_FLAG_THRESHOLD
    ]

    summary = {
        "region_uncertainty": {k: round(v, 4) for k, v in region_uncertainty.items()},
        "flag_threshold": UNCERTAINTY_FLAG_THRESHOLD,
        "note": (
            "A raw self-disagreement score from test-time augmentation on "
            "this scan, not a calibrated probability of error — see "
            "app/meshing.py's meta[\"confidence\"] for the same caveat."
        ),
    }
    return summary, notes


def run_default_analysis(
    case: dict, client: LLMClient, retrieve_fn: Callable[[str], dict]
) -> dict:
    """retrieve_fn(question) -> {"claims", "unsupported", "sources"} in the
    same shape knowledge_routes.py's _answer_to_dict() produces: the full
    Phase 3 pipeline (hybrid_search, citation-gated generation, both
    validators), run for one question. Used for brain_effects and
    treatment_information — see module docstring for why those two
    specifically no longer come from a free-form LLM call.
    """
    facts = extract_case_facts(case)
    category = _category_from_model(facts["model"])
    locations = facts["locations"]
    confidence, confidence_notes = _segmentation_confidence(facts)

    # Three independent, network-bound calls (LLM / retrieval+LLM) with no
    # data dependency between them - run them concurrently rather than
    # paying their latency three times in a row. Threads, not asyncio: the
    # LLM/retrieval clients called here are synchronous.
    with ThreadPoolExecutor(max_workers=3) as pool:
        tumor_future = pool.submit(_generate_tumor_explanation, category, client)
        brain_future = pool.submit(retrieve_fn, _brain_effects_question(locations))
        treatment_future = pool.submit(retrieve_fn, _treatment_question(category))

        tumor_explanation = tumor_future.result()
        brain_effects_cited = brain_future.result()
        treatment_cited = treatment_future.result()

    tumor_analysis = {
        "tumor_detected": facts["tumor_detected"],
        "scan_type_assumption": _scan_type_assumption(category),
        # Explicitly the whole-tumor region — see phase1_bridge.py's
        # extract_case_facts, which derives this from WT_cc specifically.
        # TC/ET/RC volumes are different numbers; a bare "tumor volume"
        # label invites a reader to assume whichever region they habitually
        # mean.
        "whole_tumor_volume_cc": facts["total_volume_cc"],
        "model_holdout_dice": facts["holdout_dice"],
        "explanation": tumor_explanation,
    }

    brain_effects = {"locations": locations, **brain_effects_cited}

    return {
        "tumor_analysis": tumor_analysis,
        "brain_effects": brain_effects,
        "growth_assessment": ASSESS_GROWTH,
        "prognosis_assessment": ASSESS_PROGNOSIS,
        "treatment_information": treatment_cited,
        "segmentation_confidence": confidence,
        "warnings": STANDING_WARNINGS + confidence_notes,
        "sources": [],
    }
