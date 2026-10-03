"""Phase 4 Clinical Analysis Agent.

Two questions get NO model call at all, ever — growth duration and
survival/prognosis. Both are hardcoded, deterministic, and identical for
every case, because no single MRI can answer either one, and that's not a
retrieval gap to paper over — it's a structural fact about what a
single-timepoint segmentation can know. See ASSESS_GROWTH / ASSESS_PROGNOSIS.

One question (what is this tumor category, in general terms) gets an
unsourced LLM call PER possible category — genuinely general
classification-level education ("what a glioma is"), not a specific
medical claim someone could act on. "Per possible category", not one
call, because the 2026-10 retrain (inference.py's module docstring) has
no classification head and was trained on BraTS-GLI + BraTS-PED (glioma,
adult and pediatric) AND BraTS-MEN (meningioma) combined - there is no
single "the category" to explain. Collapsing that into one vague blended
answer, or silently picking the more common one, would both be a worse
kind of dishonesty than just answering the question twice.

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

log = logging.getLogger("lumenbrain.knowledge")

# The only two categories this checkpoint was ever trained on (see
# inference.py's module docstring) - hardcoded, not inferred from a model
# description string, because this is what the checkpoint IS, not a
# guess. "Glioma" here covers both BraTS-GLI (adult) and BraTS-PED
# (pediatric): the general-education explanation of "what a glioma is"
# doesn't meaningfully differ by age group, so splitting those two would
# just be two near-duplicate paragraphs, not more honesty. Meningioma is a
# genuinely different disease family and gets its own.
TRAINED_CATEGORIES = ["Glioma", "Meningioma"]

# Every case runs both categories through this cache, so this is a
# near-guaranteed hit after the first case of each kind, the same way
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


def _scan_type_assumption() -> str:
    """NOT a "predicted category" - the model has four segmentation output
    channels (TC/WT/ET/RC) and no classification head at all. It was
    trained on three cohorts combined - adult post-treatment glioma,
    meningioma, and pediatric glioma (see TRAINED_CATEGORIES's docstring
    for why the explanation text only generates for two: pediatric glioma
    gets folded into "Glioma" there since the general education content
    doesn't differ by age group) - so it would produce a segmentation for
    any of the three - or for a metastasis, or anything else - without any
    way to say which. This sentence must name all three cohorts explicitly,
    even though TRAINED_CATEGORIES has two entries: collapsing pediatric
    glioma out of the scan-type assumption itself (as opposed to out of the
    educational text) would hide that it's a real, distinct possibility
    with its own holdout performance (see cohort_dice_range)."""
    return (
        "Adult post-treatment glioma, meningioma, or pediatric glioma. The "
        "segmentation model was trained on these three tumor-type cohorts "
        "combined and has no tumor-type "
        f"classification head — it cannot tell you which one (if any) "
        f"this scan actually is, or distinguish any of them from a "
        f"metastasis or any other tumor type. This is a scan-type assumption "
        f"inherited from training data, not a prediction about this scan."
    )


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

    # RC (resection cavity) ground truth exists only in the BraTS-GLI
    # training cohort - BraTS-MEN/BraTS-PED's own labeling protocols never
    # define an RC class, so the model's RC output for those cases was
    # never checked against anything during training or evaluation (see
    # meshing.py's RC_VALIDATED_COHORTS_ONLY). This is not just "unmeasured
    # elsewhere" - test_lesionwise.csv's "gate tau=0.6 (single)" rows show
    # the model actively over-predicts RC outside BraTS-GLI: 113 spurious
    # RC components on the meningioma test cases and 13 on pediatric-glioma,
    # both against zero RC ground truth there (that labeling protocol has
    # no RC class at all). The shipped gate removes most of these (113->11
    # on meningioma, 13->0 on pediatric) but not all of them on meningioma.
    # Pooled RC lesion precision is 0.254 unfiltered, 0.813 gated - almost
    # all of that gated precision comes from the adult-glioma cases (0.925
    # there) propped up against the 11 meningioma false positives that
    # survive the gate. There's no way to know which cohort THIS case
    # resembles, so this note only fires when RC was actually predicted -
    # no sense warning about a region that isn't even present.
    if "RC" in facts["regions"]:
        notes.append(
            "Resection cavity (RC) is over-predicted by this model outside "
            "the adult post-treatment glioma cohort it was validated "
            "against — on the held-out test set, the gate still let "
            "through RC false positives on meningioma scans (none had a "
            "real resection cavity to find). If this case is a meningioma "
            "or a pediatric glioma, the predicted RC region is more likely "
            "to be a false positive than a real cavity, not merely "
            "unverified."
        )

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
    locations = facts["locations"]
    confidence, confidence_notes = _segmentation_confidence(facts)

    # One tumor-explanation call and one treatment retrieval PER trained
    # category, plus the one brain-effects call - all independent of each
    # other, run concurrently rather than paying each latency in turn.
    # Threads, not asyncio: the LLM/retrieval clients called here are
    # synchronous.
    with ThreadPoolExecutor(max_workers=2 * len(TRAINED_CATEGORIES) + 1) as pool:
        tumor_futures = {
            cat: pool.submit(_generate_tumor_explanation, cat, client)
            for cat in TRAINED_CATEGORIES
        }
        treatment_futures = {
            cat: pool.submit(retrieve_fn, _treatment_question(cat))
            for cat in TRAINED_CATEGORIES
        }
        brain_future = pool.submit(retrieve_fn, _brain_effects_question(locations))

        tumor_explanations = {cat: f.result() for cat, f in tumor_futures.items()}
        treatment_cited = {cat: f.result() for cat, f in treatment_futures.items()}
        brain_effects_cited = brain_future.result()

    tumor_analysis = {
        "tumor_detected": facts["tumor_detected"],
        "scan_type_assumption": _scan_type_assumption(),
        # Explicitly the whole-tumor region — see phase1_bridge.py's
        # extract_case_facts, which derives this from WT_cc specifically.
        # TC/ET/RC volumes are different numbers; a bare "tumor volume"
        # label invites a reader to assume whichever region they habitually
        # mean.
        "whole_tumor_volume_cc": facts["total_volume_cc"],
        # Two different metrics, two different splits - see
        # phase1_bridge.py's describe_case_for_prompt docstring. Keep both
        # labeled sizes attached so a renderer can't present either number
        # without saying what it was actually measured on.
        "model_voxel_dice": facts["voxel_dice"],
        "model_voxel_dice_n": facts["validation_set_size"],
        "model_holdout_dice": facts["holdout_dice"],
        "model_holdout_dice_n": facts["holdout_set_size"],
        "model_cohort_dice_range": facts["cohort_dice_range"],
        # One explanation per possible category, not one blended
        # paragraph - see _scan_type_assumption / module docstring for why.
        "explanation_by_category": tumor_explanations,
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
