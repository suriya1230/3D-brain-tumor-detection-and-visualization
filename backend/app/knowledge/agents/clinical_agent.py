"""Phase 4 Clinical Analysis Agent — no retrieval.

Deliberately NOT built on the Phase 3 RAG pipeline (hybrid_search +
citation-gated answer_question). That pipeline answers "what does the
evidence corpus say" and refuses when the corpus doesn't cover something —
exactly right for open-ended literature questions, exactly wrong for "what
is this tumor" where the answer should come directly from this case's own
segmentation, explained in general clinical-education terms.

Two questions get NO model call at all, ever — growth duration and
survival/prognosis. Both are hardcoded, deterministic, and identical for
every case, because no single MRI can answer either one, and that's not a
retrieval gap to paper over — it's a structural fact about what a
single-timepoint segmentation can know. See ASSESS_GROWTH / ASSESS_PROGNOSIS.

The other three sections (what is it, how might it affect the brain, what
treatments are typically used) get one LLM call, under a system prompt that
constrains by TOPIC and MANDATORY HEDGING LANGUAGE rather than by citation —
a different safety mechanism than Phase 3's, appropriate to a different
kind of question (general medical education vs. literature lookup).
"""

from __future__ import annotations

import json
import logging
import re

from app.knowledge.answering import LLMClient, _strip_code_fence  # reuse, don't duplicate
from app.knowledge.phase1_bridge import extract_case_facts

log = logging.getLogger("neuroevidence.knowledge")

SYSTEM_PROMPT = """\
You are a clinical education assistant explaining an AI imaging \
segmentation result to a clinician. You are NOT diagnosing, NOT citing \
specific research papers, and NOT making patient-specific numeric \
predictions. You explain general, well-established medical concepts only.

Rules:
1. You may use general medical knowledge to explain: what the predicted \
tumor category generally means, what functions are typically associated \
with the anatomical location given, and what treatment categories are \
typically used for this general tumor type.
2. NEVER state or imply: a specific diagnosis, a WHO grade, a survival \
percentage, a life expectancy, or how long the tumor has existed. This \
system has no data that could support any of those — a single scan cannot \
establish them, and you must not fill the gap with a plausible-sounding \
number.
3. In brain_effects, distinguish "possible effects" (what the location is \
generally associated with) from "confirmed symptoms" (something you have \
no data on for this patient) — never state the patient has any symptom.
4. Output ONLY a JSON object of this exact shape, nothing else:
{"tumor_analysis": {"explanation": "..."},
 "brain_effects": {"explanation": "...", "possible_effects": ["...", "..."]},
 "treatment_information": {"explanation": "...", "typical_categories": ["...", "..."]}}
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
    "Tumor category is a segmentation-model prediction, not a confirmed "
    "diagnosis or WHO grade; that requires pathology and molecular testing.",
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


def _fallback_llm_sections(category: str) -> dict:
    """Used only if the LLM call fails outright — still honest, still
    structured, just generic. Never silently empty."""
    return {
        "tumor_analysis": {
            "explanation": f"The segmentation model's predicted category is "
            f"{category}. A general explanation could not be generated "
            f"right now (answer service unavailable) — the category above "
            f"is a segmentation-model prediction, not a confirmed diagnosis "
            f"or WHO grade, which require pathology and molecular testing."
        },
        "brain_effects": {
            "explanation": "A location-specific explanation could not be "
            "generated right now (answer service unavailable).",
            "possible_effects": [],
        },
        "treatment_information": {
            "explanation": "Treatment category information could not be "
            "generated right now (answer service unavailable).",
            "typical_categories": [],
        },
    }


def _generate_llm_sections(category: str, locations: list[str], client: LLMClient) -> dict:
    location_text = " and ".join(locations) if locations else "an unspecified region"
    user_prompt = (
        f"Predicted tumor category: {category}\n"
        f"Tumor location(s): {location_text}\n\n"
        f"Produce the three sections described in your instructions for "
        f"this category and location."
    )

    try:
        raw = client.complete(SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(_strip_code_fence(raw))
    except Exception:
        log.exception("clinical agent LLM call failed")
        return _fallback_llm_sections(category)

    # Minimal structural validation — code enforcement, not trust, same
    # principle as answer_question()'s contract checking in Phase 3.
    required = {"tumor_analysis", "brain_effects", "treatment_information"}
    if not required.issubset(parsed.keys()):
        log.warning("clinical agent returned incomplete JSON: %s", list(parsed.keys()))
        return _fallback_llm_sections(category)

    return parsed


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


def run_default_analysis(case: dict, client: LLMClient) -> dict:
    facts = extract_case_facts(case)
    category = _category_from_model(facts["model"])

    llm_sections = _generate_llm_sections(category, facts["locations"], client)
    confidence, confidence_notes = _segmentation_confidence(facts)

    tumor_analysis = {
        "tumor_detected": facts["tumor_detected"],
        "predicted_category": category,
        "tumor_volume_cc": facts["total_volume_cc"],
        "model_holdout_dice": facts["holdout_dice"],
        **llm_sections.get("tumor_analysis", {}),
    }

    brain_effects = {
        "locations": facts["locations"],
        **llm_sections.get("brain_effects", {}),
    }

    return {
        "tumor_analysis": tumor_analysis,
        "brain_effects": brain_effects,
        "growth_assessment": ASSESS_GROWTH,
        "prognosis_assessment": ASSESS_PROGNOSIS,
        "treatment_information": llm_sections.get("treatment_information", {}),
        "segmentation_confidence": confidence,
        "warnings": STANDING_WARNINGS + confidence_notes,
        "sources": [],
    }
