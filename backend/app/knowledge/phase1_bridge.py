"""What Phase 1 imaging output is allowed to contribute to generated text
(spec §7). This is a safety boundary, not a formatting convenience — the
qualifiers below are the whole point of the module.

Phase 1's case.json (backend/jobs/<id>/case.json) is a Phase 1 concept;
Phase 3 never reads that file directly elsewhere. Every value that reaches
an LLM prompt from an imaging case MUST go through describe_case_for_prompt
so the constraints in spec §7's table are structural, not something every
call site has to remember to apply by hand.
"""

from __future__ import annotations

import re


def describe_case_for_prompt(case: dict) -> list[str]:
    """Turns one Phase 1 case.json into prompt-safe lines.

    - volume_cc: used freely — it's a direct measurement.
    - location: only ever emitted with "(approximate, coordinate-based)"
      attached, because it's a threshold on the region centroid's MNI y
      coordinate, not an atlas lookup — spec §7 is explicit that once an
      LLM writes it into a sentence unqualified, it reads as a radiological
      finding, "and that gap is exactly where this kind of system does
      harm."
    - mean_probability: NEVER included. A mean sigmoid output inside a
      predicted region is not a calibrated confidence that the finding is
      correct, and spec §7 says it "should never appear as '96%
      confidence'" in generated text.
    - holdout_select_dice: included only as model provenance ("this model's
      general performance on its holdout set"), never rephrased as
      per-case confidence.
    """
    lines: list[str] = []

    model = case.get("model", "unknown segmentation model")
    holdout_dice = case.get("holdout_select_dice")
    if holdout_dice is not None:
        lines.append(
            f"Segmentation model: {model}. Its holdout-set Dice score is "
            f"{holdout_dice:.3f} — a general property of the model's "
            f"validated performance, not a confidence figure for this "
            f"specific scan."
        )

    for key, region in (case.get("regions") or {}).items():
        parts = [f"Region {key}"]
        vol = region.get("volume_cc")
        if vol is not None:
            parts.append(f"measured volume {vol} cm3")
        loc = region.get("location")
        if loc:
            parts.append(f"approximate, coordinate-based location: {loc}")
        # mean_probability is intentionally omitted — see docstring.
        lines.append(", ".join(parts) + ".")

    return lines


def extract_case_facts(case: dict) -> dict:
    """Structured version of describe_case_for_prompt(), for callers that
    build JSON directly instead of a prompt string (app/knowledge/agents/
    clinical_agent.py). Same safety rules apply: mean_probability is never
    included, holdout_dice is model provenance only.
    """
    regions = case.get("regions") or {}
    total_volume = (case.get("derived") or {}).get("WT_cc")
    if total_volume is None and regions:
        total_volume = round(sum(r.get("volume_cc", 0) or 0 for r in regions.values()), 2)

    return {
        "tumor_detected": bool(regions),
        "model": case.get("model", "unknown segmentation model"),
        "holdout_dice": case.get("holdout_select_dice"),
        "total_volume_cc": total_volume,
        "regions": {
            key: {
                "volume_cc": r.get("volume_cc"),
                "location": r.get("location"),
                # tta_uncertainty is a raw self-disagreement score from
                # inference.py's test-time augmentation, not a calibrated
                # confidence - same non-fabrication rule as mean_probability
                # above, it just isn't banned outright because it's not
                # being presented as one.
                "tta_uncertainty": r.get("tta_uncertainty"),
            }
            for key, r in regions.items()
        },
        "locations": sorted({r["location"] for r in regions.values() if r.get("location")}),
    }


# A cheap, imperfect defense-in-depth check, not a substitute for the
# omission above: if generated text ever pairs a percentage with the word
# "confidence"/"certain"/"probability" near a region label, that's exactly
# the failure mode spec §7 warns about, and it should be caught before
# rendering rather than trusted to not happen.
_SUSPICIOUS_CONFIDENCE_RE = re.compile(
    r"\b\d{1,3}(\.\d+)?\s*%\s*(confiden\w*|certain\w*|probab\w*|likely)", re.IGNORECASE
)


def find_leaked_confidence_language(text: str) -> list[str]:
    """Returns the offending substrings, if any. Callers should treat a
    non-empty result as a bug to fix in the prompt/pipeline, not something
    to silently strip and ship.
    """
    return [m.group(0) for m in _SUSPICIOUS_CONFIDENCE_RE.finditer(text)]
