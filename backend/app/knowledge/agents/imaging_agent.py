"""Imaging agent (spec §10): answers questions about the scan itself
directly from Phase 1's measurements, with no retrieval and no LLM call.
This is what makes the planner's routing decision actually save something —
a question like "how big is this tumor" doesn't need 8 passages and a
model call when the answer is already a known, measured fact.

Always returns the full measurement set for the case, not just the part
that seems relevant to the question — picking out "the" relevant region
from a free-text question is exactly the kind of interpretation this
module is built to avoid. Every line goes through phase1_bridge, so the
same qualifiers (approximate location, no mean_probability, no per-case
confidence) apply here as everywhere else in the system.
"""

from __future__ import annotations

from app.knowledge.phase1_bridge import describe_case_for_prompt


def answer_from_imaging(case: dict) -> dict:
    return {
        "claims": [],
        "unsupported": [],
        "sources": [],
        "measurements": describe_case_for_prompt(case),
        "routed_to": "imaging_agent",
    }
