"""Planner (spec §10): decides which source(s) a question actually needs,
instead of always fanning out to the full corpus. Cheap and rule-based on
purpose — spec calls this out as something to build first specifically
*because* it's cheap, and most questions only need one source, not seven.

This is NOT a semantic router. It's a fast, explainable, deterministic set
of checks over the question text (and the case, if there is one) that
picks a RoutingDecision before any retrieval happens. If it's wrong, the
fallback (search everything) is exactly today's pre-planner behavior — so
a routing mistake costs recall, not correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.knowledge.normalization.tumor_types import find_tumour_types_in_text
from app.knowledge.schema import SourceType

Route = Literal["imaging_agent", "corpus"]


@dataclass
class RoutingDecision:
    route: Route
    source_types: list[SourceType] | None  # None = no filter, search everything
    tumour_type: str | None
    reasoning: str  # always populated — this is a rule-based router, not a black box


# Phrases that mean "this scan"/"this patient's own measurements", not "the
# literature in general". Deliberately conservative: a false negative here
# just means the question falls through to normal corpus search, which is
# always a safe default.
_IMAGING_PATTERNS = re.compile(
    r"\b(this (scan|case|patient|tumor|tumour)|"
    r"my (scan|case|patient|tumor|tumour)|"
    r"the resection cavity|how (big|large) is|"
    r"what('?s| is) the (volume|size|location) of|"
    r"where is (it|the tumor|the tumour))\b",
    re.IGNORECASE,
)

_NCI_HINT = re.compile(
    r"\b(nci|pdq|standard of care|treatment option|guideline|recommended treatment)\b",
    re.IGNORECASE,
)
_RESEARCH_HINT = re.compile(
    r"\b(study|trial|research|paper|mechanism|pathway|molecular|marker|biomarker|gene)\b",
    re.IGNORECASE,
)


def route_question(question: str, *, has_case: bool) -> RoutingDecision:
    tumour_types = find_tumour_types_in_text(question)
    tumour_type = tumour_types[0] if tumour_types else None

    if has_case and _IMAGING_PATTERNS.search(question):
        return RoutingDecision(
            route="imaging_agent",
            source_types=None,
            tumour_type=tumour_type,
            reasoning="question refers to this scan/case directly — answered "
            "from measurements, no corpus search needed",
        )

    nci_hit = bool(_NCI_HINT.search(question))
    research_hit = bool(_RESEARCH_HINT.search(question))

    if nci_hit and not research_hit:
        return RoutingDecision(
            route="corpus",
            source_types=["nci_pdq"],
            tumour_type=tumour_type,
            reasoning="question asks about standard-of-care/guidelines — NCI PDQ only",
        )
    if research_hit and not nci_hit:
        return RoutingDecision(
            route="corpus",
            source_types=["pubmed", "pmc_oa"],
            tumour_type=tumour_type,
            reasoning="question asks about mechanisms/research — PubMed/PMC only",
        )

    return RoutingDecision(
        route="corpus",
        source_types=None,
        tumour_type=tumour_type,
        reasoning="no strong signal either way — searching the full corpus",
    )
