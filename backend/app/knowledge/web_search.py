"""Live web search (Tavily), scoped to an authoritative-domain allowlist.

Not part of the curated Phase 3 corpus (spec §0/§1) — this hits the network
on every call, is never chunked or indexed, and its results are tagged
source_type="web" everywhere downstream (citations, the UI) so they're
never visually confused with the vetted NCI PDQ / PubMed / PMC OA corpus.

The domain restriction is the entire safety argument for this module:
unrestricted web search would pull in whatever ranks well — blogs, forums,
unmoderated health content — presented with the same citation format as
peer-reviewed literature. `include_domains` + `include_domains_mode:
"filter"` makes Tavily refuse to return anything outside the allowlist,
enforced server-side, not by us filtering results after the fact.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date, datetime

import requests

from app.knowledge.schema import Chunk

TAVILY_URL = "https://api.tavily.com/search"

# Government, intergovernmental, and major oncology-society sources only.
# Adding a domain here is a real decision — it means results from that site
# will be presented to users with the same trust framing as NCI PDQ.
ALLOWED_DOMAINS = [
    "cancer.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "medlineplus.gov",
    "cdc.gov",
    "who.int",
    "cochranelibrary.com",
    "asco.org",
    "astro.org",
]


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z").date()
    except ValueError:
        return None


def search_web(query: str, *, max_results: int = 3) -> list[Chunk]:
    """Returns live web results as Chunks (source_type="web"), or an empty
    list on any failure — a web-search outage should never take down a
    question that the curated corpus alone can still answer.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        resp = requests.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "include_domains": ALLOWED_DOMAINS,
                "include_domains_mode": "filter",
                "max_results": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    today = date.today()
    chunks: list[Chunk] = []
    for r in data.get("results", []):
        url = r.get("url", "")
        text = r.get("content", "")
        if not url or not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
                source_id=url,
                source_type="web",
                title=r.get("title") or url,
                section=None,
                text=text,
                url=url,
                published=_parse_date(r.get("published_date")),
                retrieved=today,
                licence="Live web result — not a stored/redistributed copy; "
                "see the source site for its own terms.",
                token_count=len(text.split()),
            )
        )
    return chunks
