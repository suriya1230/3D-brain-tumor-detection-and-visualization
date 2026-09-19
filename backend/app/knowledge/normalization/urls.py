"""One canonical URL per article, everywhere a URL gets constructed or
received — otherwise the same PMC article shows up as two different
sources (old-style www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxx/ vs the
current pmc.ncbi.nlm.nih.gov/articles/PMCxxxx), and worse: web_search.py
hashes the URL into a chunk_id, so the same article under two spellings
becomes two different chunk_ids that will never dedupe against each
other. Applied at the edges (ingestion, web search results) so everything
downstream only ever sees the canonical form.
"""

from __future__ import annotations

import re

_OLD_PMC_RE = re.compile(
    r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)/?", re.IGNORECASE
)


def normalize_url(url: str) -> str:
    match = _OLD_PMC_RE.match(url)
    if match:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{match.group(1)}"
    return url
