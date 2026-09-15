"""NCI PDQ ingest — parses local Bookshelf HTML fixtures. Makes NO network
calls.

NCBI Bookshelf's copyright page (https://www.ncbi.nlm.nih.gov/books/about/
copyright/) states: "Crawlers and other automated processes may NOT be used
to systematically retrieve content from the Bookshelf web site." So unlike
pubmed.py, this module never fetches anything itself — it only parses HTML
files that were captured by a one-off, human-directed action and checked
into backend/data/knowledge/fixtures/nci_pdq/ (see manifest.json there for
what was fetched, when, and under what licence). Adding more PDQ summaries
means repeating that manual capture, not extending this module to crawl.

DOM shape this relies on (confirmed by fetching NBK65982 and NBK65893 on
2026-09-12): the body content lives in a single
<div class="... body-content ...">, and each section is a
<div id="CDR...N"><h2|h3|h4>Title</h2>...</div>, nested to mirror the
heading hierarchy. A section's own text is its direct <p>/<ul>/<ol> children;
deeper <div id="CDR..."> children are sub-sections, recursed into
separately. This *is* the structure the spec asks for ("keep the section
hierarchy") — the source is HTML, but the parse is DOM-aware, not a
sliding-window scrape.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import warnings

from bs4 import XMLParsedAsHTMLWarning, BeautifulSoup, Tag

# The Bookshelf pages declare an XHTML doctype, which trips bs4's XML
# sniffing heuristic even though this is HTML we intentionally parse with
# an HTML parser.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from app.knowledge.ingest.common import RawSection

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "fixtures" / "nci_pdq"

# Sections that are administrative/citation scaffolding, not clinical prose.
# Filtered by exact (case-insensitive) heading text.
_SKIP_HEADINGS = {
    "references",
    "about this pdq summary",
    "latest updates to this summary",
}


def _section_own_text(div: Tag) -> str:
    """Direct-child prose of one section div, excluding nested section divs
    (those are separate sections, walked separately) and the heading itself.
    """
    parts: list[str] = []
    for child in div.find_all(recursive=False):
        if child.name in ("h2", "h3", "h4", "h5"):
            continue
        if child.name == "div" and (child.get("id") or "").startswith("CDR"):
            continue  # nested sub-section, handled by the recursive walk
        text = child.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _walk(
    div: Tag,
    heading_path: list[str],
    *,
    source_id_prefix: str,
) -> list[tuple[str, str, str]]:
    """Returns (anchor_id, heading_path_str, text) for this section and all
    descendant sections, depth-first. Skips sections under _SKIP_HEADINGS.
    """
    heading = div.find(["h2", "h3", "h4", "h5"], recursive=False)
    title = heading.get_text(" ", strip=True) if heading else None

    if title and title.strip().lower() in _SKIP_HEADINGS:
        return []

    path = heading_path + [title] if title else heading_path
    anchor = div.get("id") or ""

    records: list[tuple[str, str, str]] = []
    own_text = _section_own_text(div)
    if own_text:
        records.append((anchor, " > ".join(path), own_text))

    for child in div.find_all("div", recursive=False):
        child_id = child.get("id") or ""
        if child_id.startswith("CDR"):
            records.extend(_walk(child, path, source_id_prefix=source_id_prefix))

    return records


def parse_pdq_html(
    html: str,
    *,
    nbk_id: str,
    title: str,
    url: str,
    published: date | None,
    retrieved: date,
    licence: str,
) -> list[RawSection]:
    soup = BeautifulSoup(html, "lxml")
    container = soup.find("div", class_=re.compile(r"\bbody-content\b"))
    if container is None:
        raise ValueError(f"{nbk_id}: could not find the body-content container — page structure may have changed")

    sections: list[RawSection] = []
    for child in container.find_all("div", recursive=False):
        child_id = child.get("id") or ""
        if not child_id.startswith("CDR"):
            continue  # preamble/disclaimer text before the first heading
        for anchor, section_path, text in _walk(child, [], source_id_prefix=nbk_id):
            sections.append(
                RawSection(
                    source_id=f"{nbk_id}#{anchor}",
                    source_type="nci_pdq",
                    title=title,
                    section=section_path,
                    text=text,
                    url=f"{url}#{anchor}" if anchor else url,
                    published=published,
                    retrieved=retrieved,
                    licence=licence,
                )
            )
    return sections


def load_all_pdq_fixtures() -> list[RawSection]:
    """Parses every document listed in fixtures/nci_pdq/manifest.json."""
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))
    licence = manifest["licence"]

    all_sections: list[RawSection] = []
    for doc in manifest["documents"]:
        html = (FIXTURES_DIR / doc["html_file"]).read_text(encoding="utf-8")
        all_sections.extend(
            parse_pdq_html(
                html,
                nbk_id=doc["nbk_id"],
                title=doc["title"],
                url=doc["url"],
                published=datetime.strptime(doc["published"], "%Y-%m-%d").date(),
                retrieved=datetime.strptime(doc["retrieved"], "%Y-%m-%d").date(),
                licence=licence,
            )
        )
    return all_sections


if __name__ == "__main__":
    secs = load_all_pdq_fixtures()
    print(f"{len(secs)} sections parsed from {FIXTURES_DIR / 'manifest.json'}")
    for s in secs[:5]:
        print(f"- [{s.source_id}] {s.section!r} ({len(s.text)} chars)")
