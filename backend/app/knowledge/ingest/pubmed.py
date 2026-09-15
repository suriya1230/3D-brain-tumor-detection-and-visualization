"""PubMed metadata + PMC OA full-text ingest (spec §1, §9 step 2).

Unlike nci_pdq.py, this module DOES make network calls: NLM's own PMC terms
name E-utilities, the BioC API, and OAI-PMH as the sanctioned automated
retrieval paths for the OA subset (spec §1's "3 req/s without a key, 10
with one" is this module's rate budget).

Two-tier storage per article, matching spec §1's field list:
  - Abstract + bibliographic metadata: always stored (PubmedRecord).
  - Full text: only stored when the article resolves to a PMC ID AND the
    BioC API actually returns it — that endpoint only serves the PMC Open
    Access Subset, so a successful response is itself the licence check.
    The returned `license` infon (e.g. "CC BY") is recorded verbatim.

No `email` param is sent on E-utilities requests. NCBI's usage guidelines
suggest one so they can reach a caller about a problem; this app does not
forward a user's personal email to a third-party service for that purpose.
`tool` is sent, which is what NCBI uses to attribute traffic.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

import requests
from pydantic import BaseModel, Field

from app.knowledge.ingest.common import RawSection

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
TOOL_NAME = "neuroevidence-phase3"

# BioC section_type codes that are administrative scaffolding, not prose —
# mirrors the _SKIP_HEADINGS filter in nci_pdq.py for the same reason.
_SKIP_SECTION_TYPES = {
    "REF", "COMP_INT", "AUTH_CONT", "ACK_FUND", "TITLE", "SUPPL", "ABSTRACT",
    "FIG", "TABLE",
}
# ABSTRACT is skipped from BioC output specifically because
# pubmed_record_to_sections() already emits a clean copy of it from the
# PubMed metadata record — keeping both would duplicate that text in the
# index under two source_types.
# FIG/TABLE are captions without their figure/table data — exactly spec
# §4's "a half-table retrieves as gibberish" case.


class _RateLimiter:
    """Sleeps as needed to keep calls at or under `rate` requests/second."""

    def __init__(self, rate: float):
        self._min_interval = 1.0 / rate
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


_API_KEY = os.getenv("NCBI_API_KEY")
_LIMITER = _RateLimiter(10.0 if _API_KEY else 3.0)


def _get(url: str, params: dict) -> requests.Response:
    params = {**params, "tool": TOOL_NAME}
    if _API_KEY:
        params["api_key"] = _API_KEY
    _LIMITER.wait()
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp


class PubmedRecord(BaseModel):
    pmid: str
    pmcid: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    journal: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    licence: str
    has_full_text: bool = False
    retrieved: date


def _text(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    return "".join(el.itertext()).strip() or None


def _parse_pubmed_article(art: ET.Element, retrieved: date) -> PubmedRecord:
    pmid = art.findtext("./MedlineCitation/PMID") or ""
    article = art.find("./MedlineCitation/Article")

    title = _text(article.find("ArticleTitle")) if article is not None else None

    abstract_parts = [
        _text(t) for t in (article.findall("Abstract/AbstractText") if article is not None else [])
    ]
    abstract = "\n\n".join(p for p in abstract_parts if p) or None

    journal = _text(article.find("Journal/Title")) if article is not None else None

    year = None
    if article is not None:
        year_text = article.findtext("Journal/JournalIssue/PubDate/Year")
        if not year_text:
            medline_date = article.findtext("Journal/JournalIssue/PubDate/MedlineDate") or ""
            year_text = medline_date[:4] if medline_date[:4].isdigit() else None
        if year_text:
            year = int(year_text)

    authors = []
    if article is not None:
        for a in article.findall("AuthorList/Author"):
            last = a.findtext("LastName")
            fore = a.findtext("ForeName")
            collective = a.findtext("CollectiveName")
            if last and fore:
                authors.append(f"{fore} {last}")
            elif last:
                authors.append(last)
            elif collective:
                authors.append(collective)

    pub_types = [
        t.text for t in (article.findall("PublicationTypeList/PublicationType") if article is not None else [])
        if t.text
    ]

    mesh_terms = [
        d.text
        for d in art.findall("./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName")
        if d.text
    ]

    # Own article ids only — NOT the ones nested under ReferenceList, which
    # are the ids of *cited* papers.
    own_ids = art.findall("./PubmedData/ArticleIdList/ArticleId")
    pmcid = next((i.text for i in own_ids if i.get("IdType") == "pmc"), None)
    doi = next((i.text for i in own_ids if i.get("IdType") == "doi"), None)

    return PubmedRecord(
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=title or "(untitled)",
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        publication_types=pub_types,
        mesh_terms=mesh_terms,
        licence="Abstract © publisher; indexed via NLM/PubMed. Full text not stored "
        "unless separately fetched from the PMC Open Access subset.",
        has_full_text=False,
        retrieved=retrieved,
    )


def fetch_pubmed_metadata(pmids: list[str], *, batch_size: int = 150) -> list[PubmedRecord]:
    """efetch db=pubmed in batches, parsed into PubmedRecord (spec §1 fields)."""
    today = date.today()
    records: list[PubmedRecord] = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        resp = _get(
            f"{EUTILS_BASE}/efetch.fcgi",
            {"db": "pubmed", "id": ",".join(batch), "rettype": "xml", "retmode": "xml"},
        )
        root = ET.fromstring(resp.content)
        for art in root.findall("PubmedArticle"):
            records.append(_parse_pubmed_article(art, today))

    return records


@dataclass
class BiocSection:
    section_type: str
    title: str | None
    text: str


@dataclass
class BiocDocument:
    licence: str
    sections: list[BiocSection]


def fetch_pmc_oa_fulltext(pmcid: str) -> BiocDocument | None:
    """BioC full text for one PMC OA article, grouped into sections.

    Returns None if the article isn't in the OA subset (BioC 404s) or the
    response has no usable passages — treated as "abstract only" upstream.
    """
    try:
        resp = _get(f"{BIOC_BASE}/BioC_xml/PMC{pmcid.removeprefix('PMC')}/unicode", {})
        root = ET.fromstring(resp.content)
    except (requests.HTTPError, ET.ParseError):
        # Not in the OA subset, or a transient non-XML error response —
        # either way, this article falls back to abstract-only.
        return None

    doc = root.find("document")
    if doc is None:
        return None

    licence = None
    sections: list[BiocSection] = []
    current_type: str | None = None
    current_title: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_type and current_parts and current_type not in _SKIP_SECTION_TYPES:
            sections.append(
                BiocSection(section_type=current_type, title=current_title, text="\n\n".join(current_parts))
            )

    for passage in doc.findall("passage"):
        infons = {i.get("key"): i.text for i in passage.findall("infon")}
        if licence is None and infons.get("license") and len(infons["license"]) < 40:
            # The short-form license infon (e.g. "CC BY"); a longer,
            # human-readable restatement is also present under the same
            # key and would overwrite this if we didn't guard on length.
            licence = infons["license"]

        sec_type = infons.get("section_type") or "OTHER"
        text = _text(passage.find("text")) or ""
        is_title_passage = (infons.get("type") or "").endswith("title") or (infons.get("type") or "").endswith("title_1")

        if sec_type != current_type:
            flush()
            current_type = sec_type
            current_title = text if is_title_passage else None
            current_parts = [] if is_title_passage else ([text] if text else [])
        else:
            if is_title_passage and current_title is None:
                current_title = text
            elif text:
                current_parts.append(text)
    flush()

    if not sections:
        return None
    return BiocDocument(licence=licence or "CC (exact variant unknown — see article)", sections=sections)


def pubmed_record_to_sections(record: PubmedRecord, bioc: BiocDocument | None) -> list[RawSection]:
    url = f"https://pubmed.ncbi.nlm.nih.gov/{record.pmid}/"
    sections: list[RawSection] = []

    if record.abstract:
        sections.append(
            RawSection(
                source_id=record.pmid,
                source_type="pubmed",
                title=record.title,
                section="Abstract",
                text=record.abstract,
                url=url,
                published=date(record.year, 1, 1) if record.year else None,
                retrieved=record.retrieved,
                licence=record.licence,
            )
        )

    if bioc:
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{record.pmcid.removeprefix('PMC')}/"
        for idx, sec in enumerate(bioc.sections):
            label = sec.title or sec.section_type
            sections.append(
                RawSection(
                    source_id=f"{record.pmcid}#{idx}",
                    source_type="pmc_oa",
                    title=record.title,
                    section=label,
                    text=sec.text,
                    url=pmc_url,
                    published=date(record.year, 1, 1) if record.year else None,
                    retrieved=record.retrieved,
                    licence=bioc.licence,
                )
            )

    return sections


def ingest_pubmed(pmids: list[str]) -> tuple[list[PubmedRecord], list[RawSection]]:
    """Full pipeline: metadata for every PMID, full text for the ones that
    resolve to a PMC OA article. This is the live ingester — the same code
    runs at fixture scale (a short PMID list) and at full-ingestion scale
    (a long one); only the input list changes.
    """
    records = fetch_pubmed_metadata(pmids)
    all_sections: list[RawSection] = []

    for record in records:
        bioc = fetch_pmc_oa_fulltext(record.pmcid) if record.pmcid else None
        if bioc:
            record.has_full_text = True
            record.licence = bioc.licence
        all_sections.extend(pubmed_record_to_sections(record, bioc))

    return records, all_sections


if __name__ == "__main__":
    import json
    from pathlib import Path

    FIXTURE_PMIDS = [
        "41057317", "40255400", "39911397", "39387957", "38396785",
        "37857607", "33396284", "33261657", "29938005",  # PMC OA full text
        "38724526", "40939590", "32478924", "34504304",  # abstract-only
        "28281173", "35135556",  # PMC OA full text
        # pseudoprogression imaging + recurrence management (2026-09-12)
        "26275367", "23151413", "41134378", "25438810", "40528226",
        "39587462", "39759563", "21655151", "40991008",  # PMC OA full text
        "32408220",  # abstract-only
    ]

    recs, secs = ingest_pubmed(FIXTURE_PMIDS)

    out_dir = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "fixtures" / "pubmed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(r.model_dump_json() + "\n")

    print(f"{len(recs)} PubMed records, {sum(r.has_full_text for r in recs)} with PMC OA full text")
    print(f"{len(secs)} raw sections total")
    for s in secs[:3]:
        print(f"- [{s.source_type}] {s.source_id} / {s.section!r} ({len(s.text)} chars)")
