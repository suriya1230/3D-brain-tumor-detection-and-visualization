"""Controlled-vocabulary lookup for CNS tumour types (spec §3).

Without this, "GBM", "glioblastoma" and "glioblastoma multiforme" are three
unrelated strings and filtered retrieval silently misses most of the corpus.
Everything in the vocabulary itself lives in tumor_types.yaml, reviewable and
versioned independently of this code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_VOCAB_PATH = Path(__file__).resolve().parent / "tumor_types.yaml"


@dataclass(frozen=True)
class TumourEntity:
    id: str
    who_name: str
    category: str
    synonyms: tuple[str, ...]


def _norm(s: str) -> str:
    # Lowercase, collapse whitespace, drop everything but alphanumerics/space
    # so "IDH-wildtype", "IDH wildtype" and "idh  wildtype" all match.
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _vocab() -> tuple[dict[str, TumourEntity], dict[str, str]]:
    """Returns (entities by id, normalized-alias -> id lookup table)."""
    raw = yaml.safe_load(_VOCAB_PATH.read_text(encoding="utf-8"))
    entities: dict[str, TumourEntity] = {}
    alias_to_id: dict[str, str] = {}

    for e in raw["entities"]:
        ent = TumourEntity(
            id=e["id"],
            who_name=e["who_name"],
            category=e["category"],
            synonyms=tuple(e.get("synonyms") or ()),
        )
        entities[ent.id] = ent
        alias_to_id[_norm(ent.who_name)] = ent.id
        for syn in ent.synonyms:
            alias_to_id[_norm(syn)] = ent.id

    return entities, alias_to_id


def all_entities() -> list[TumourEntity]:
    return list(_vocab()[0].values())


def get_entity(canonical_id: str) -> TumourEntity | None:
    return _vocab()[0].get(canonical_id)


def normalize_tumour_type(raw: str) -> str | None:
    """Exact (post-normalization) match against WHO names + synonyms.

    Returns the canonical id, or None if nothing matches. Intentionally not
    fuzzy: a wrong silent match is worse than a documented miss you can add
    a synonym for.
    """
    entities, alias_to_id = _vocab()
    return alias_to_id.get(_norm(raw))


def find_tumour_types_in_text(text: str) -> list[str]:
    """Longest-match scan for known tumour-type mentions inside free text.

    Used at ingest time to tag a chunk's `tumour_types`. Longest alias first
    so "glioblastoma, idh-wildtype" matches before the bare "glioblastoma"
    substring inside it would.
    """
    entities, alias_to_id = _vocab()
    norm_text = _norm(text)
    found: set[str] = set()

    for alias in sorted(alias_to_id, key=len, reverse=True):
        if alias and f" {alias} " in f" {norm_text} ":
            found.add(alias_to_id[alias])

    return sorted(found)
