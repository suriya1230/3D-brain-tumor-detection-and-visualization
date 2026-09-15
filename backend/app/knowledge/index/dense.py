"""Dense retrieval over chunks (spec §5), via a biomedical embedding model
and a local Chroma vector store.

Spec §5 says evaluate at least two embedding models — the gap between a
general-purpose model and a biomedical one is supposed to be large on this
kind of text, and that claim should be checked, not assumed. Two embedders
are wired up here:

  - MedCPTEmbedder: ncbi/MedCPT-{Query,Article}-Encoder, a dual encoder
    trained by NCBI specifically for PubMed-scale biomedical retrieval —
    the purpose-built option for this exact corpus.
  - MiniLMEmbedder: sentence-transformers/all-MiniLM-L6-v2, a strong
    general-purpose baseline, kept only as the comparison point the spec
    asks for.

Each gets its own Chroma collection so eval (step 5) can measure both and
the loser can be dropped rather than guessed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import chromadb
import torch
from transformers import AutoModel, AutoTokenizer

from app.knowledge.schema import Chunk

PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "chroma"


class Embedder(Protocol):
    name: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class MedCPTEmbedder:
    name = "medcpt"

    def __init__(self) -> None:
        self._q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self._q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()
        self._d_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self._d_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").eval()

    @torch.no_grad()
    def _embed(self, tokenizer, model, texts: list[str], max_length: int) -> list[list[float]]:
        out: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True, return_tensors="pt", max_length=max_length
            )
            embeds = model(**enc).last_hidden_state[:, 0, :]  # [CLS]
            out.extend(embeds.cpu().numpy().tolist())
        return out

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(self._q_tok, self._q_model, texts, max_length=64)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # MedCPT's article encoder expects "title [SEP] text"; we don't
        # always have a clean single title per chunk section, so text alone
        # is used — a known small deviation from the paper's exact recipe.
        return self._embed(self._d_tok, self._d_model, texts, max_length=512)


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


def _client() -> chromadb.ClientAPI:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


def build_dense_index(chunks: list[Chunk], embedder: Embedder) -> None:
    client = _client()
    collection_name = f"chunks_{embedder.name}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # didn't exist yet
    collection = client.create_collection(collection_name)

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_type": c.source_type,
                "title": c.title,
                "section": c.section or "",
                "tumour_types": ",".join(c.tumour_types),
            }
            for c in chunks
        ],
    )


def search_dense(
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 20,
    source_types: list[str] | None = None,
    tumour_type: str | None = None,
) -> list[tuple[str, float]]:
    """Returns [(chunk_id, distance), ...], best (smallest distance) first."""
    client = _client()
    collection_name = f"chunks_{embedder.name}"
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    where = None
    clauses = []
    if source_types:
        clauses.append({"source_type": {"$in": source_types}})
    if tumour_type:
        # Chroma has no substring/array-contains match on a comma-joined
        # string; tumour_type filtering is applied as a post-filter by the
        # caller (fuse.py) instead of pushed down here.
        pass
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    [query_embedding] = embedder.embed_queries([query])
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

    ids = result["ids"][0]
    distances = result["distances"][0]
    return list(zip(ids, distances))
