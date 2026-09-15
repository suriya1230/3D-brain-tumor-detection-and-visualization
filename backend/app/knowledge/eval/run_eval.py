"""Runs the 40-question eval set (spec §8, build order step 5: retrieval
only — no LLM yet). Measures Recall@20 for each of the two embedders
(spec §5: "evaluate at least two") so the choice between them is measured,
not assumed.

    python -m app.knowledge.eval.run_eval
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from app.knowledge.index.dense import Embedder, MedCPTEmbedder, MiniLMEmbedder
from app.knowledge.index.fuse import hybrid_search, load_chunk_lookup
from app.knowledge.eval.metrics import recall_at_k

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def run_recall_eval(embedder: Embedder, questions: list[dict], chunk_lookup: dict) -> dict:
    per_question: dict[str, float] = {}
    by_category: dict[str, list[float]] = {}

    for q in questions:
        if not q["expected_chunk_ids"]:
            continue  # unanswerable — no recall target
        retrieved = hybrid_search(
            q["question"], embedder, chunk_lookup, top_k_each=20, top_k_fused=20
        )
        retrieved_ids = [c.chunk_id for c in retrieved]
        score = recall_at_k(q["expected_chunk_ids"], retrieved_ids, k=20)
        per_question[q["id"]] = score
        by_category.setdefault(q["category"], []).append(score)

    overall = statistics.mean(per_question.values()) if per_question else float("nan")
    category_means = {cat: statistics.mean(scores) for cat, scores in by_category.items()}

    return {
        "embedder": embedder.name,
        "overall_recall_at_20": overall,
        "by_category": category_means,
        "misses": [qid for qid, score in per_question.items() if score == 0.0],
    }


if __name__ == "__main__":
    questions = load_questions()
    chunk_lookup = load_chunk_lookup()

    for embedder_cls in (MedCPTEmbedder, MiniLMEmbedder):
        embedder = embedder_cls()
        result = run_recall_eval(embedder, questions, chunk_lookup)
        n = sum(1 for q in questions if q["expected_chunk_ids"])
        print(f"\n=== {result['embedder']} ===")
        print(f"Recall@20 (overall, n={n}): {result['overall_recall_at_20']:.3f}")
        for cat, mean in result["by_category"].items():
            print(f"  {cat}: {mean:.3f}")
        if result["misses"]:
            print(f"  misses: {', '.join(result['misses'])}")
