"""Step 6 eval: citation precision, faithfulness, abstention rate (spec §8).

IMPORTANT: this runs against StubClient by default. StubClient is a
deterministic term-overlap heuristic, not a language model — it exists to
prove the pipeline (prompt building, contract enforcement, metric
computation) is wired correctly end-to-end. The numbers below measure the
STUB's behavior, not real answer quality, and should not be quoted as
Phase 3's citation precision or faithfulness. Set GROQ_API_KEY or
ANTHROPIC_API_KEY and pass "groq" or "anthropic" instead of "stub" to get
numbers that mean something.

    python -m app.knowledge.eval.run_answer_eval [stub|groq|anthropic]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from app.knowledge.answering import AnthropicClient, GroqClient, StubClient, answer_question
from app.knowledge.index.dense import MiniLMEmbedder
from app.knowledge.index.fuse import hybrid_search, load_chunk_lookup
from app.knowledge.eval.metrics import abstained, citation_precision, faithfulness

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


def main(client_name: str) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    chunk_lookup = load_chunk_lookup()
    embedder = MiniLMEmbedder()  # spec §5 measured winner on this fixture-scale corpus
    clients = {"anthropic": AnthropicClient, "groq": GroqClient, "stub": StubClient}
    client = clients[client_name]()

    precisions, faithfulnesses = [], []
    unanswerable_abstained = 0
    unanswerable_total = 0

    for q in questions:
        passages = hybrid_search(q["question"], embedder, chunk_lookup, top_k_each=20, top_k_fused=8)
        answer = answer_question(q["question"], passages, client)

        if q["category"] == "unanswerable":
            unanswerable_total += 1
            if abstained(answer, q["id"]):
                unanswerable_abstained += 1
            continue

        p = citation_precision(answer, chunk_lookup)
        f = faithfulness(answer, chunk_lookup)
        if p == p:  # not NaN
            precisions.append(p)
        if f == f:
            faithfulnesses.append(f)

    print(f"client: {client.name}")
    print(f"citation precision (mean over answerable questions): {statistics.mean(precisions):.3f}")
    print(f"faithfulness (mean over answerable questions): {statistics.mean(faithfulnesses):.3f}")
    print(f"abstention rate on unanswerable: {unanswerable_abstained}/{unanswerable_total}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stub")
