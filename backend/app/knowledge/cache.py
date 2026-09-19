"""A small in-process TTL cache for the knowledge pipeline's expensive,
deterministic calls (query rewriting, cited answers) — a latency
optimisation, not a correctness mechanism, and not trusted to be one.

Bounded TTL, not infinite: this app can rebuild its corpus/index
(`cli.py build-corpus`) while the server keeps running, and an unbounded
cache would keep serving pre-rebuild answers forever with no way to know
they'd gone stale. Bounded SIZE too (LRU eviction) — this is a dict living
in one process's memory, not a real cache backend, so it's sized for a
single dev/demo process, not for scaling this beyond one worker.

Two calls in this pipeline are worth caching for a very concrete reason,
not caching-for-its-own-sake: Phase 4's brain_effects/treatment_information
questions are fully determined by (tumor category, location) — a small,
repeating vocabulary across cases — so the second case with "left frontal
lobe" reuses the first one's retrieval + LLM call instead of paying for it
again. Query rewriting is cached the same way for the same reason repeat
or near-repeat questions are common in a demo/testing session.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_TTL_S = 3600  # long enough that repeat questions in one session
# hit it; short enough that a corpus rebuild isn't stuck behind stale
# answers for more than an hour.
DEFAULT_MAX_ENTRIES = 512


class TTLCache:
    """One lock per instance, not one global lock - clinical_agent.py runs
    several of these concurrently (ThreadPoolExecutor) for genuinely
    independent questions, and knowledge_routes.py deliberately keeps
    separate cache instances so, e.g., a burst of rewrite-cache misses
    can't block cited-answer-cache lookups. The lock serializes access
    within one instance (a miss on key A briefly blocks a lookup on key B
    in the SAME cache) rather than allowing genuinely concurrent
    computation for different keys - a real tradeoff, accepted here
    because this is a small in-process cache, not a scalability target.
    """

    def __init__(self, ttl_s: float = DEFAULT_TTL_S, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl_s
        self._max = max_entries
        self._store: "OrderedDict[str, tuple[float, T]]" = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, key: str, compute: Callable[[], T]) -> T:
        with self._lock:
            now = time.monotonic()
            hit = self._store.get(key)
            if hit is not None:
                expires_at, value = hit
                if now < expires_at:
                    self._store.move_to_end(key)  # LRU touch
                    return value
                del self._store[key]  # expired - fall through and recompute

            value = compute()
            self._store[key] = (now + self._ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # evict least-recently-used
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
