"""Answer generation under the spec §6 contract.

The LLM never sees a bare question — only the question plus retrieved
passages (spec §6: "It does not receive the question without passages").
Its job is to produce claims, each citing the chunk_id(s) that support it,
and to say what it can't answer rather than filling the gap.

The prompt *asks* for that; it is not trusted to deliver it. Every claim
coming back is re-checked here: an empty chunk_ids list, or a chunk_id that
isn't one of the passages actually shown to the model, moves that claim's
text to `unsupported` and it never reaches `claims`. That's spec §6's "a
prompt instruction to cite is a request; a schema check is a guarantee" —
enforced in this module, not left to the model's compliance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Protocol

from openai import RateLimitError

from app.knowledge.schema import Answer, Chunk, Citation, Claim

log = logging.getLogger("neuroevidence.knowledge")

MAX_PASSAGES = 8  # spec §5: "pass 6-8 to the LLM"
MAX_PASSAGE_CHARS = 2000  # keeps the prompt bounded even for long chunks

SYSTEM_PROMPT = """\
You are a clinical evidence retrieval assistant for NeuroEvidence. You are \
not a diagnostic tool and you do not supply medical facts from your own \
knowledge — you only summarize the passages you are given below.

Rules:
1. Every factual claim you make MUST be grounded in one or more of the \
numbered passages below (each starts with a bracketed number like [1], \
[2]), and MUST cite that number — never leave "sources" empty for a claim \
you're actually making.
2. If the question asks something the passages do not address, do not \
guess or use outside knowledge — put that part of the question, in your \
own words, into "unsupported" instead.
3. Output ONLY a JSON object of this exact shape, nothing else. "sources" \
holds the passage NUMBERS (integers, not the chunk_id text) that support \
each claim:
{"claims": [{"text": "...", "sources": [1, 3]}], "unsupported": ["..."]}

Example, given passages numbered [1] and [2]: if [1] says a drug helps and \
[2] is about something unrelated, respond with:
{"claims": [{"text": "The drug helps.", "sources": [1]}], "unsupported": []}
"""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """The real client. Needs ANTHROPIC_API_KEY set — nothing in this
    module runs a network call unless this class is explicitly selected.
    """

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class GroqClient:
    """Free-tier-friendly alternative to AnthropicClient. Needs
    GROQ_API_KEY. Groq's chat endpoint is OpenAI-compatible, so this uses
    the `openai` SDK pointed at Groq's base URL rather than a separate
    `groq` package.
    """

    name = "groq"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        # llama-3.3-70b-versatile is no longer served on this account's Groq
        # catalog (confirmed via /models on 2026-09-12 — verify with
        # `client.models.list()` if this stops working again). gpt-oss-120b
        # is the closest available large general-purpose alternative.
        self._model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        # Separate daily-token-quota bucket on Groq's free tier from the
        # primary model — confirmed live on 2026-09-12 when gpt-oss-120b's
        # 200k/day cap was hit mid-session and gpt-oss-20b kept working.
        # Used automatically on a 429, not as the default (20b is less
        # reliable about the citation-format contract than 120b).
        self._fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

    def complete(self, system: str, user: str) -> str:
        try:
            return self._complete_with(self._model, system, user)
        except RateLimitError:
            if self._fallback_model == self._model:
                raise
            log.warning(
                "Groq model %s rate-limited, retrying once with %s",
                self._model, self._fallback_model,
            )
            return self._complete_with(self._fallback_model, system, user)

    def _complete_with(self, model: str, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            # gpt-oss-120b is a reasoning model — it spends a chunk of the
            # token budget on hidden chain-of-thought before the visible
            # answer, so this needs real headroom, not just the answer's
            # own expected length.
            max_tokens=3000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class FallbackClient:
    """Tries a list of clients in order, falling through to the next one
    on any failure — not just a rate limit, since a provider can also be
    down or erroring for other reasons. Used when more than one LLM key is
    configured (e.g. GROQ_API_KEY and ANTHROPIC_API_KEY both set) so one
    provider's outage doesn't take down every request. Logs which client
    actually answered, so a persistent problem with one provider is
    visible rather than silently masked forever by the next.
    """

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackClient needs at least one client")
        self._clients = clients
        self.name = "+".join(c.name for c in clients)

    def complete(self, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for client in self._clients:
            try:
                result = client.complete(system, user)
                if client is not self._clients[0]:
                    log.warning("answered via fallback client %s", client.name)
                return result
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
                log.warning("client %s failed (%s), trying next", client.name, exc)
                last_exc = exc
        raise last_exc


class StubClient:
    """No network, no API key. Deterministic: for each passage that shares
    enough vocabulary with the question, emits one claim built from that
    passage's first sentence, citing it; passages with no real overlap are
    left uncited and folded into `unsupported`. Exists so the pipeline —
    prompt construction, contract enforcement, eval metrics — is provable
    end-to-end before an ANTHROPIC_API_KEY exists to run the real thing.
    """

    name = "stub"

    def complete(self, system: str, user: str) -> str:
        passages = re.findall(
            r'\[(\d+)\] chunk_id=(\S+).*?\ntext: (.+?)(?=\n\[\d+\]|\nQUESTION:|\Z)',
            user,
            re.DOTALL,
        )
        question_match = re.search(r"QUESTION:\s*(.+)", user)
        question = question_match.group(1).strip() if question_match else ""
        q_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", question)}

        claims = []
        for _, chunk_id, text in passages:
            p_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)}
            if len(q_terms & p_terms) < 2:
                continue
            first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
            claims.append({"text": first_sentence[:300], "chunk_ids": [chunk_id]})

        unsupported = [] if claims else [question]
        return json.dumps({"claims": claims, "unsupported": unsupported})


def _strip_code_fence(text: str) -> str:
    """Open-weight instruction-tuned models (Llama included) often wrap
    JSON in ```json ... ``` even when told to emit raw JSON only; strip
    that before parsing rather than failing the whole answer over it.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _build_prompt(question: str, passages: list[Chunk], *, prior_question: str | None = None) -> str:
    lines = []
    if prior_question:
        lines.append(
            f"CONTEXT: the user's previous question in this conversation was: "
            f"{prior_question!r}. If the question below uses a word like "
            f"\"this\"/\"it\"/\"that\" without saying what it refers to, "
            f"resolve it using that previous question — but still only "
            f"answer using the numbered passages below, and still cite them."
        )
    for i, p in enumerate(passages, start=1):
        lines.append(
            f"[{i}] chunk_id={p.chunk_id} source={p.source_type} title={p.title!r} "
            f"section={p.section!r}\ntext: {p.text[:MAX_PASSAGE_CHARS]}"
        )
    if prior_question:
        lines.append(
            f"\nQUESTION (a follow-up to \"{prior_question}\" — read it in that "
            f"light, don't treat it as a stand-alone question with no subject): {question}"
        )
    else:
        lines.append(f"\nQUESTION: {question}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    passages: list[Chunk],
    client: LLMClient,
    *,
    prior_question: str | None = None,
) -> Answer:
    passages = passages[:MAX_PASSAGES]
    valid_chunk_ids = {p.chunk_id for p in passages}
    chunk_by_id = {p.chunk_id: p for p in passages}

    prompt = _build_prompt(question, passages, prior_question=prior_question)
    try:
        raw = client.complete(SYSTEM_PROMPT, prompt)
    except Exception:
        # Any LLM-side failure (rate limit exhausted past the fallback,
        # network error, provider outage) degrades to an honest "can't
        # answer right now" instead of a 500 — an outage is not a reason
        # to crash the endpoint, and it's definitely not a reason to
        # fabricate an answer from no model output at all.
        log.exception("LLM call failed answering %r", question)
        return Answer(
            claims=[],
            unsupported=[f"The answer service is temporarily unavailable — couldn't process: {question}"],
            sources=[],
        )

    try:
        parsed = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        # The model didn't follow the output contract at all — this is a
        # total-abstention outcome, not a crash. Nothing here is trustworthy
        # enough to turn into a claim.
        return Answer(claims=[], unsupported=[question], sources=[])

    claims: list[Claim] = []
    unsupported: list[str] = list(parsed.get("unsupported", []))
    cited_ids: set[str] = set()

    def _resolve(raw_ids) -> list[str]:
        """Passage numbers (the documented format: 1-based index into
        `passages`) resolved to real chunk_ids; a literal chunk_id string
        is also accepted in case the model ignores the numbering scheme —
        either way, only ids actually shown to the model count.
        """
        resolved: list[str] = []
        for raw_id in raw_ids:
            if isinstance(raw_id, str) and raw_id in valid_chunk_ids:
                resolved.append(raw_id)
                continue
            try:
                idx = int(raw_id)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(passages):
                resolved.append(passages[idx - 1].chunk_id)
        return resolved

    for raw_claim in parsed.get("claims", []):
        text = raw_claim.get("text", "").strip()
        chunk_ids = _resolve(raw_claim.get("sources") or raw_claim.get("chunk_ids") or [])

        if not text:
            continue
        if not chunk_ids:
            # Spec §6: a claim with no valid citation is dropped, not
            # rendered — the model naming a chunk_id that wasn't even shown
            # to it is treated the same as citing nothing.
            unsupported.append(text)
            continue

        claims.append(Claim(text=text, chunk_ids=chunk_ids))
        cited_ids.update(chunk_ids)

    sources = [
        Citation(
            chunk_id=cid,
            source_id=chunk_by_id[cid].source_id,
            source_type=chunk_by_id[cid].source_type,
            title=chunk_by_id[cid].title,
            section=chunk_by_id[cid].section,
            url=chunk_by_id[cid].url,
        )
        for cid in sorted(cited_ids)
    ]

    return Answer(claims=claims, unsupported=unsupported, sources=sources)
