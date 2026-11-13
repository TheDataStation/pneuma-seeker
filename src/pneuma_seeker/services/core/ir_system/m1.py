"""
M1 (windowed batching), ported from Agentic Catalog (catalog_service/retrieval.py):
an LLM judges a BM25 ranking from the top, a window of batches at a time, and
stops after an empty window or when the token budget is spent.
"""

from __future__ import annotations

import functools
import hashlib
import random
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from logging import Logger
from typing import Any

from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role

# Both criteria are copied verbatim from Agentic Catalog.
TABLE_INSTRUCTIONS = (
    "You decide which candidate tables are relevant to an information need. A candidate is relevant if and "
    "only if answering the information need from the data could require reading it: its rows are filtered, "
    "joined, grouped, aggregated or returned, including a table needed only to join two others. A candidate "
    "may stand for a family of tables with the same columns; it is relevant if any of them is. When in doubt, "
    "mark the candidate relevant: a missing table makes the answer impossible, while an extra one only costs "
    "a little.\n"
    "Judge every candidate. Return a JSON object with one boolean per candidate id: true if relevant, "
    "false otherwise. Output only the JSON object, without any explanation."
)

TRIBAL_KNOWLEDGE_INSTRUCTIONS = (
    "You decide which candidate notes about a dataset are relevant to an information need.\n"
    "A candidate is relevant if and only if the answer to the information need would differ depending on whether "
    "the candidate is taken into account: an analyst who answers the information need from the data while following "
    "the candidate would reach a different answer than one who ignores it. This holds when the candidate implies a "
    "concrete change to how this answer is computed from the data, e.g., which records to include or exclude, how to "
    "interpret, convert or correct a value, which definition of a term to use, or how records relate. A candidate that "
    "implies no such change is not relevant, however closely it relates to the topic of the information need.\n"
    "Judge every candidate. Return a JSON object with one boolean per candidate id: true if relevant, "
    "false otherwise."
)


@functools.cache
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """o200k tokens (~4 characters per token if tiktoken is unavailable)."""
    try:
        return len(_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


def seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("\n".join(parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass
class M1Result:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    read: int = 0
    calls: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.errors)


def _pack(entries: list[dict], tau: int) -> list[list[dict]]:
    """Batches of at most tau tokens; an entry longer than tau is a batch of its own."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for entry in entries:
        tokens = entry["_tokens"]
        if current and size + tokens > tau:
            batches.append(current)
            current, size = [], 0
        current.append(entry)
        size += tokens
        if tokens > tau:
            batches.append(current)
            current, size = [], 0
    if current:
        batches.append(current)
    return batches


def _shuffled(entries: list[dict], rng: random.Random) -> list[dict]:
    entries = list(entries)
    rng.shuffle(entries)
    return entries


class M1:
    def __init__(self, language_model_api: LanguageModelAPI, logger: Logger | None = None) -> None:
        self.language_model_api = language_model_api
        self.logger = logger

    def select(
        self,
        need: str,
        ranked: Iterable[dict[str, Any]],
        rng: random.Random,
        *,
        tau: int,
        window_batches: int,
        empty_windows: int,
        budget: int,
        instructions: str,
    ) -> M1Result:
        """`ranked`: best first, each with an `id` and `content`. Returns the accepted, in rank order."""
        out = M1Result()
        source: Iterator[dict[str, Any]] = iter(ranked)
        read: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None
        exhausted = False
        accepted: set[str] = set()
        spent, empty_run = 0, 0
        while True:
            window: list[dict[str, Any]] = []
            filled = 0
            while True:
                if pending is None:
                    if exhausted:
                        break
                    try:
                        pending = dict(next(source))
                    except StopIteration:
                        exhausted = True
                        break
                    pending["_tokens"] = count_tokens(pending["content"])
                cost = min(pending["_tokens"], tau)
                if window and filled + cost > window_batches * tau:
                    break
                window.append(pending)
                read.append(pending)
                filled += cost
                pending = None
            if not window:
                break
            batches, over_budget = [], False
            for batch in _pack(_shuffled(window, rng), tau):
                batch_tokens = sum(e["_tokens"] for e in batch)
                if spent + batch_tokens > budget:
                    over_budget = True
                    break
                spent += batch_tokens
                batches.append(_shuffled(batch, rng))
            found = False
            if batches:
                with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                    results = list(pool.map(lambda b: self._classify(need, b, instructions), batches))
                for relevant, error in results:
                    out.calls += 1
                    if error:
                        out.errors.append(error)
                    accepted |= relevant
                    found = found or bool(relevant)
            if over_budget:
                break
            empty_run = 0 if found else empty_run + 1
            if empty_run >= empty_windows:
                break
        out.read = len(read)
        out.accepted = [
            {k: v for k, v in e.items() if k != "_tokens"} for e in read if e["id"] in accepted
        ]
        return out

    def _classify(self, need: str, batch: list[dict], instructions: str) -> tuple[set[str], str | None]:
        labels = [f"c{i}" for i in range(1, len(batch) + 1)]
        candidates = "\n\n".join(f"[{label}] {e['content']}" for label, e in zip(labels, batch))
        prompt = f"Information need:\n{need}\n\nCandidates:\n\n{candidates}"
        try:
            reply = "".join(
                self.language_model_api.chat(
                    [
                        LLMMessage(role=Role.SYSTEM.value, content=instructions),
                        LLMMessage(role=Role.USER.value, content=prompt),
                    ],
                    LLMOption(json_mode=True),
                )
            )
            verdict = parse_json(reply)
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"[M1] classifier call failed: {exc!r}")
            return set(), repr(exc)[:300]
        if not isinstance(verdict, dict):
            return set(), None
        return {e["id"] for label, e in zip(labels, batch) if verdict.get(label) is True}, None

