"""
Tribal knowledge for a turn: one retrieval per information need (concurrently),
merged. Each need ranks the visible entries by BM25, then keeps what M1 accepts,
or (method "bm25", or if M1 fails) the top TRIBAL_KNOWLEDGE_RETRIEVAL_TOP_K
entries sharing a term with the need. An entry matching several needs is kept
once, under the first.
"""

from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import Any

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.m1 import (
    M1,
    TRIBAL_KNOWLEDGE_INSTRUCTIONS,
    seeded_rng,
)
from pneuma_seeker.shared.config import Config


class TribalKnowledgeRetriever:
    def __init__(
        self,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ) -> None:
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.m1 = M1(language_model_api, logger)

    def retrieve(
        self,
        dataset_name: str,
        group_ids: list[str],
        queries: list[str],
        chat_id: str = "",
    ) -> list[dict[str, Any]]:
        if not queries:
            return []
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            per_query = list(
                pool.map(lambda q: self._retrieve_one(dataset_name, group_ids, q, chat_id), queries)
            )
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for query, entries in zip(queries, per_query):
            for entry in entries:
                if entry["memory_id"] in seen:
                    continue
                seen.add(entry["memory_id"])
                results.append(
                    {
                        "memory_id": entry["memory_id"],
                        "memory_type": entry["memory_type"],
                        "content": entry["content"],
                        "query": query,
                    }
                )
        return results

    def _retrieve_one(
        self, dataset_name: str, group_ids: list[str], query: str, chat_id: str
    ) -> list[dict[str, Any]]:
        ranked = self.db_api.rank_tribal_knowledge(dataset_name, group_ids, query)
        if not ranked:
            return []
        if self.config.TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD == "m1":
            result = self.m1.select(
                query,
                ({**e, "id": e["memory_id"]} for e in ranked),
                seeded_rng(chat_id, query, dataset_name),
                tau=self.config.TRIBAL_KNOWLEDGE_BATCH_TOKENS,
                window_batches=self.config.M1_WINDOW_BATCHES,
                empty_windows=self.config.M1_EMPTY_WINDOWS,
                budget=self.config.M1_TOKEN_BUDGET,
                instructions=TRIBAL_KNOWLEDGE_INSTRUCTIONS,
            )
            if not result.failed:
                self.logger.info(
                    f"[Tribal Knowledge] {query!r}: read {result.read} of {len(ranked)} entries "
                    f"in {result.calls} call(s), accepted {len(result.accepted)}"
                )
                return result.accepted
            self.logger.warning(
                f"[Tribal Knowledge] classifier failed ({result.errors[0]}): using BM25"
            )
        matching = [e for e in ranked if e["score"] > 0]
        return matching[: self.config.TRIBAL_KNOWLEDGE_RETRIEVAL_TOP_K]
