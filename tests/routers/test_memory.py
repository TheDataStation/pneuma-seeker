import os
import sys
import unittest
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.routers import memory
from pneuma_seeker.routers.auth import get_current_user
from pneuma_seeker.services.db.memory.models import MemoryEntryRecord
from pneuma_seeker.services.db.users.models import UserRecord


class FakeMemoryManager:
    """In-memory stand-in for MemoryManager's entry CRUD."""

    def __init__(self):
        self.entries: dict[str, MemoryEntryRecord] = {}

    def get_by_key(self, memory_type, dataset_name, key_text):
        for e in self.entries.values():
            if (e.memory_type, e.dataset_name, e.key_text) == (memory_type, dataset_name, key_text):
                return e
        return None

    def get_entry(self, memory_id):
        return self.entries.get(memory_id)

    def create_entry(self, memory_type, content, source, source_user_id=None, **kw):
        record = MemoryEntryRecord(
            memory_id=str(uuid4()),
            memory_type=memory_type,
            content=content,
            source=source,
            source_user_id=source_user_id,
            updated_at=datetime(2026, 11, 12),
            **{k: kw.get(k) for k in ("dataset_name", "key_text")},
        )
        self.entries[record.memory_id] = record
        return record

    def update_entry(self, memory_id, content):
        self.entries[memory_id].content = content
        return True

    def delete_entry(self, memory_id):
        return self.entries.pop(memory_id, None) is not None


def _deny_secret(dataset_name, _user):
    if dataset_name == "secret":
        raise HTTPException(status_code=403, detail="nope")


@patch("pneuma_seeker.routers.memory.require_dataset_access", side_effect=_deny_secret)
class TestDatasetDescription(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(memory.router)
        self.mm = FakeMemoryManager()
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u@x.io", username="u@x.io", group_id="g1", is_active=True
        )
        self.app.dependency_overrides[memory.get_memory_manager] = lambda: self.mm
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_missing_description_is_empty(self, _):
        res = self.client.get("/memory/dataset_description/environment")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["description"], "")

    def test_set_update_and_clear(self, _):
        url = "/memory/dataset_description/environment"
        res = self.client.put(url, json={"description": "  Massachusetts water data  "})
        self.assertEqual(res.json()["description"], "Massachusetts water data")
        self.client.put(url, json={"description": "Boston beach and rainfall data"})

        # Still a single entry per dataset, keyed by the dataset's name.
        self.assertEqual(len(self.mm.entries), 1)
        (entry,) = self.mm.entries.values()
        self.assertEqual(
            (entry.memory_type, entry.dataset_name, entry.key_text),
            ("dataset_metadata", "environment", "environment"),
        )
        self.assertEqual(
            self.client.get(url).json()["description"], "Boston beach and rainfall data"
        )

        res = self.client.put(url, json={"description": "   "})
        self.assertEqual(res.json()["description"], "")
        self.assertEqual(self.mm.entries, {})

    def test_requires_dataset_access(self, _):
        self.assertEqual(
            self.client.get("/memory/dataset_description/secret").status_code, 403
        )
        res = self.client.put("/memory/dataset_description/secret", json={"description": "x"})
        self.assertEqual(res.status_code, 403)
        self.assertEqual(self.mm.entries, {})

    def test_listable_as_dataset_scoped_memory(self, _):
        self.assertIn("dataset_metadata", memory._DATASET_SCOPED_TYPES)
        # Created only through the PUT endpoint, which keeps one entry per dataset.
        self.assertNotIn("dataset_metadata", memory._MANUALLY_CREATABLE_TYPES)



@patch("pneuma_seeker.routers.memory.require_dataset_access")
class TestDatasetDescriptionPostgres(unittest.TestCase):
    """End to end on the real Postgres memory store (the session's testcontainer):
    written through the API, read back the way the agents read it."""

    def setUp(self):
        import logging
        import tempfile

        from pneuma_seeker.services.core.api.db import DBAPI
        from pneuma_seeker.services.db.memory.manager import MemoryManager
        from pneuma_seeker.shared.config import Config

        config = Config(".env.test")
        tmp = tempfile.mkdtemp()
        config.RETRIEVAL_INDEX_ROOT = f"{tmp}/indices"
        self.config = config
        logger = logging.getLogger("test_memory_pg")
        self.logger = logger
        self.mm = MemoryManager(config, logger)
        self.db_api = DBAPI(
            config, logger, dataset_db_path=f"{tmp}/d", workspace_db_path=f"{tmp}/w"
        )
        self.dataset = f"ds_{uuid4().hex[:8]}"
        self.app = FastAPI()
        self.app.include_router(memory.router)
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u@x.io", username="u@x.io", group_id="g1", is_active=True
        )
        self.app.dependency_overrides[memory.get_memory_manager] = lambda: self.mm
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.put(f"/memory/dataset_description/{self.dataset}", json={"description": ""})
        self.app.dependency_overrides.clear()

    def test_round_trip_reaches_agents(self, _):
        url = f"/memory/dataset_description/{self.dataset}"
        self.assertEqual(self.db_api.get_dataset_description(self.dataset), "")

        self.client.put(url, json={"description": "First"})
        self.client.put(url, json={"description": "Second"})
        self.assertEqual(self.client.get(url).json()["description"], "Second")
        self.assertEqual(self.db_api.get_dataset_description(self.dataset), "Second")
        listed = self.mm.list_entries("dataset_metadata", dataset_name=self.dataset)
        self.assertEqual(len(listed["items"]), 1)

        self.client.put(url, json={"description": ""})
        self.assertEqual(self.db_api.get_dataset_description(self.dataset), "")

    def _retriever(self, method: str, llm=None):
        from pneuma_seeker.services.core.ir_system.tribal_knowledge import (
            TribalKnowledgeRetriever,
        )

        self.config.TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD = method
        return TribalKnowledgeRetriever(self.config, self.logger, self.db_api, llm)

    def _entry(self, content, group, dataset):
        return self.mm.create_entry(
            "tribal_knowledge", content, "user", "u1", group_id=group, dataset_name=dataset
        )

    def test_ranking_is_tribal_knowledge_only(self, _):
        """Rows of removed types left in Postgres from before are never surfaced."""
        group = f"g_{uuid4().hex[:8]}"
        tk = self._entry("rainfall is in inches", group, self.dataset)
        legacy = [
            self.mm.create_entry(
                "user_preference", "rainfall charts please", "llm_extraction", "u1",
                user_id="u1", dataset_name=self.dataset,
            ),
            self.mm.create_entry(
                "agent_learning", "rainfall answer", "agent_learning", None,
                dataset_name=self.dataset, key_text="rainfall?",
            ),
        ]
        try:
            ranked = self.db_api.rank_tribal_knowledge(self.dataset, [group], "rainfall")
            self.assertEqual([e["memory_id"] for e in ranked], [tk.memory_id])
        finally:
            for record in [tk, *legacy]:
                self.mm.delete_entry(record.memory_id)

    def test_bm25_method_per_need_with_relevance_cutoff(self, _):
        group = f"g_{uuid4().hex[:8]}"
        rain = self._entry("Precipitation values are recorded in inches.", group, self.dataset)
        beach = self._entry("Beach status code 3 means closed for swimming.", group, None)  # untagged
        other_group = self._entry(
            "Precipitation is also tracked monthly.", f"g_{uuid4().hex[:8]}", self.dataset
        )
        try:
            needs = [
                "What unit is the precipitation in?",
                "Which beaches were closed?",
                "Which rows count toward the precipitation total?",
                "What is the answer's rounding?",
            ]
            entries = self._retriever("bm25").retrieve(self.dataset, [group], needs)
            # Each entry once, under the first need it matched; nothing for a need
            # that shares no term with any entry; other groups' entries never.
            self.assertEqual(
                [(e["memory_id"], e["query"]) for e in entries],
                [(rain.memory_id, needs[0]), (beach.memory_id, needs[1])],
            )
            self.assertEqual(
                self._retriever("bm25").retrieve(self.dataset, [group], ["unrelated words"]), []
            )
        finally:
            for record in (rain, beach, other_group):
                self.mm.delete_entry(record.memory_id)

    def test_m1_method_judges_ranked_entries(self, _):
        """M1 accepts what the LLM judges relevant — including an entry sharing no
        word with the need, which BM25 alone would drop."""
        import json as _json
        import re as _re

        group = f"g_{uuid4().hex[:8]}"
        rain = self._entry("Precipitation values are recorded in inches.", group, self.dataset)
        code = self._entry("Status 3 means the station was offline.", group, self.dataset)
        noise = self._entry("Precipitation dashboards refresh nightly.", group, self.dataset)

        class JudgeLLM:
            def chat(self, messages, option=None):
                batch = messages[1]["content"].split("Candidates:", 1)[1]
                labels = _re.findall(r"\[(c\d+)\] ([^\n]*)", batch)
                yield _json.dumps({l: ("inches" in t or "offline" in t) for l, t in labels})

        try:
            entries = self._retriever("m1", JudgeLLM()).retrieve(
                self.dataset, [group], ["How much rain fell?"]
            )
            self.assertEqual({e["memory_id"] for e in entries}, {rain.memory_id, code.memory_id})
        finally:
            for record in (rain, code, noise):
                self.mm.delete_entry(record.memory_id)

    def test_m1_falls_back_to_bm25_when_the_llm_fails(self, _):
        group = f"g_{uuid4().hex[:8]}"
        rain = self._entry("Precipitation values are recorded in inches.", group, self.dataset)
        other = self._entry("Beach codes mean closures.", group, self.dataset)

        class BrokenLLM:
            def chat(self, messages, option=None):
                raise RuntimeError("model unavailable")
                yield  # pragma: no cover

        try:
            entries = self._retriever("m1", BrokenLLM()).retrieve(
                self.dataset, [group], ["precipitation unit"]
            )
            self.assertEqual([e["memory_id"] for e in entries], [rain.memory_id])
        finally:
            for record in (rain, other):
                self.mm.delete_entry(record.memory_id)

    def test_index_is_persisted_and_follows_writes(self, _):
        from pathlib import Path

        from pneuma_seeker.services.db.memory.tribal_index import TribalKnowledgeIndex

        group = f"g_{uuid4().hex[:8]}"
        root = Path(self.config.RETRIEVAL_INDEX_ROOT)
        first = self._entry("Precipitation values are recorded in inches.", group, self.dataset)
        try:
            self.assertEqual(
                [e["memory_id"] for e in self.db_api.rank_tribal_knowledge(self.dataset, [group], "inches")],
                [first.memory_id],
            )
            current = (root / "tribal_knowledge" / "CURRENT").read_text()

            # A fresh process loads the same index from disk instead of rebuilding it.
            fresh = TribalKnowledgeIndex(root, self.mm)
            self.assertGreater(fresh.scores("inches")[first.memory_id], 0)
            self.assertEqual((root / "tribal_knowledge" / "CURRENT").read_text(), current)

            # A new entry is searchable at once; a deleted one is gone.
            second = self._entry("Snowfall counts as precipitation in inches of water.", group, self.dataset)
            ranked = self.db_api.rank_tribal_knowledge(self.dataset, [group], "snowfall")
            self.assertEqual(ranked[0]["memory_id"], second.memory_id)
            self.mm.delete_entry(second.memory_id)
            ranked = self.db_api.rank_tribal_knowledge(self.dataset, [group], "snowfall")
            self.assertEqual([e["memory_id"] for e in ranked], [first.memory_id])
            self.assertEqual(ranked[0]["score"], 0.0)  # kept, last, for M1 to decide
        finally:
            self.mm.delete_entry(first.memory_id)


class TestRemovedMemoryTypes(unittest.TestCase):
    """User preferences and agent learnings were removed; only tribal knowledge
    (plus dataset/table/column metadata) is left."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(memory.router)
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u@x.io", username="u@x.io", group_id="g1", is_active=True
        )
        self.app.dependency_overrides[memory.get_memory_manager] = FakeMemoryManager
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_removed_types_are_rejected(self):
        for removed in ("user_preference", "agent_learning"):
            res = self.client.get(f"/memory/entries?memory_type={removed}")
            self.assertEqual(res.status_code, 400, removed)
            res = self.client.post(
                "/memory/entries", json={"memory_type": removed, "content": "x"}
            )
            self.assertEqual(res.status_code, 400, removed)

    def test_memory_types_left(self):
        from pneuma_seeker.services.db.memory.models import MemoryType

        self.assertEqual(
            {t.value for t in MemoryType},
            {"tribal_knowledge", "dataset_metadata", "table_metadata", "column_metadata"},
        )



if __name__ == "__main__":
    unittest.main()
