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
        logger = logging.getLogger("test_memory_pg")
        self.mm = MemoryManager(config, logger)
        tmp = tempfile.mkdtemp()
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

    def test_memory_context_is_tribal_knowledge_only(self, _):
        """Rows of removed types left in Postgres from before are never surfaced."""
        group = f"g_{uuid4().hex[:8]}"
        tk = self.mm.create_entry(
            "tribal_knowledge", "rainfall is in inches", "user", "u1",
            group_id=group, dataset_name=self.dataset,
        )
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
            entries = self.db_api.retrieve_tribal_knowledge(self.dataset, [group], ["rainfall"], 10)
            self.assertEqual([e["memory_id"] for e in entries], [tk.memory_id])
        finally:
            for record in [tk, *legacy]:
                self.mm.delete_entry(record.memory_id)

    def test_tribal_knowledge_per_need_with_relevance_cutoff(self, _):
        group = f"g_{uuid4().hex[:8]}"
        rain = self.mm.create_entry(
            "tribal_knowledge", "Precipitation values are recorded in inches.", "user", "u1",
            group_id=group, dataset_name=self.dataset,
        )
        beach = self.mm.create_entry(
            "tribal_knowledge", "Beach status code 3 means closed for swimming.", "user", "u1",
            group_id=group, dataset_name=None,  # untagged: applies to every dataset
        )
        other_group = self.mm.create_entry(
            "tribal_knowledge", "Precipitation is also tracked monthly.", "user", "u1",
            group_id=f"g_{uuid4().hex[:8]}", dataset_name=self.dataset,
        )
        try:
            needs = [
                "What unit is the precipitation in?",
                "Which beaches were closed?",
                "Which rows count toward the precipitation total?",
                "What is the answer's rounding?",
            ]
            entries = self.db_api.retrieve_tribal_knowledge(self.dataset, [group], needs, 10)
            # Each entry once, under the first need it matched; nothing for a need
            # that shares no term with any entry; other groups' entries never.
            self.assertEqual(
                [(e["memory_id"], e["query"]) for e in entries],
                [(rain.memory_id, needs[0]), (beach.memory_id, needs[1])],
            )
            self.assertEqual(
                self.db_api.retrieve_tribal_knowledge(self.dataset, [group], ["unrelated words"], 10),
                [],
            )
        finally:
            for record in (rain, beach, other_group):
                self.mm.delete_entry(record.memory_id)



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
