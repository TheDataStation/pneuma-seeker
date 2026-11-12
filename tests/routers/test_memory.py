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



if __name__ == "__main__":
    unittest.main()
