import os
import sys
import unittest
from logging import getLogger
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")),
)

import pneuma_seeker.services.indexing.main as indexing_main
from pneuma_seeker.services.db.datasets.registry import DatasetSource
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import Table, TableContext


class DummyDBAPI:
    def __init__(self, tables: list[str], descriptions: dict[str, str] | None = None):
        self.tables = tables
        self.descriptions = descriptions or {}
        sources = {
            "ds": DatasetSource("ds", "postgres", {"host": "h", "dbname": "d", "user": "u", "password": "secret"})
        }
        self.pneuma_db = SimpleNamespace(get_dataset_source=sources.get)

    def list_dataset_tables(self, dataset_name):
        return self.tables

    def get_table_description(self, dataset_name, table_name):
        return self.descriptions.get(table_name, "")

    def dataset_fingerprint(self, dataset_name):
        return "fp-1"


class DummyRetriever:
    def __init__(self, user_id, chat_id, config, db_api, language_model_api):
        self.indexed_docs = None
        self.summaries = None

    def index(self, documents, overwrite):
        self.indexed_docs = documents

    def index_with_existing_schema_summaries(self, documents, existing_schema_summaries, overwrite):
        self.indexed_docs = documents
        self.summaries = existing_schema_summaries


class DummyMetadataStore:
    def __init__(self):
        self.started = []
        self.succeeded = None
        self.failed = None

    def record_run_started(self, dataset_name, source_type, source_config, snapshot_id):
        self.started.append((dataset_name, source_type, source_config, snapshot_id))
        return f"run-{len(self.started)}"

    def mark_run_succeeded(self, run_id, indexed_stream_count, indexed_table_count):
        self.succeeded = (run_id, indexed_table_count)

    def mark_run_failed(self, run_id, reason):
        self.failed = (run_id, reason)

    def get_latest_run(self, dataset_name):
        return None

    def close(self):
        pass


class IndexingServiceTests(unittest.TestCase):
    def service(self, db_api: DummyDBAPI) -> indexing_main.IndexingService:
        with patch.object(indexing_main, "LanguageModelAPI", lambda c, l: None), \
                patch.object(indexing_main, "PneumaRetriever", DummyRetriever), \
                patch.object(indexing_main, "IndexingMetadataStore", DummyMetadataStore):
            return indexing_main.IndexingService(Config(), getLogger("t"), db_api)  # type: ignore[arg-type]

    def test_run_is_recorded_with_redacted_config_and_fingerprint(self):
        service = self.service(DummyDBAPI(["t"]))
        run_id = service.start_indexing_run("ds")
        self.assertEqual(run_id, "run-1")
        dataset, source_type, config, snapshot = service.metadata_store.started[0]
        self.assertEqual((dataset, source_type, snapshot), ("ds", "postgres", "fp-1"))
        self.assertNotIn("secret", str(config))

    def test_unknown_dataset_is_refused(self):
        with self.assertRaises(ValueError):
            self.service(DummyDBAPI(["t"])).start_indexing_run("nope")

    def test_indexes_every_table_with_its_description(self):
        service = self.service(DummyDBAPI(["orders", "users"], {"users": "People"}))
        service.run_indexing_job("ds", "run-1")
        docs = service.retriever.indexed_docs
        tables = [d for d in docs if isinstance(d, Table)]
        contexts = [d for d in docs if isinstance(d, TableContext)]
        self.assertEqual([t.metadata["table_name"] for t in tables], ["orders", "users"])
        self.assertEqual([(c.metadata["table_name"], c.content) for c in contexts], [("users", "People")])
        self.assertEqual(service.metadata_store.succeeded, ("run-1", 2))

    def test_dataset_without_tables_marks_the_run_failed(self):
        service = self.service(DummyDBAPI([]))
        with self.assertRaises(ValueError):
            service.run_indexing_job("ds", "run-1")
        self.assertEqual(service.metadata_store.failed[0], "run-1")


if __name__ == "__main__":
    unittest.main()
