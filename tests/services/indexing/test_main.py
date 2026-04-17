import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.services.indexing.connectors.base import SourceConnector
from pneuma_seeker.services.indexing.main import (
    IndexingService,
)
from pneuma_seeker.services.indexing.metadata_store import IndexingMetadataStore
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import Table, TableContext


class FakeConnector(SourceConnector):
    def __init__(self, config):
        super().__init__(config)

    @property
    def source_type(self) -> str:
        return "fake"

    def check_connection(self) -> bool:
        return True

    def discover(self):
        return [
            {
                "stream": "public.users",
                "table_name": "users",
                "description": "User table",
            },
            {
                "stream": "public.orders",
                "table_name": "orders",
            },
        ]

    def read(self, stream: str):
        if stream == "public.users":
            yield {"id": 1, "name": "Alice"}
            yield {"id": 2, "name": "Bob"}
            return
        if stream == "public.orders":
            yield {"order_id": 10, "amount": 100}
            return
        raise KeyError(stream)


class FakePostgresConnector(FakeConnector):
    @property
    def source_type(self) -> str:
        return "postgres"

    @property
    def connection_string(self) -> str:
        return "host=localhost port=5432 user=x password=y dbname=z"


class BrokenConnector(FakeConnector):
    def check_connection(self) -> bool:
        return False


class TestIndexingMetadataStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = IndexingMetadataStore(os.path.join(self.tmpdir, "indexing.db"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_run_started_and_succeeded(self):
        run_id = self.store.record_run_started(
            dataset_name="my_dataset",
            source_type="csv",
            source_config={"type": "csv", "file_path": "/tmp/x.csv"},
            snapshot_id="snapshot-1",
        )

        started = self.store.get_run(run_id)
        self.assertIsNotNone(started)
        assert started is not None

        self.assertEqual(started["dataset_name"], "my_dataset")
        self.assertEqual(started["status"], "RUNNING")

        self.store.mark_run_succeeded(run_id, indexed_stream_count=3, indexed_table_count=3)
        completed = self.store.get_run(run_id)
        assert completed is not None

        self.assertEqual(completed["status"], "SUCCEEDED")
        self.assertEqual(completed["indexed_stream_count"], 3)
        self.assertEqual(completed["indexed_table_count"], 3)

    def test_mark_run_failed(self):
        run_id = self.store.record_run_started(
            dataset_name="my_dataset",
            source_type="csv",
            source_config={"type": "csv", "file_path": "/tmp/x.csv"},
            snapshot_id="snapshot-1",
        )

        self.store.mark_run_failed(run_id, "boom")
        failed = self.store.get_run(run_id)
        assert failed is not None

        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error_message"], "boom")


class TestIndexingService(unittest.TestCase):
    def setUp(self):
        self.db_api = MagicMock()
        self.retriever = MagicMock()
        self.metadata_store = MagicMock()

        self.service = IndexingService(
            Config(),
            MagicMock(),
        )

        self.service.db_api = self.db_api
        self.service.language_model_api = MagicMock()
        self.service.retriever = self.retriever
        self.service.metadata_store = self.metadata_store
        self.service.metadata_store.record_run_started.return_value = "run-1"

    def test_index_dataset_success_with_csv_like_connector(self):
        self.service.register_connector("fake", FakeConnector)

        run_id = self.service.index_dataset(
            dataset_name="sales",
            connector_config={"type": "fake"},
            metadata_available=True,
        )

        self.assertEqual(run_id, "run-1")
        self.db_api.ingest_dataset.assert_called_once()
        self.metadata_store.mark_run_succeeded.assert_called_once_with(
            run_id="run-1", indexed_stream_count=2, indexed_table_count=2
        )

        args, kwargs = self.retriever.index.call_args
        docs = args[0]
        self.assertEqual(len([doc for doc in docs if isinstance(doc, Table)]), 2)
        self.assertEqual(len([doc for doc in docs if isinstance(doc, TableContext)]), 1)

    def test_index_dataset_uses_existing_schema_summaries(self):
        self.service.register_connector("fake", FakeConnector)

        schema_summaries = pd.DataFrame(
            [
                {"table_name": "users", "column_name": "id", "summary": "User id"},
                {
                    "table_name": "users",
                    "column_name": "name",
                    "summary": "User name",
                },
            ]
        )

        self.service.index_dataset(
            dataset_name="sales",
            connector_config={"type": "fake"},
            schema_summaries=schema_summaries,
        )

        self.retriever.index_with_existing_schema_summaries.assert_called_once()
        self.retriever.index.assert_not_called()

    def test_index_dataset_postgres_registers_connection_string(self):
        self.service.register_connector("postgres", FakePostgresConnector)

        self.service.index_dataset(
            dataset_name="pg_sales",
            connector_config={"type": "postgres"},
        )

        self.db_api.register_postgres_dataset.assert_called_once_with(
            "pg_sales", "host=localhost port=5432 user=x password=y dbname=z"
        )
        self.db_api.ingest_dataset.assert_not_called()

    def test_index_dataset_marks_failed_when_connection_invalid(self):
        self.service.register_connector("broken", BrokenConnector)

        with self.assertRaises(RuntimeError):
            self.service.index_dataset(
                dataset_name="sales",
                connector_config={"type": "broken"},
            )

        self.metadata_store.mark_run_failed.assert_called_once()


class TestIndexingAPI(unittest.TestCase):
    def setUp(self):
        self.mock_service = MagicMock()
        app.dependency_overrides[get_indexing_service] = lambda: self.mock_service
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_root_ok(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_index_dataset_success(self):
        self.mock_service.index_dataset.return_value = "run-123"
        self.mock_service.get_latest_index_metadata.return_value = {
            "run_id": "run-123",
            "status": "SUCCEEDED",
        }

        payload = {
            "dataset_name": "sales",
            "connector_config": {"type": "csv", "directory_path": "/tmp/data"},
            "metadata_available": True,
            "schema_summaries": [
                {"table_name": "users", "column_name": "id", "summary": "User id"}
            ],
        }

        response = self.client.post("/index", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["run_id"], "run-123")
        self.assertEqual(body["dataset_name"], "sales")
        self.assertEqual(body["latest_metadata"]["status"], "SUCCEEDED")

        _, kwargs = self.mock_service.index_dataset.call_args
        self.assertEqual(kwargs["dataset_name"], "sales")
        self.assertEqual(kwargs["connector_config"]["type"], "csv")
        self.assertTrue(kwargs["metadata_available"])
        self.assertIsInstance(kwargs["schema_summaries"], pd.DataFrame)

    def test_index_dataset_handles_value_error_as_400(self):
        self.mock_service.index_dataset.side_effect = ValueError("bad request")

        response = self.client.post(
            "/index",
            json={
                "dataset_name": "sales",
                "connector_config": {"type": "fake"},
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad request")

    def test_get_latest_metadata_success(self):
        self.mock_service.get_latest_index_metadata.return_value = {
            "run_id": "run-001",
            "status": "SUCCEEDED",
        }

        response = self.client.get("/index/sales/latest")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["dataset_name"], "sales")
        self.assertEqual(body["latest_metadata"]["run_id"], "run-001")

    def test_get_latest_metadata_not_found(self):
        self.mock_service.get_latest_index_metadata.return_value = None

        response = self.client.get("/index/unknown/latest")
        self.assertEqual(response.status_code, 404)
        self.assertIn("No indexing metadata found", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
