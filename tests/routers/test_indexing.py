import logging
import os
import sys
from unittest.mock import MagicMock
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.routers import indexing


class TestIndexRouter(unittest.TestCase):
    """Tests for the indexing router handling dataset index requests and metadata retrieval."""

    def setUp(self):
        self.logger = logging.getLogger("test")
        
        self.app = FastAPI()
        self.app.include_router(indexing.router)

        self.mock_indexing_service = MagicMock()

        self.app.dependency_overrides[indexing.get_indexing_service] = lambda: self.mock_indexing_service

        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_index_dataset_success_schedules_background_task(self):
        """Tests that a valid index request starts a run, triggers a background job, and returns metadata."""
        # Arrange
        fake_run_id = "run_abc123"
        fake_metadata = {"num_rows": 100, "status": "processing"}
        
        self.mock_indexing_service.start_indexing_run.return_value = fake_run_id
        self.mock_indexing_service.get_latest_index_metadata.return_value = fake_metadata

        payload = {
            "dataset_name": "test_dataset",
            "connector_config": {"type": "s3", "bucket": "my-bucket"},
            "schema_summaries": [{"table_name": "Table 1", "summary": "ID: ID of the record"}, {"table_name": "Table 2", "summary": "Name: Name of the record"}],
            "overwrite": True
        }

        # Act
        response = self.client.post("/index/", json=payload)

        # Assert HTTP layer responses
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["run_id"], fake_run_id)
        self.assertEqual(body["dataset_name"], "test_dataset")
        self.assertEqual(body["latest_metadata"], fake_metadata)

        # Assert router correctly extracted data into a Pandas DataFrame 
        # and passed it to the service's background execution method
        self.mock_indexing_service.run_indexing_job.assert_called_once()
        called_kwargs = self.mock_indexing_service.run_indexing_job.call_args.kwargs
        
        passed_df = called_kwargs.get("schema_summaries")
        self.assertIsInstance(passed_df, pd.DataFrame)
        self.assertEqual(len(passed_df), 2)
        self.assertEqual(list(passed_df.columns), ["table_name", "summary"])

    def test_get_latest_metadata_success(self):
        """Tests that existing metadata can be cleanly fetched by dataset name."""
        # Arrange
        fake_metadata = {"last_indexed": "2026-06-08", "state": "healthy"}
        self.mock_indexing_service.get_latest_index_metadata.return_value = fake_metadata

        # Act
        response = self.client.get("/index/my_awesome_dataset/latest")

        # Assert
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["dataset_name"], "my_awesome_dataset")
        self.assertEqual(body["latest_metadata"], fake_metadata)
        self.mock_indexing_service.get_latest_index_metadata.assert_called_once_with("my_awesome_dataset")

    def test_index_dataset_value_error_returns_400(self):
        """Tests that a ValueError from the service transforms into a 400 Bad Request."""
        # Arrange
        self.mock_indexing_service.start_indexing_run.side_effect = ValueError("Invalid connector credentials")

        # Act - Added missing 'overwrite' parameter to payload
        response = self.client.post(
            "/index/", 
            json={"dataset_name": "bad_dataset", "connector_config": {}, "overwrite": False}
        )

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid connector credentials")

    def test_index_dataset_runtime_error_returns_502(self):
        """Tests that a RuntimeError from the service transforms into a 502 Bad Gateway."""
        # Arrange
        self.mock_indexing_service.start_indexing_run.side_effect = RuntimeError("Database cluster unreachable")

        # Act - Added missing 'overwrite' parameter to payload
        response = self.client.post(
            "/index/", 
            json={"dataset_name": "unreachable_dataset", "connector_config": {}, "overwrite": False}
        )

        # Assert
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Database cluster unreachable")

    def test_index_dataset_unexpected_exception_returns_500(self):
        """Tests that any unexpected systemic exception returns a 500 Internal Server Error."""
        # Arrange
        self.mock_indexing_service.start_indexing_run.side_effect = Exception("Out of memory on engine node")

        # Act - Added missing 'overwrite' parameter to payload
        response = self.client.post(
            "/index/", 
            json={"dataset_name": "crash_dataset", "connector_config": {}, "overwrite": False}
        )

        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to start indexing: Out of memory on engine node", response.json()["detail"])

    def test_get_latest_metadata_missing_dataset_returns_404(self):
        """Tests that asking for a dataset that hasn't been indexed returns a 404 Not Found."""
        # Arrange
        self.mock_indexing_service.get_latest_index_metadata.return_value = None

        # Act
        response = self.client.get("/index/ghost_dataset/latest")

        # Assert
        self.assertEqual(response.status_code, 404)
        self.assertIn("No indexing metadata found for dataset 'ghost_dataset'.", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
