import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../src"))
)

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.ir_system.retriever.impl.enumerator import Enumerator
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table


class EnumeratorTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()

        self.dataset_root = Path(self.tmpdir) / "datasets"
        self.workspace_root = Path(self.tmpdir) / "workspaces"
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self.user_id = "uX"
        self.chat_id = "cX"
        self.dataset_name = "test_ds"

        self.db_api = DBAPI(self.config, self.logger)
        # DBAPI builds PneumaDB paths relative to source; override to temp dirs for unit tests.
        self.db_api.pneuma_db.dataset_db_path = self.dataset_root
        self.db_api.pneuma_db.workspace_db_path = self.workspace_root

        self.dataset_dir = Path(self.tmpdir) / "dataset_src"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.enumerator = Enumerator(
            self.user_id,
            self.chat_id,
            self.config,
            self.db_api,
            MagicMock(),
        )

    def tearDown(self):
        try:
            self.db_api.pneuma_db.close_all_connections()
        except Exception:
            pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, name: str, df: pd.DataFrame) -> Path:
        path = self.dataset_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def _write_metadata(self, rows: list[dict]) -> Path:
        ds_dir = self.dataset_root / self.dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = ds_dir / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata_path, index=False)
        return metadata_path

    def test_retrieve_filters_by_regex_and_returns_tables(self):
        self._write_csv("users", pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}))
        self._write_csv("orders", pd.DataFrame({"order_id": [10], "total": [99]}))
        self.db_api.ingest_dataset(self.dataset_name, self.dataset_dir.as_posix())

        self._write_metadata(
            [
                {"table_name": "users", "description": "Users table"},
                {"table_name": "orders", "description": "Orders table"},
            ]
        )

        results = self.enumerator.retrieve(
            r"^us.*", self.dataset_name, k=10, sample_only=False
        )

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        doc = results[0]
        self.assertIsInstance(doc, Table)
        self.assertEqual(doc.doc_id, "users")
        self.assertEqual(doc.retriever_type, RetrieverType.ENUMERATOR)
        self.assertEqual(doc.metadata.get("dataset_name"), self.dataset_name)
        self.assertEqual(doc.metadata.get("description"), "Users table")
        self.assertEqual(len(doc.content), 2)

    def test_retrieve_sample_only_defaults_to_5(self):
        df = pd.DataFrame({"id": list(range(10))})
        self._write_csv("users", df)
        self.db_api.ingest_dataset(self.dataset_name, self.dataset_dir.as_posix())

        # No metadata.csv -> description should be empty string.
        results = self.enumerator.retrieve(
            r"^users$", self.dataset_name, k=1, sample_only=True
        )
        self.assertEqual(len(results), 1)
        doc = results[0]

        self.assertIsInstance(doc, Table)
        self.assertEqual(doc.doc_id, "users")
        self.assertLessEqual(len(doc.content), 5)
        self.assertEqual(doc.metadata.get("description"), "")
        self.assertEqual(doc.metadata.get("dataset_name"), self.dataset_name)

    def test_retrieve_sample_only_custom_limit(self):
        df = pd.DataFrame({"id": list(range(10))})
        self._write_csv("users", df)
        self.db_api.ingest_dataset(self.dataset_name, self.dataset_dir.as_posix())

        results = self.enumerator.retrieve(
            r"^users$", self.dataset_name, k=1, sample_only=True, sample_size=2
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].content), 2)

    def test_retrieve_description_empty_when_table_missing_in_metadata(self):
        self._write_csv("users", pd.DataFrame({"id": [1]}))
        self.db_api.ingest_dataset(self.dataset_name, self.dataset_dir.as_posix())

        # metadata exists but doesn't include users
        self._write_metadata([{"table_name": "orders", "description": "Orders"}])

        results = self.enumerator.retrieve(
            r"^users$", self.dataset_name, k=1, sample_only=False
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata.get("description"), "")


if __name__ == "__main__":
    unittest.main()
