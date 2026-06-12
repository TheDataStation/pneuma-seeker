import logging
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config


class TestPneumaDBInit(unittest.TestCase):
    """Tests for PneumaDB initialization."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_with_default_paths(self):
        """Test initialization with default paths."""
        db = PneumaDB(logger=self.logger, config=self.config)
        self.assertIsNotNone(db.dataset_db_path)
        self.assertIsNotNone(db.workspace_db_path)
        db.close_all_connections()

    def test_init_with_custom_paths(self):
        """Test initialization with custom paths."""
        dataset_path = os.path.join(self.tmpdir, "custom_datasets")
        workspace_path = os.path.join(self.tmpdir, "custom_workspaces")
        db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=dataset_path,
            workspace_db_path=workspace_path,
        )
        self.assertTrue(str(db.dataset_db_path).endswith(dataset_path))
        self.assertTrue(str(db.workspace_db_path).endswith(workspace_path))
        db.close_all_connections()


    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        dataset_path = os.path.join(self.tmpdir, "datasets")
        workspace_path = os.path.join(self.tmpdir, "workspaces")
        db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=dataset_path,
            workspace_db_path=workspace_path,
        )
        self.assertTrue(os.path.exists(dataset_path))
        self.assertTrue(os.path.exists(workspace_path))
        db.close_all_connections()


if __name__ == "__main__":
    unittest.main()
