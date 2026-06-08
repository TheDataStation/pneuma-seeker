import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.db.main import PneumaDB
from pneuma_seeker.shared.config import Config


class TestWorkspaceConnections(unittest.TestCase):
    """Tests for workspace database connection management."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=(Path(self.tmpdir) / "datasets").as_posix(),
            workspace_db_path=(Path(self.tmpdir) / "workspaces").as_posix(),
        )

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_workspace_connection_creates_db_file_and_tables(self):
        """Tests that requesting a workspace connection creates the expected database file and tables."""
        con = self.db.get_ws_db_connection("user_1", "chat_1")

        db_file = self.db.workspace_db_path / "user_1" / "chat_1" / "ws.db"
        table_names = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

        self.assertTrue(db_file.exists())
        self.assertIn("chat_history", table_names)
        self.assertIn("conductor_state", table_names)
        self.assertIn("provenance_nodes", table_names)
        self.assertIn("documents", table_names)
        self.assertIn("document_metadata", table_names)
        self.assertIn("state_document_roles", table_names)
        self.assertIn("provenance_edges", table_names)

    def test_get_workspace_connection_reuses_cached_connection(self):
        """Tests that requesting a workspace connection multiple times returns the same cached connection instance."""
        con1 = self.db.get_ws_db_connection("user_1", "chat_1")
        con2 = self.db.get_ws_db_connection("user_1", "chat_1")

        self.assertIs(con1, con2)
        self.assertEqual(len(self.db._conn_cache), 1)

    def test_close_workspace_connection_removes_cached_connection(self):
        """Tests that closing a workspace connection removes it from the cache and subsequent requests create a new connection."""
        con1 = self.db.get_ws_db_connection("user_1", "chat_1")

        self.db.close_workspace_connection("user_1", "chat_1")
        con2 = self.db.get_ws_db_connection("user_1", "chat_1")

        self.assertIsNot(con1, con2)
        self.assertEqual(len(self.db._conn_cache), 1)

    def test_close_all_connections_clears_cache(self):
        """Tests that closing all connections clears the entire connection cache."""
        self.db.get_ws_db_connection("user_1", "chat_1")
        self.db.get_ws_db_connection("user_1", "chat_2")

        self.db.close_all_connections()

        self.assertEqual(self.db._conn_cache, {})


class TestWorkspaceQueries(unittest.TestCase):
    """Tests for executing and persisting workspace data."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=(Path(self.tmpdir) / "datasets").as_posix(),
            workspace_db_path=(Path(self.tmpdir) / "workspaces").as_posix(),
        )

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_execute_query_returns_dataframe(self):
        """Tests that executing a SQL query returns a DataFrame with the expected results."""
        result = self.db.execute_query(
            "user_1",
            "chat_1",
            "SELECT ?::INTEGER AS value",
            (42,),
        )

        self.assertEqual(result.iloc[0]["value"], 42)

    def test_persist_df_creates_table(self):
        """Tests that persisting a DataFrame creates a new table in the workspace database and allows querying the data."""
        df = pd.DataFrame([{"id": 1, "name": "Alice"}])

        self.db.persist_df("user_1", "chat_1", df, "people", overwrite_content=True)
        result = self.db.execute_query("user_1", "chat_1", 'SELECT * FROM "people"')

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_persist_df_respects_overwrite_flag(self):
        """Tests that persisting a DataFrame with overwrite_content=False does not modify existing table content, while overwrite_content=True does."""
        first = pd.DataFrame([{"id": 1, "name": "Alice"}])
        second = pd.DataFrame([{"id": 2, "name": "Bob"}])

        self.db.persist_df("user_1", "chat_1", first, "people", overwrite_content=True)
        self.db.persist_df("user_1", "chat_1", second, "people", overwrite_content=False)
        result = self.db.execute_query(
            "user_1",
            "chat_1",
            'SELECT * FROM "people" ORDER BY id',
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_persist_df_overwrites_when_requested(self):
        """Tests that persisting a DataFrame with overwrite_content=True replaces existing table content."""
        first = pd.DataFrame([{"id": 1, "name": "Alice"}])
        second = pd.DataFrame([{"id": 2, "name": "Bob"}])

        self.db.persist_df("user_1", "chat_1", first, "people", overwrite_content=True)
        self.db.persist_df("user_1", "chat_1", second, "people", overwrite_content=True)
        result = self.db.execute_query("user_1", "chat_1", 'SELECT * FROM "people"')

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Bob")

    def test_register_temporary_df_allows_querying_without_persisting(self):
        """Tests that registering a temporary DataFrame allows it to be queried in the workspace without being persisted as a table, and does not affect the database schema."""
        df = pd.DataFrame([{"id": 1, "name": "Alice"}])

        self.db.register_temporary_df("user_1", "chat_1", df, "people_preview")
        result = self.db.execute_query(
            "user_1",
            "chat_1",
            'SELECT * FROM "people_preview"',
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_register_temporary_df_rejects_existing_persistent_table(self):
        """Tests that registering a temporary DataFrame is rejected if a persistent table with the same name exists."""
        df = pd.DataFrame([{"id": 1, "name": "Alice"}])
        self.db.persist_df("user_1", "chat_1", df, "people", overwrite_content=True)

        with self.assertRaises(ValueError):
            self.db.register_temporary_df("user_1", "chat_1", df, "people")


class TestWorkspaceSessionPersistence(unittest.TestCase):
    """Tests for persisted chat history and append-only state snapshots."""

    def setUp(self):
        self.config = Config()
        self.config.ENABLE_FINE_GRAINED_STATE_CHANGE_TRACKING = False
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=(Path(self.tmpdir) / "datasets").as_posix(),
            workspace_db_path=(Path(self.tmpdir) / "workspaces").as_posix(),
        )

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _persist_turn(self, user_input: str, assistant_response: str, script: str):
        """Helper to persist a single turn of chat history and conductor state."""
        state = ConductorState()
        state.S = script
        self.db.persist_session(
            "user_1",
            "chat_1",
            user_input,
            assistant_response,
            state,
            ProvenanceGraph(self.logger),
            [],
            [],
        )

    def test_persist_session_keeps_all_chat_history(self):
        """Tests that persisting multiple turns of a chat session retains the full history of user inputs and assistant responses in the correct order."""
        self._persist_turn("hello", "hi there", "x = 1")
        self._persist_turn("next", "answer", "x = 2")

        history = self.db.load_chat_history("user_1", "chat_1")

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_persist_session_keeps_every_state_even_when_switch_is_false(self):
        """Tests that persisting multiple turns of a chat session retains every state snapshot in the conductor_state table, even when fine-grained state change tracking is disabled."""
        self._persist_turn("hello", "hi there", "x = 1")
        self._persist_turn("next", "answer", "x = 2")

        con = self.db.get_ws_db_connection("user_1", "chat_1")
        state_rows = con.execute(
            """
            SELECT python_script
            FROM conductor_state
            ORDER BY creation_timestamp ASC
            """
        ).fetchall()
        chat_count = con.execute("SELECT COUNT(*) FROM chat_history").fetchone()
        assert chat_count is not None
        chat_count = chat_count[0]

        self.assertEqual([row[0] for row in state_rows], ["x = 1", "x = 2"])
        self.assertEqual(chat_count, 4)

    def test_load_session_returns_latest_state_and_full_history(self):
        """Tests that loading a session returns the full chat history and the latest state snapshot, allowing reconstruction of the entire session state from the initial state and all subsequent changes."""
        self._persist_turn("hello", "hi there", "x = 1")
        self._persist_turn("next", "answer", "x = 2")

        history, state, *_ = self.db.load_session("user_1", "chat_1")

        self.assertEqual(state.S, "x = 2")
        self.assertEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()
