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
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config


class TestWorkspaceSessionDiscovery(unittest.TestCase):
    """Tests for discovering, reading metadata, and paginating historical chat sessions."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        
        # PneumaDB wraps WorkspaceManager internally
        self.db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=(Path(self.tmpdir) / "datasets").as_posix(),
            workspace_db_path=(Path(self.tmpdir) / "workspaces").as_posix(),
        )

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mock_session(self, user_id: str, chat_id: str, first_message: str):
        """Helper to explicitly seed a workspace session with historical messages."""
        state = ConductorState()
        self.db.persist_session(
            user_id=user_id,
            chat_id=chat_id,
            dataset_name="test_dataset",
            new_user_input=first_message,
            new_system_response="Understood, executing processing pipeline.",
            conductor_state=state,
            provenance_graph=ProvenanceGraph(self.logger),
            retrieved_tables=[],
            enumerated_tables=[],
        )
        # Force close the connection so it does not reside in the active cache
        self.db.close_workspace_connection(user_id, chat_id)

    def test_get_user_chat_sessions_returns_metadata_sorted_by_recency(self):
        """Tests that session discovery extracts correct titles and sorts descending by activity."""
        user_id = "user_discovery_test"
        
        # 1. Seed historical sessions sequentially to guarantee distinct timestamps
        self._create_mock_session(user_id, "chat_old", "Short prompt")
        self._create_mock_session(user_id, "chat_new", "An exceptionally long prompt that exceeds twenty characters")

        # 2. Discover sessions
        result = self.db.get_user_chat_sessions(user_id=user_id, limit=10, offset=0)
        chats = result["chats"]

        # 3. Verify sorting, truncation, and structure
        self.assertEqual(len(chats), 2)
        self.assertFalse(result["has_more"])
        self.assertIsNone(result["next_offset"])

        # The most recently written session must appear first
        self.assertEqual(chats[0]["id"], "chat_new")
        self.assertEqual(chats[0]["title"], "An exceptionally lon...") # Verifies 20-character truncation logic
        self.assertTrue(chats[0]["lastActive"].endswith("Z") or "T" in chats[0]["lastActive"])

        # The older session must appear second
        self.assertEqual(chats[1]["id"], "chat_old")
        self.assertEqual(chats[1]["title"], "Short prompt")

    def test_get_user_chat_sessions_pagination_boundaries(self):
        """Tests that limit and offset windowing correctly segments results and reports remaining items."""
        user_id = "user_pagination_test"
        
        # Seed three distinct sessions
        self._create_mock_session(user_id, "chat_alpha", "First message text")
        self._create_mock_session(user_id, "chat_beta", "Second message text")
        self._create_mock_session(user_id, "chat_gamma", "Third message text")

        # Page 1: Request a slice of size 2
        page_1 = self.db.get_user_chat_sessions(user_id=user_id, limit=2, offset=0)
        self.assertEqual(len(page_1["chats"]), 2)
        self.assertTrue(page_1["has_more"])
        self.assertEqual(page_1["next_offset"], 2)

        # Page 2: Request the remainder using the next_offset
        page_2 = self.db.get_user_chat_sessions(user_id=user_id, limit=2, offset=page_1["next_offset"])
        self.assertEqual(len(page_2["chats"]), 1)
        self.assertFalse(page_2["has_more"])
        self.assertIsNone(page_2["next_offset"])

        # Ensure no duplicate elements crossed the pagination boundary
        page_1_ids = {session["id"] for session in page_1["chats"]}
        page_2_ids = {session["id"] for session in page_2["chats"]}
        self.assertTrue(page_1_ids.isdisjoint(page_2_ids))

    def test_get_user_chat_sessions_handles_empty_or_corrupted_directories(self):
        """Tests that scanning handles missing paths, empty directories, or invalid DB files without raising exceptions."""
        user_id = "user_empty_test"

        # Case A: User directory completely missing from disk
        missing_result = self.db.get_user_chat_sessions(user_id=user_id)
        self.assertEqual(missing_result["chats"], [])
        self.assertFalse(missing_result["has_more"])

        # Create the user directory for isolation
        user_dir = self.db.workspace_db_path / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Case B: Chat subdirectories exist but contain no ws.db files, or contain zero bytes
        empty_chat_dir = user_dir / "chat_empty"
        empty_chat_dir.mkdir(exist_ok=True)
        
        corrupted_chat_dir = user_dir / "chat_corrupted"
        corrupted_chat_dir.mkdir(exist_ok=True)
        (corrupted_chat_dir / "ws.db").write_text("INVALID_DUCKDB_BINARY_DATA")

        # Validate execution completes cleanly by passing over structural anomalies
        resilience_result = self.db.get_user_chat_sessions(user_id=user_id)
        self.assertEqual(resilience_result["chats"], [])
        self.assertFalse(resilience_result["has_more"])


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
        self.assertIn("session_metadata", table_names)

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
            "test_dataset",
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


class TestWorkspaceSessionSearch(unittest.TestCase):
    """Tests for ILIKE content search across chat sessions."""

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

    def _create_session(self, user_id: str, chat_id: str, user_input: str, response: str = "OK"):
        state = ConductorState()
        self.db.persist_session(
            user_id=user_id,
            chat_id=chat_id,
            dataset_name="test",
            new_user_input=user_input,
            new_system_response=response,
            conductor_state=state,
            provenance_graph=ProvenanceGraph(self.logger),
            retrieved_tables=[],
            enumerated_tables=[],
        )
        self.db.close_workspace_connection(user_id, chat_id)

    def test_search_returns_only_matching_chats(self):
        """Tests that search returns only chats whose messages contain the query string."""
        user_id = "user_search_basic"
        self._create_session(user_id, "chat_1", "Find revenue data")
        self._create_session(user_id, "chat_2", "Show customer list")
        self._create_session(user_id, "chat_3", "Revenue by region")

        result = self.db.workspace_manager.search_chat_sessions(user_id, "revenue")
        ids = {c["id"] for c in result["chats"]}

        self.assertEqual(len(result["chats"]), 2)
        self.assertIn("chat_1", ids)
        self.assertIn("chat_3", ids)
        self.assertNotIn("chat_2", ids)

    def test_search_is_case_insensitive(self):
        """Tests that ILIKE search ignores case."""
        user_id = "user_search_case"
        self._create_session(user_id, "chat_1", "Find REVENUE data")

        self.assertEqual(len(self.db.workspace_manager.search_chat_sessions(user_id, "revenue")["chats"]), 1)
        self.assertEqual(len(self.db.workspace_manager.search_chat_sessions(user_id, "REVENUE")["chats"]), 1)
        self.assertEqual(len(self.db.workspace_manager.search_chat_sessions(user_id, "Revenue")["chats"]), 1)

    def test_search_returns_empty_when_no_match(self):
        """Tests that search returns an empty list when no messages match the query."""
        user_id = "user_search_no_match"
        self._create_session(user_id, "chat_1", "Something completely different")

        result = self.db.workspace_manager.search_chat_sessions(user_id, "xyznonexistent")
        self.assertEqual(result["chats"], [])
        self.assertFalse(result["has_more"])
        self.assertIsNone(result["next_offset"])

    def test_search_matches_assistant_responses(self):
        """Tests that search also matches content in assistant responses, not only user messages."""
        user_id = "user_search_response"
        self._create_session(user_id, "chat_1", "What is 2+2?", response="The answer is four")

        result = self.db.workspace_manager.search_chat_sessions(user_id, "answer")
        self.assertEqual(len(result["chats"]), 1)
        self.assertEqual(result["chats"][0]["id"], "chat_1")

    def test_search_pagination(self):
        """Tests that search results are paginated using limit and offset."""
        user_id = "user_search_pagination"
        for i in range(3):
            self._create_session(user_id, f"chat_{i}", f"Sales data query {i}")

        page1 = self.db.workspace_manager.search_chat_sessions(user_id, "sales", limit=2, offset=0)
        self.assertEqual(len(page1["chats"]), 2)
        self.assertTrue(page1["has_more"])
        self.assertEqual(page1["next_offset"], 2)

        page2 = self.db.workspace_manager.search_chat_sessions(user_id, "sales", limit=2, offset=2)
        self.assertEqual(len(page2["chats"]), 1)
        self.assertFalse(page2["has_more"])
        self.assertIsNone(page2["next_offset"])

        all_ids = {c["id"] for c in page1["chats"]} | {c["id"] for c in page2["chats"]}
        self.assertEqual(len(all_ids), 3)

    def test_search_handles_missing_user_directory(self):
        """Tests that search returns an empty result when the user has no workspace directory."""
        result = self.db.workspace_manager.search_chat_sessions("user_nonexistent", "anything")
        self.assertEqual(result["chats"], [])
        self.assertFalse(result["has_more"])


class TestWorkspaceSessionDeletion(unittest.TestCase):
    """Tests for permanently deleting historical chat sessions and evacuating connections."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        
        # PneumaDB wraps WorkspaceManager internally
        self.db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=(Path(self.tmpdir) / "datasets").as_posix(),
            workspace_db_path=(Path(self.tmpdir) / "workspaces").as_posix(),
        )

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mock_session(self, user_id: str, chat_id: str, first_message: str):
        """Helper to explicitly seed a workspace session with historical messages."""
        state = ConductorState()
        self.db.persist_session(
            user_id=user_id,
            chat_id=chat_id,
            dataset_name="test_dataset",
            new_user_input=first_message,
            new_system_response="Understood, executing processing pipeline.",
            conductor_state=state,
            provenance_graph=ProvenanceGraph(self.logger),
            retrieved_tables=[],
            enumerated_tables=[],
        )

    def test_delete_chat_session_removes_directory_and_evicts_cache(self):
        """Tests that deleting a session removes its workspace directory from disk and evicts it from the connection cache."""
        user_id = "user_deletion_test"
        chat_id = "chat_to_delete"
        
        # 1. Establish session and ensure connection resides within the active cache
        self._create_mock_session(user_id, chat_id, "Session to be deleted")
        con = self.db.get_ws_db_connection(user_id, chat_id)
        
        chat_dir = self.db.workspace_db_path / user_id / chat_id
        db_file = chat_dir / "ws.db"
        
        self.assertTrue(db_file.exists())
        self.assertIn((user_id, chat_id), self.db._conn_cache)

        # 2. Execute deletion sequence via the underlying workspace manager
        # Note: If PneumaDB exposes a delete wrapper, use self.db.delete_chat_session here
        self.db.workspace_manager.delete_chat_session(user_id, chat_id)

        # 3. Structural and cache verifications
        self.assertFalse(chat_dir.exists())
        self.assertNotIn((user_id, chat_id), self.db._conn_cache)

    def test_delete_chat_session_raises_file_not_found_for_missing_session(self):
        """Tests that attempting to delete a non-existent session explicitly raises a FileNotFoundError."""
        user_id = "user_deletion_test"
        chat_id = "chat_missing"

        with self.assertRaises(FileNotFoundError):
            self.db.workspace_manager.delete_chat_session(user_id, chat_id)


if __name__ == "__main__":
    unittest.main()
