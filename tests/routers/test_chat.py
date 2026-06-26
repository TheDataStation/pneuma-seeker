import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.routers import chat
from pneuma_seeker.routers.auth import get_current_user
from pneuma_seeker.services.core.conductor.models import ConductorResponse, ConductorResponseType
from pneuma_seeker.services.db.users.models import UserRecord


class TestChatRouter(unittest.TestCase):
    """Unit tests covering chat streaming execution loops, states, scripts and file packaging operations."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(chat.router)

        # Mock user injection
        self.fake_user = UserRecord(
            user_id="user_999",
            email="developer@pneuma.io",
            username="developer@pneuma.io",
            group_id="developer",
            is_active=True,
        )
        self.app.dependency_overrides[get_current_user] = lambda: self.fake_user

        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    @patch("pneuma_seeker.routers.chat.session_manager")
    @patch("pneuma_seeker.routers.chat.config")
    def test_chat_stream_success(self, mock_config, mock_session_manager):
        """Tests that a regular streaming chat request functions cleanly and yields formatted tokens."""
        # Arrange
        mock_config.ENABLE_MEMORY_PROFILING = False

        mock_chat_session = MagicMock()
        # Mock chat loop returning distinct output actions
        mock_chat_session.chat.return_value = [
            ConductorResponse(ConductorResponseType.LOG, "Analyzing structures"),
            ConductorResponse(ConductorResponseType.FINAL_RESPONSE, "SELECT * FROM alpha"),
            ConductorResponse(ConductorResponseType.DONE, ""),
        ]
        mock_session_manager.get_chat_session.return_value = mock_chat_session

        payload = {
            "chat_id": "session_001",
            "dataset_name": "test_dataset",
            "message": "Find missing nodes",
            "files": [],
        }

        # Act
        response = self.client.post("/chat/", json=payload)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/x-ndjson")

        # Parse NDJSON chunk lines
        lines = response.text.strip().split("\n")
        parsed_chunks = [json.loads(line) for line in lines]

        # We expect initial connection log + 3 iteration tokens
        self.assertEqual(len(parsed_chunks), 4)
        self.assertEqual(parsed_chunks[0]["sender"], "log")
        self.assertEqual(parsed_chunks[1]["sender"], "log")
        self.assertEqual(parsed_chunks[2]["sender"], "assistant")
        self.assertEqual(parsed_chunks[3]["sender"], "done")

        mock_chat_session.persist_session.assert_called_once()

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_chat_missing_message_returns_400(self, mock_session_manager):
        """Validates that requests with missing user queries fail immediately with 400 Bad Request."""
        # Act
        response = self.client.post("/chat/", json={"chat_id": "session_001", "dataset_name": "test_dataset"})

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Missing user message")

    @patch("pneuma_seeker.routers.chat.session_manager")
    @patch("pneuma_seeker.routers.chat.config")
    def test_execute_code_success(self, mock_config, mock_session_manager):
        """Tests execution layer falling back into action execution blocks when query evaluation misses."""
        # Arrange
        mock_config.TABLE_MAX_ROWS_DISPLAY = 50
        mock_conductor = MagicMock()

        # Simulate empty evaluation results initially to force execution paths
        mock_conductor.db_api.execute_query.return_value = []
        fake_df = pd.DataFrame({"node_id": [1, 2], "val": ["A", "B"]})
        mock_conductor.action_set.execute_code.return_value = fake_df

        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        # Act
        response = self.client.get("/chat/execute_code/session_001")

        # Assert
        self.assertEqual(response.status_code, 200)
        mock_conductor.action_set.execute_code.assert_called_once()

    @patch("pneuma_seeker.routers.chat.session_manager")
    @patch("pneuma_seeker.routers.chat.config")
    def test_get_state_success(self, mock_config, mock_session_manager):
        """Tests gathering system states alongside visual structural graph descriptions."""
        # Arrange
        mock_config.TABLE_MAX_ROWS_DISPLAY = 10
        mock_conductor = MagicMock()

        mock_conductor.state.get_current_state_instance.return_value = {
            "active_views": []
        }
        mock_conductor.state.is_T_materialized = True
        mock_conductor.materializer.prov_graph.get_graph_explanation.return_value = [
            "Step 1",
            "Step 2",
        ]

        # Mock structural tables
        mock_doc = MagicMock()
        mock_doc.doc_id = "table_alpha"
        mock_doc.content = pd.DataFrame({"x": [1]})
        mock_conductor.retrieved_tables = [mock_doc]

        mock_chat_session = MagicMock()
        mock_chat_session.conductor = mock_conductor
        mock_chat_session.dataset_name = "test_dataset"
        mock_session_manager.get_chat_session.return_value = mock_chat_session

        # Act
        response = self.client.get("/chat/state/session_001")

        # Assert
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("state", body)
        self.assertEqual(body["prov_steps"], ["Step 1", "Step 2"])
        self.assertIn("table_alpha", body["retrieved_tables"])
        self.assertEqual(body["dataset_name"], "test_dataset")

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_get_chat_history(self, mock_session_manager):
        """Validates retrieval of existing message tracking matrices."""
        # Arrange
        mock_chat_session = MagicMock()
        mock_chat_session.messages = [{"role": "user", "content": "hello"}]
        mock_chat_session.dataset_name = None
        mock_session_manager.get_chat_session.return_value = mock_chat_session

        # Act
        response = self.client.get("/chat/chat/session_001/history")

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["messages"], [{"role": "user", "content": "hello"}]
        )

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_get_target_views_not_found(self, mock_session_manager):
        """Verifies downloading target metrics throws 404 when structural dimensions have missing targets."""
        # Arrange
        mock_conductor = MagicMock()
        mock_conductor.state.T = None  # No tables generated
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        # Act
        response = self.client.get("/chat/target_views/session_001")

        # Assert
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "No target tables found")

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_get_e2e_script_success(self, mock_session_manager):
        """Checks script emission pipelines generation tasks."""
        # Arrange
        mock_chat_session = MagicMock()
        mock_chat_session.conductor.materializer.prov_graph.get_graph_code.return_value = (
            "print('Pneuma Code')"
        )
        mock_session_manager.get_chat_session.return_value = mock_chat_session

        # Act
        response = self.client.get("/chat/e2e_script/session_001")

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "print('Pneuma Code')")
        self.assertIn(
            "attachment; filename=materializer_user_999_session_001.py",
            response.headers["content-disposition"],
        )

    @patch("pneuma_seeker.routers.chat.pneuma_db")
    def test_list_sessions_delegates_to_search_when_query_provided(self, mock_pneuma_db):
        """Validates that a query parameter routes to search_chat_sessions instead of get_user_chat_sessions."""
        mock_pneuma_db.search_chat_sessions.return_value = {
            "chats": [{"id": "chat_1", "title": "Revenue analysis", "lastActive": "2024-01-01T00:00:00Z"}],
            "has_more": False,
            "next_offset": None,
        }

        response = self.client.get("/chat/sessions?query=revenue&limit=5&offset=0")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["chats"]), 1)
        mock_pneuma_db.search_chat_sessions.assert_called_once_with(
            user_id="user_999",
            query="revenue",
            limit=5,
            offset=0,
        )
        mock_pneuma_db.get_user_chat_sessions.assert_not_called()

    @patch("pneuma_seeker.routers.chat.pneuma_db")
    def test_list_sessions_uses_get_sessions_without_query(self, mock_pneuma_db):
        """Validates that without a query parameter, list_chats uses get_user_chat_sessions."""
        mock_pneuma_db.get_user_chat_sessions.return_value = {
            "chats": [],
            "has_more": False,
            "next_offset": None,
        }

        response = self.client.get("/chat/sessions")

        self.assertEqual(response.status_code, 200)
        mock_pneuma_db.get_user_chat_sessions.assert_called_once()
        mock_pneuma_db.search_chat_sessions.assert_not_called()

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_get_provenance_nodes_success(self, mock_session_manager):
        """Tests clean structural decomposition maps out dependency graphs properly."""
        # Arrange
        mock_node = MagicMock()
        mock_node.id = "node_0"
        mock_node.source_retriever.value = "RetrieverAlpha"
        mock_node.python_code = "df = load()"
        mock_node.description = "Root layer"
        mock_node.parents = []
        mock_node.children = []

        mock_prov_graph = MagicMock()
        mock_prov_graph.nodes = {"node_0": mock_node}
        mock_session_manager.get_chat_session.return_value.conductor.materializer.prov_graph = (
            mock_prov_graph
        )

        # Act
        response = self.client.get("/chat/provenance_nodes/session_001")

        # Assert
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["node_count"], 1)
        self.assertEqual(body["nodes"][0]["id"], "node_0")


    @patch("pneuma_seeker.routers.chat.rmtree")
    @patch("pneuma_seeker.routers.chat.session_manager")
    @patch("pneuma_seeker.routers.chat.pneuma_db")
    def test_delete_chat_session_success(self, mock_pneuma_db, mock_session_manager, mock_rmtree):
        """Tests that a successful deletion evicts the in-memory session, calls prepare_chat_deletion, and schedules directory removal."""
        from pathlib import Path

        fake_dir = Path("/tmp/user_999/session_del")
        mock_pneuma_db.prepare_chat_deletion.return_value = fake_dir

        response = self.client.delete("/chat/session_del")

        self.assertEqual(response.status_code, 200)
        self.assertIn("session_del", response.json()["detail"])
        mock_pneuma_db.prepare_chat_deletion.assert_called_once_with(
            user_id="user_999",
            chat_id="session_del",
        )
        mock_session_manager.evict_chat_session.assert_called_once_with("user_999", "session_del")
        mock_rmtree.assert_called_once_with(fake_dir)

    @patch("pneuma_seeker.routers.chat.pneuma_db")
    def test_delete_chat_session_not_found_returns_404(self, mock_pneuma_db):
        """Tests that deleting a non-existent chat session returns 404."""
        mock_pneuma_db.prepare_chat_deletion.side_effect = FileNotFoundError("not found")

        response = self.client.delete("/chat/ghost_session")

        self.assertEqual(response.status_code, 404)
        self.assertIn("ghost_session", response.json()["detail"])

    @patch("pneuma_seeker.routers.chat.pneuma_db")
    def test_delete_chat_session_unexpected_error_returns_500(self, mock_pneuma_db):
        """Tests that an unexpected error during deletion preparation returns 500."""
        mock_pneuma_db.prepare_chat_deletion.side_effect = RuntimeError("db failure")

        response = self.client.delete("/chat/session_fail")

        self.assertEqual(response.status_code, 500)


class TestQueryTableEndpoint(unittest.TestCase):
    """Unit tests for the GET /chat/table_query/{chat_id} endpoint."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.fake_user = UserRecord(
            user_id="user_999",
            email="developer@pneuma.io",
            username="developer@pneuma.io",
            group_id="developer",
            is_active=True,
        )
        self.app.dependency_overrides[get_current_user] = lambda: self.fake_user
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def _mock_db_side_effects(self, columns, total_count, rows_data):
        """Returns the three DataFrames that execute_query yields in order."""
        cols_df = pd.DataFrame(columns=columns)
        count_df = pd.DataFrame({"cnt": [total_count]})
        rows_df = pd.DataFrame(rows_data)
        return [cols_df, count_df, rows_df]

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_basic_success(self, mock_session_manager):
        """Returns rows, total_count, and columns on a straightforward query."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["id", "name", "city"],
            total_count=100,
            rows_data={"id": [1, 2], "name": ["Alice", "Bob"], "city": ["NY", "LA"]},
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        response = self.client.get("/chat/table_query/session_001?table_id=users")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_count"], 100)
        self.assertEqual(body["columns"], ["id", "name", "city"])
        self.assertEqual(len(body["rows"]), 2)
        self.assertEqual(body["rows"][0]["name"], "Alice")

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_invalid_table_id_returns_400(self, mock_session_manager):
        """Rejects table_id values that contain characters outside [a-zA-Z0-9_]."""
        response = self.client.get(
            "/chat/table_query/session_001?table_id=bad;drop+table+users"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid table_id", response.json()["detail"])

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_session_not_found_returns_404(self, mock_session_manager):
        """Returns 404 when the chat session does not exist."""
        mock_session_manager.get_chat_session.side_effect = Exception("session not found")
        response = self.client.get("/chat/table_query/ghost_session?table_id=users")
        self.assertEqual(response.status_code, 404)
        self.assertIn("ghost_session", response.json()["detail"])

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_missing_table_returns_404(self, mock_session_manager):
        """Returns 404 when the table does not exist in the session workspace."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = Exception("table does not exist")
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        response = self.client.get("/chat/table_query/session_001?table_id=nonexistent")

        self.assertEqual(response.status_code, 404)
        self.assertIn("nonexistent", response.json()["detail"])

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_uses_ilike_for_search(self, mock_session_manager):
        """Verifies the search parameter produces ILIKE (not LIKE) in the SQL."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["name"],
            total_count=1,
            rows_data={"name": ["Illinois"]},
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get("/chat/table_query/session_001?table_id=states&search=illi")

        calls = mock_conductor.db_api.execute_query.call_args_list
        # Second call is the COUNT query, which carries the WHERE clause
        count_sql = calls[1].args[2]
        self.assertIn("ILIKE", count_sql)
        self.assertNotIn(" LIKE ", count_sql)
        # The search pattern is passed as a parameter, not interpolated
        count_params = calls[1].args[3]
        self.assertIn("%illi%", count_params)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_with_dataset_name_qualifies_reference(self, mock_session_manager):
        """When dataset_name is provided, the SQL uses "dataset"."table" qualified form."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["id"],
            total_count=5,
            rows_data={"id": [1]},
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        response = self.client.get(
            "/chat/table_query/session_001?table_id=reports&dataset_name=csn_2024"
        )

        self.assertEqual(response.status_code, 200)
        mock_conductor.db_api.link_dataset_tables.assert_called_once_with(
            "user_999", "session_001", "csn_2024"
        )
        cols_sql = mock_conductor.db_api.execute_query.call_args_list[0].args[2]
        self.assertIn('"csn_2024"."reports"', cols_sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_dataset_name_strips_quotes(self, mock_session_manager):
        """A dataset_name containing quotes has the quote chars removed before interpolation."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["id"], total_count=0, rows_data={"id": []}
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        # dataset_name with an embedded quote: 'ds"injected' → sanitized to 'dsinjected'
        self.client.get(
            '/chat/table_query/session_001?table_id=t&dataset_name=ds"injected'
        )

        cols_sql = mock_conductor.db_api.execute_query.call_args_list[0].args[2]
        # The outer quotes are the DuckDB identifier delimiters; no extra " should appear inside
        self.assertNotIn('ds"injected', cols_sql)
        # The sanitized form should be present
        self.assertIn('"dsinjected"', cols_sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_order_by_reflected_in_sql(self, mock_session_manager):
        """order_by and order_dir params appear in the final SELECT SQL."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["name"],
            total_count=2,
            rows_data={"name": ["Bob", "Alice"]},
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get(
            "/chat/table_query/session_001?table_id=users&order_by=name&order_dir=desc"
        )

        calls = mock_conductor.db_api.execute_query.call_args_list
        rows_sql = calls[2].args[2]
        self.assertIn('"name"', rows_sql)
        self.assertIn("DESC", rows_sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_invalid_order_dir_defaults_to_asc(self, mock_session_manager):
        """An invalid order_dir value silently falls back to ASC."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["id"], total_count=1, rows_data={"id": [1]}
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get(
            "/chat/table_query/session_001?table_id=t&order_by=id&order_dir=DROP+TABLE"
        )

        rows_sql = mock_conductor.db_api.execute_query.call_args_list[2].args[2]
        self.assertIn("ASC", rows_sql)
        self.assertNotIn("DROP", rows_sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_limit_clamped_to_500(self, mock_session_manager):
        """limit values above 500 are clamped to 500."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["id"], total_count=0, rows_data={"id": []}
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get("/chat/table_query/session_001?table_id=t&limit=9999")

        rows_sql = mock_conductor.db_api.execute_query.call_args_list[2].args[2]
        self.assertIn("LIMIT 500", rows_sql)
        self.assertNotIn("LIMIT 9999", rows_sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_query_table_pagination_offset_in_sql(self, mock_session_manager):
        """offset param is reflected in the SELECT SQL."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = self._mock_db_side_effects(
            columns=["id"], total_count=200, rows_data={"id": [51]}
        )
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get("/chat/table_query/session_001?table_id=t&offset=50")

        rows_sql = mock_conductor.db_api.execute_query.call_args_list[2].args[2]
        self.assertIn("OFFSET 50", rows_sql)


class TestDownloadTableEndpoint(unittest.TestCase):
    """Unit tests for GET /chat/table_download/{chat_id} streaming CSV export."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.fake_user = UserRecord(
            user_id="user_999",
            email="developer@pneuma.io",
            username="developer@pneuma.io",
            group_id="developer",
            is_active=True,
        )
        self.app.dependency_overrides[get_current_user] = lambda: self.fake_user
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_returns_csv(self, mock_session_manager):
        """Successful request streams CSV with correct content-type and disposition."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = [
            pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
            pd.DataFrame(),
        ]
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        response = self.client.get("/chat/table_download/session_001?table_id=users")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        self.assertIn('filename="users.csv"', response.headers["content-disposition"])
        text = response.text
        self.assertIn("id", text)
        self.assertIn("Alice", text)
        self.assertIn("Bob", text)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_csv_has_header_row(self, mock_session_manager):
        """The first line of the streamed CSV contains column names."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = [
            pd.DataFrame({"col_a": [10], "col_b": ["x"]}),
            pd.DataFrame(),
        ]
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        response = self.client.get("/chat/table_download/session_001?table_id=t")

        first_line = response.text.splitlines()[0]
        self.assertIn("col_a", first_line)
        self.assertIn("col_b", first_line)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_invalid_table_id_returns_400(self, mock_session_manager):
        """table_id with disallowed characters returns 400 before any DB access."""
        mock_session_manager.get_chat_session.return_value.conductor = MagicMock()

        response = self.client.get("/chat/table_download/session_001?table_id=bad-id!")

        self.assertEqual(response.status_code, 400)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_session_not_found_returns_404(self, mock_session_manager):
        """Missing chat session returns 404."""
        mock_session_manager.get_chat_session.side_effect = Exception("no session")

        response = self.client.get("/chat/table_download/missing?table_id=users")

        self.assertEqual(response.status_code, 404)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_with_dataset_name_qualifies_reference(self, mock_session_manager):
        """dataset_name is used to qualify the table reference and link_dataset_tables is called."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = [
            pd.DataFrame({"id": [1]}),
            pd.DataFrame(),
        ]
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get(
            "/chat/table_download/session_001?table_id=reports&dataset_name=csn_2024"
        )

        mock_conductor.db_api.link_dataset_tables.assert_called_once_with(
            "user_999", "session_001", "csn_2024"
        )
        sql = mock_conductor.db_api.execute_query.call_args_list[0].args[2]
        self.assertIn('"csn_2024"."reports"', sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_dataset_name_strips_quotes(self, mock_session_manager):
        """Quotes in dataset_name are stripped before interpolation."""
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = [
            pd.DataFrame({"id": [1]}),
            pd.DataFrame(),
        ]
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        self.client.get(
            '/chat/table_download/session_001?table_id=t&dataset_name=ds"injected'
        )

        sql = mock_conductor.db_api.execute_query.call_args_list[0].args[2]
        self.assertNotIn('ds"injected', sql)
        self.assertIn('"dsinjected"', sql)

    @patch("pneuma_seeker.routers.chat.session_manager")
    def test_download_table_streams_multiple_chunks(self, mock_session_manager):
        """When the first chunk is full (CHUNK rows), a second DB call is made for the next chunk."""
        chunk_size = 5_000
        mock_conductor = MagicMock()
        mock_conductor.db_api.execute_query.side_effect = [
            pd.DataFrame({"id": list(range(chunk_size))}),  # full chunk → another call expected
            pd.DataFrame({"id": [chunk_size]}),              # partial chunk → stop
            pd.DataFrame(),                                   # safety: never reached
        ]
        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        response = self.client.get("/chat/table_download/session_001?table_id=big_table")

        self.assertEqual(response.status_code, 200)
        # Two data chunks means two execute_query calls (plus no third)
        self.assertEqual(mock_conductor.db_api.execute_query.call_count, 2)
        # Second call should use OFFSET equal to chunk_size
        second_sql = mock_conductor.db_api.execute_query.call_args_list[1].args[2]
        self.assertIn(f"OFFSET {chunk_size}", second_sql)


if __name__ == "__main__":
    unittest.main()
