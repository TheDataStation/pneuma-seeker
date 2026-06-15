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
            "LOG: Analyzing structures",
            "SELECT * FROM alpha",
            "DONE",
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

        mock_session_manager.get_chat_session.return_value.conductor = mock_conductor

        # Act
        response = self.client.get("/chat/state/session_001")

        # Assert
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("state", body)
        self.assertEqual(body["prov_steps"], ["Step 1", "Step 2"])
        self.assertIn("table_alpha", body["retrieved_tables"])

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


if __name__ == "__main__":
    unittest.main()
