# tests/test_chat_session.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure the source directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pneuma_seeker.chat_session import ChatSession
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class DummyConductor:
    """A clean mock replacement for the Conductor."""
    def __init__(self, user_id, chat_id, config, logger, prov_graph, db_api, lm_api):
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.lm_api = lm_api
        
        # Mocks for properties referenced inside ChatSession
        self.prov_graph = MagicMock()
        self.prov_graph.nodes = {}
        self.state = MagicMock()
        self.retrieved_tables = []
        self.enumerated_tables = []
        self.web_search_result = None
        self.web_crawl_result = None
        self.join_paths = None

    def chat(self, last_content, interaction_history, external_data_paths):
        # Emulate the stream yielding a few chunks, ending with the complete message
        yield "resp:"
        yield f"resp:{last_content}"

    def set_prov_graph(self, provenance_graph):
        self.prov_graph = provenance_graph


class ChatSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        
        # Default mock return value for load_session
        self.db_api.load_session.return_value = (
            [],            # messages
            MagicMock(),   # conductor_state
            MagicMock(),   # provenance_graph
            [],            # retrieved_tables
            [],            # enumerated_tables
            None,          # web_search_result
            None,          # web_crawl_result
            None,          # join_paths
            None,
        )

    @patch("pneuma_seeker.chat_session.Conductor", new=DummyConductor)
    @patch("pneuma_seeker.chat_session.ProvenanceGraph")
    def test_chat_yields_conductor_responses_and_done(self, mock_prov_graph):
        cs = ChatSession(
            "u1", "c1", self.cfg, self.logger, self.db_api, self.lm_api
        )
        
        # Execute chat streaming
        out = list(cs.chat(latest_user_message="hello"))
        
        # Should include chunks from DummyConductor and the final "DONE"
        self.assertIn("resp:", out)
        self.assertIn("resp:hello", out)
        self.assertIn("DONE", out)
        
        # Verify message history contains the final accumulated assistant message
        self.assertEqual(cs.messages[0]["role"], "user")
        self.assertEqual(cs.messages[0]["content"], "hello")
        self.assertEqual(cs.messages[1]["role"], "assistant")
        self.assertEqual(cs.messages[1]["content"], "resp:hello")

    @patch("pneuma_seeker.chat_session.Conductor", new=DummyConductor)
    @patch("pneuma_seeker.chat_session.ProvenanceGraph")
    def test_chat_with_loaded_history_appends_correctly(self, mock_prov_graph):
        self.db_api.load_session.return_value = (
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
            ],
            MagicMock(),
            MagicMock(),
            [],
            [],
            None,
            None,
            None,
            None,
        )

        cs = ChatSession(
            "u1", "c1", self.cfg, self.logger, self.db_api, self.lm_api
        )
        
        out = list(cs.chat(latest_user_message="second"))

        self.assertIn("resp:second", out)
        self.assertEqual(
            cs.messages,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "resp:second"},
            ],
        )

    @patch("pneuma_seeker.chat_session.Conductor", new=DummyConductor)
    @patch("pneuma_seeker.chat_session.ProvenanceGraph")
    def test_chat_raises_value_error_on_empty_message(self, mock_prov_graph):
        cs = ChatSession(
            "u1", "c1", self.cfg, self.logger, self.db_api, self.lm_api
        )
        with self.assertRaises(ValueError):
            list(cs.chat(latest_user_message="   "))

    @patch("pneuma_seeker.chat_session.Conductor", new=DummyConductor)
    @patch("pneuma_seeker.chat_session.ProvenanceGraph")
    def test_persist_session_extracts_correct_last_messages(self, mock_prov_graph):
        cs = ChatSession(
            "u1", "c1", self.cfg, self.logger, self.db_api, self.lm_api
        )
        
        # Artificially populate messages history
        cs.messages = [
            LLMMessage(role="user", content="old query"),
            LLMMessage(role="assistant", content="old answer"),
            LLMMessage(role="user", content="latest query"),
            LLMMessage(role="assistant", content="latest answer"),
        ]
        
        cs.persist_session("dataset_name")
        
        # Verify that db_api.persist_session was called using the parsed history strings
        self.db_api.persist_session.assert_called_once()
        args, kwargs = self.db_api.persist_session.call_args
        
        # positional args match: (user_id, chat_id, dataset_name, last_user_input, last_system_response, ...)
        self.assertEqual(args[0], "u1")
        self.assertEqual(args[1], "c1")
        self.assertEqual(args[2], "dataset_name")
        self.assertEqual(args[3], "latest query")
        self.assertEqual(args[4], "latest answer")


if __name__ == "__main__":
    unittest.main()
