# tests/pneuma_seeker/core/conductor/test_main.py
import logging
import os
import sys
import tempfile
from pathlib import Path


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.main import Conductor
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    Text,
)


class ConductorTests(unittest.TestCase):
    def setUp(self):
        config = Config(".env.test")
        config.ENABLE_WEB_SEARCH = True
        config.ENABLE_WEB_CRAWL = True
        config.LLM_PATH = "mock"
        config.EMBED_MODEL_PATH = "mock"

        self.logger = logging.getLogger("test_conductor")
        self.logger.setLevel(logging.ERROR)

        self.tmpdir = tempfile.mkdtemp()
        dataset_db_path = Path(os.path.join(self.tmpdir, "datasets"))
        workspace_db_path = Path(os.path.join(self.tmpdir, "workspaces"))

        self.conductor = Conductor(
            user_id="uX",
            chat_id="cX",
            config=config,
            logger=self.logger,
            prov_graph=ProvenanceGraph(self.logger),
            db_api=DBAPI(
                config,
                self.logger,
                dataset_db_path=str(dataset_db_path),
                workspace_db_path=str(workspace_db_path),
            ),
            language_model_api=LanguageModelAPI(config, self.logger),
        )

    def tearDown(self):
        patch.stopall()

    def test_table_retrieve_updates_retrieved_tables(self):
        # set the queued responses on the underlying mock LLM instance
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}
        ]}}""",
            f"""{{"plan": [
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"done"}}}}
        ]}}""",
        ]
        self.conductor.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[
                Table(
                    doc_id="table1",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
                    metadata={},
                )
            ]
        )

        gen = self.conductor.chat(
            user_input="Find relevant tables",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn(
            "done",
            responses[-1],
            "Expected final user-facing response to contain 'done'",
        )
        self.assertTrue(
            len(self.conductor.retrieved_tables) > 0,
            "retrieved_tables should have been updated",
        )

    def test_web_search_sets_web_search_result(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.WEB_SEARCH.value}","args":{{"prompt":"web search query"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"web done"}}}}
        ]}}"""
        ]
        self.conductor.action_set.retrieve_documents = MagicMock(
            return_value=[
                Text(
                    doc_id="web_result_1",
                    retriever_type=RetrieverType.WEB_SEARCH,
                    content="This is a web search result.",
                    metadata={},
                )
            ]
        )

        gen = self.conductor.chat(
            user_input="Look up web",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("web done", responses[-1], "Expected web done in final response")
        self.assertIsNotNone(
            self.conductor.web_search_result,
            "web_search_result should be set after web_search call",
        )

    def test_web_crawl_sets_web_crawl_result(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.WEB_CRAWL.value}","args":{{"url":"http://example.com"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"web crawl done"}}}}
        ]}}"""
        ]

        self.conductor.action_set.retrieve_documents = MagicMock(
            return_value=[
                Text(
                    doc_id="web_result_1",
                    retriever_type=RetrieverType.WEB_CRAWL,
                    content="This is a web crawl result.",
                    metadata={},
                )
            ]
        )

        gen = self.conductor.chat(
            user_input="Look up this URL: http://example.com",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn(
            "web crawl done", responses[-1], "Expected web crawl done in final response"
        )
        self.assertIsNotNone(
            self.conductor.web_crawl_result,
            "web_crawl_result should be set after web_crawl call",
        )

    def test_table_enumerator_updates_enumerated_ids(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.TABLE_ENUMERATION.value}","args":{{"patterns":["pattern"]}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"enum done"}}}}
        ]}}"""
        ]

        self.conductor.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[
                Table(
                    doc_id="table1",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
                    metadata={},
                )
            ]
        )

        gen = self.conductor.chat(
            user_input="enumerate",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("enum done", responses[-1])
        self.assertIsInstance(self.conductor.enumerated_tables, list)

    def test_state_manipulation_sets_only_S(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"state_manipulation","args":{{"S":"result = something"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"S set"}}}}
        ]}}"""
        ]
        gen = self.conductor.chat(
            user_input="set S",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("S set", responses[-1])

        state = self.conductor.state
        self.assertEqual(state.S, "result = something")
        self.assertFalse(state.T)
        self.assertFalse(state.is_T_materialized)
        self.assertFalse(state.is_S_executed)
        self.assertEqual(state.column_descriptions, {})

    def test_state_manipulation_sets_only_T(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"state_manipulation","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"T set"}}}}
        ]}}"""
        ]

        gen = self.conductor.chat(
            user_input="set T",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("T set", responses[-1])

        state = self.conductor.state
        self.assertIn("t1", state.T)
        self.assertIsInstance(state.T["t1"], AbstractDocument)
        self.assertEqual(set(state.T["t1"].content.columns), {"a", "b"})
        self.assertEqual(state.column_descriptions, {"t1": {"a": "col a"}})
        self.assertEqual(state.S, "")
        self.assertFalse(state.is_T_materialized)
        self.assertFalse(state.is_S_executed)

    def test_state_manipulation_sets_S_and_T(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result = pd.DataFrame({{'sum': [tables['t1']['a'].sum()]}})"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"state done"}}}}
        ]}}"""
        ]
        gen = self.conductor.chat(
            user_input="set S and T",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)

        self.assertIn("state done", responses[-1])

        state = self.conductor.state
        self.assertIn("t1", state.T)
        self.assertIsInstance(state.T["t1"], AbstractDocument)
        self.assertEqual(set(state.T["t1"].content.columns), {"a", "b"})
        self.assertEqual(state.column_descriptions, {"t1": {"a": "col a"}})
        self.assertEqual(
            state.S, "result = pd.DataFrame({'sum': [tables['t1']['a'].sum()]})"
        )
        self.assertFalse(state.is_T_materialized)
        self.assertFalse(state.is_S_executed)

    def test_state_manipulation_redefines_T_cleans_previous_tables(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"T set"}}}}
        ]}}"""
        ]

        responses = list(
            self.conductor.chat(
                user_input="set T",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertIn("T set", responses[-1])

        tables_after_first = set(
            self.conductor.db_api.execute_query(
                self.conductor.user_id, self.conductor.chat_id, "SHOW TABLES;"
            )["name"].tolist()
        )
        self.assertIn("t1", tables_after_first)

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t2":["a","b"]}},"column_descriptions":{{"t2":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"T reset"}}}}
        ]}}"""
        ]

        responses = list(
            self.conductor.chat(
                user_input="set T again",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertIn("T reset", responses[-1])

        tables_after_second = set(
            self.conductor.db_api.execute_query(
                self.conductor.user_id, self.conductor.chat_id, "SHOW TABLES;"
            )["name"].tolist()
        )
        self.assertIn("t2", tables_after_second)
        self.assertNotIn("t1", tables_after_second)

    def test_materializer_and_executor(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
                {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result = pd.DataFrame({{'sum': [tables['t1']['a'].sum()]}})"}}}},
                {{"action":"{ActionNames.MATERIALIZER.value}","args":{{"note":""}}}},
                {{"action":"{ActionNames.PYTHON_EXECUTOR.value}","args":{{}}}}
            ]}}""",
            f"""{{"plan": [
                {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"materialization and execution done"}}}}
            ]}}""",
        ]
        self.conductor.materializer.materialize_T = MagicMock(
            return_value=(
                [],
                None,
                None,
                None,
                {"t1": pd.DataFrame({"a": [1, 2], "b": [3, 4]})},
            )
        )
        self.conductor.action_set.execute_code = MagicMock(
            return_value=pd.DataFrame({"sum": [3]})
        )

        gen = self.conductor.chat(
            user_input="materialize T",
            interaction_history=[],
            external_table_paths=[],
        )
        list(gen)

        self.assertTrue(
            self.conductor.state.is_T_materialized,
            "T should be marked as materialized",
        )
        self.assertTrue(
            self.conductor.state.is_S_executed,
            "S should be marked as executed",
        )
        self.assertEqual(self.conductor.state.T["t1"].content.shape, (2, 2))
        self.assertEqual(list(self.conductor.state.T["t1"].content["a"]), [1, 2])
        self.assertEqual(list(self.conductor.state.T["t1"].content["b"]), [3, 4])

    def test_assumption_check_produces_expected_string(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "x", "y"]})
        self.conductor.retrieved_tables = [
            Table(
                doc_id="table1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=df,
                metadata={},
            )
        ]

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.CONTEXT_EXTRACTION.value}","args":{{"code":"result = tables['table1']['A'].mean()"}}}},
        ]}}""",
            f"""{{"plan": [
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"info provided"}}}}
        ]}}""",
        ]
        gen = self.conductor.chat(
            user_input="check assumptions",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("info provided", responses[-1])

    def test_external_table_upload_creates_provenance_node(self):
        """Tests that uploading an external table results in a new provenance node."""
        import tempfile

        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()
        df.to_csv(tmp_path, index=False)

        uploaded_table = Table(
            doc_id="uploaded_table_1",
            retriever_type=RetrieverType.USER,
            content=pd.DataFrame(),
            metadata={},
            path=tmp_path,
        )
        self.conductor.table_reader.process_external_tables = MagicMock(
            return_value=[uploaded_table]
        )

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{"message":"External data read successfuly."}}}}
        ]}}"""
        ]
        gen = self.conductor.chat(
            user_input="upload",
            interaction_history=[],
            external_table_paths=[tmp_path],
        )
        list(gen)

        self.assertTrue(len(self.conductor.prov_graph.nodes) == 2)
        prov_graph_code_lines = [
            self.conductor.prov_graph.ROOT_NODE_CODE,
            self.conductor.action_set.generate_read_external_tables_code(
                1, uploaded_table
            ),
        ]
        expected_prov_graph_code_concat = "\n\n".join(prov_graph_code_lines)
        self.assertEqual(
            expected_prov_graph_code_concat,
            self.conductor.prov_graph.get_graph_code(),
        )


if __name__ == "__main__":
    unittest.main()
