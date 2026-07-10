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
            frontend_callback=lambda _: None,
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
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "done",
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
            user_message="Find relevant tables",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn(
            "done",
            responses[-1].message,
            "Expected final user-facing response to contain 'done'",
        )
        self.assertTrue(
            len(self.conductor.retrieved_tables) > 0,
            "retrieved_tables should have been updated",
        )

    def test_web_search_sets_web_search_result(self):
        self.conductor.language_model_api.llm._responses = [
            f"""{{"plan": [
            {{"action":"{ActionNames.WEB_SEARCH.value}","args":{{"prompt":"web search query"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "web done",
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
            user_message="Look up web",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn(
            "web done", responses[-1].message, "Expected web done in final response"
        )
        self.assertIsNotNone(
            self.conductor.web_search_result,
            "web_search_result should be set after web_search call",
        )

    def test_web_crawl_sets_web_crawl_result(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.WEB_CRAWL.value}","args":{{"url":"http://example.com"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "web crawl done",
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
            user_message="Look up this URL: http://example.com",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn(
            "web crawl done",
            responses[-1].message,
            "Expected web crawl done in final response",
        )
        self.assertIsNotNone(
            self.conductor.web_crawl_result,
            "web_crawl_result should be set after web_crawl call",
        )

    def test_table_enumerator_updates_enumerated_ids(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.TABLE_ENUMERATION.value}","args":{{"patterns":["pattern"]}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "enum done",
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
            user_message="enumerate",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("enum done", responses[-1].message)
        self.assertIsInstance(self.conductor.enumerated_tables, list)

    def test_state_manipulation_sets_only_S(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"state_manipulation","args":{{"S":"result = something"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "S set",
        ]
        gen = self.conductor.chat(
            user_message="set S",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("S set", responses[-1].message)

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
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "T set",
        ]

        gen = self.conductor.chat(
            user_message="set T",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("T set", responses[-1].message)

        state = self.conductor.state
        self.assertIn("t1", state.T)
        self.assertIsInstance(state.T["t1"], AbstractDocument)
        self.assertEqual(set(state.T["t1"].content.columns), {"a", "b"})
        self.assertEqual(state.column_descriptions, {"t1": {"a": "col a"}})
        self.assertEqual(state.S, "")
        self.assertFalse(state.is_T_materialized)
        self.assertFalse(state.is_S_executed)

    def test_state_manipulation_T_only_resets_is_S_executed(self):
        """Redefining only T must force S to be re-run, even if S was already executed
        against the previous T."""
        self.conductor.state.S = "result = 1"
        self.conductor.state.is_S_executed = True
        self.conductor.state.is_T_materialized = True

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "T-only refinement done",
        ]
        list(
            self.conductor.chat(
                user_message="update T only",
                interaction_history=[],
                external_table_paths=[],
            )
        )

        self.assertFalse(
            self.conductor.state.is_S_executed,
            "is_S_executed must be reset to False when T changes, even if S itself is unchanged",
        )

    def test_state_manipulation_sets_S_and_T(self):
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result = pd.DataFrame({{'sum': [tables['t1']['a'].sum()]}})"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "state done",
        ]
        gen = self.conductor.chat(
            user_message="set S and T",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)

        self.assertIn("state done", responses[-1].message)

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

    def test_state_manipulation_redefines_T_updates_in_memory_only(self):
        """STATE_MANIPULATION is a pure in-memory op; it must not create or drop DB tables."""
        baseline_tables = set(
            self.conductor.db_api.execute_query(
                self.conductor.user_id, self.conductor.chat_id, "SHOW TABLES;"
            )["name"].tolist()
        )

        # First STATE_MANIPULATION: define T={t1}
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "T set",
        ]
        responses = list(
            self.conductor.chat(
                user_message="set T",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertIn("T set", responses[-1].message)

        # In-memory state updated
        self.assertIn("t1", self.conductor.state.T)
        self.assertFalse(self.conductor.state.is_T_materialized)

        # DB must be unchanged — no t1 table created
        tables_after_first = set(
            self.conductor.db_api.execute_query(
                self.conductor.user_id, self.conductor.chat_id, "SHOW TABLES;"
            )["name"].tolist()
        )
        self.assertEqual(tables_after_first, baseline_tables)

        # Second STATE_MANIPULATION: redefine T={t2}
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t2":["a","b"]}},"column_descriptions":{{"t2":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "T reset",
        ]
        responses = list(
            self.conductor.chat(
                user_message="set T again",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertIn("T reset", responses[-1].message)

        # In-memory: t1 replaced by t2
        self.assertNotIn("t1", self.conductor.state.T)
        self.assertIn("t2", self.conductor.state.T)
        self.assertFalse(self.conductor.state.is_T_materialized)

        # DB still unchanged — neither t1 nor t2 created
        tables_after_second = set(
            self.conductor.db_api.execute_query(
                self.conductor.user_id, self.conductor.chat_id, "SHOW TABLES;"
            )["name"].tolist()
        )
        self.assertEqual(tables_after_second, baseline_tables)

    def test_state_manipulation_does_not_drop_existing_t_tables_from_db(self):
        """Manually persist a T table; STATE_MANIPULATION redefining T must leave it in DB."""
        self.conductor.db_api.persist_df(
            self.conductor.user_id,
            self.conductor.chat_id,
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "t1",
            True,
        )

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b","c"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "T redefined",
        ]
        list(
            self.conductor.chat(
                user_message="redefine T",
                interaction_history=[],
                external_table_paths=[],
            )
        )

        # t1 must still exist in DB with original content
        result = self.conductor.db_api.execute_query(
            self.conductor.user_id, self.conductor.chat_id, 'SELECT * FROM "t1";'
        )
        self.assertEqual(len(result), 2)
        self.assertListEqual(list(result.columns), ["a", "b"])

    def test_materializer_mode_update_passed_correctly(self):
        """MATERIALIZER with mode=update must call materialize_T with update_mode=True."""
        captured = {}

        def fake_stream(T, col_desc, S, note="", update_mode=False, *args, **kwargs):
            captured["update_mode"] = update_mode
            return ([], None, None, None, {"t1": pd.DataFrame({"a": [1]})})

        self.conductor.materializer.materialize_T = fake_stream  # type: ignore

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.MATERIALIZER.value}","args":{{"note":"add col","mode":"update"}}}}
        ]}}""",
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "done",
        ]
        list(
            self.conductor.chat(
                user_message="update mode test",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertTrue(
            captured.get("update_mode"),
            "materialize_T must be called with update_mode=True when mode='update'",
        )

    def test_materializer_mode_fresh_when_mode_omitted(self):
        """MATERIALIZER without mode arg must call materialize_T with update_mode=False."""
        captured = {}

        def fake_stream(T, col_desc, S, note="", update_mode=False, *args, **kwargs):
            captured["update_mode"] = update_mode
            return ([], None, None, None, {"t1": pd.DataFrame({"a": [1]})})

        self.conductor.materializer.materialize_T = fake_stream  # type: ignore

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.MATERIALIZER.value}","args":{{"note":""}}}}
        ]}}""",
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "done",
        ]
        list(
            self.conductor.chat(
                user_message="fresh mode test",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertFalse(
            captured.get("update_mode"),
            "materialize_T must be called with update_mode=False when mode is omitted",
        )

    def test_materializer_mode_fresh_when_mode_is_reset(self):
        """MATERIALIZER with mode=reset must call materialize_T with update_mode=False."""
        captured = {}

        def fake_stream(T, col_desc, S, note="", update_mode=False, *args, **kwargs):
            captured["update_mode"] = update_mode
            return ([], None, None, None, {"t1": pd.DataFrame({"a": [1]})})

        self.conductor.materializer.materialize_T = fake_stream  # type: ignore

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.MATERIALIZER.value}","args":{{"note":"","mode":"reset"}}}}
        ]}}""",
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "done",
        ]
        list(
            self.conductor.chat(
                user_message="reset mode test",
                interaction_history=[],
                external_table_paths=[],
            )
        )
        self.assertFalse(
            captured.get("update_mode"),
            "materialize_T must be called with update_mode=False when mode='reset'",
        )

    def test_materializer_and_executor(self):
        """_accelerate_plan queues PYTHON_EXECUTOR after MATERIALIZER and
        USER_FACING_COMMUNICATION after PYTHON_EXECUTOR, without an extra planning
        round-trip. Explicitly bundling PYTHON_EXECUTOR after MATERIALIZER (as this plan
        still does) must not cause a duplicate queue entry or run S/the final
        communication twice — _accelerate_plan's dedup against the rest of the plan is
        what prevents it here, not a handler-level guard."""
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
                {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result = pd.DataFrame({{'sum': [tables['t1']['a'].sum()]}})"}}}},
                {{"action":"{ActionNames.MATERIALIZER.value}","args":{{"note":""}}}},
                {{"action":"{ActionNames.PYTHON_EXECUTOR.value}","args":{{}}}}
            ]}}""",
            "materialization and execution done",
        ]
        expected_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self.conductor.materializer.materialize_T = MagicMock(
            return_value=([], None, None, None, {"t1": expected_df})
        )
        self.conductor.action_set.execute_code = MagicMock(
            return_value=pd.DataFrame({"sum": [3]})
        )

        gen = self.conductor.chat(
            user_message="materialize T",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)

        self.assertEqual(responses[-1].message, "materialization and execution done")
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
        self.conductor.action_set.execute_code.assert_called_once()
        self.conductor.materializer.materialize_T.assert_called_once()

    def test_python_executor_self_trigger_chain_executes_s_once(self):
        """python_executor with T undefined-but-materializable is intercepted by
        _accelerate_plan, which substitutes materializer for this step and requeues
        python_executor onto the plan. Once materializer succeeds, python_executor runs
        for real from the queue and its own acceleration queues user_facing_communication.
        S must execute exactly once and the final response must be produced exactly once."""
        self.conductor.state.T = {
            "t1": Table(
                doc_id="t1",
                retriever_type=RetrieverType.CONDUCTOR,
                content=pd.DataFrame(columns=["a"]),
                metadata={},
            )
        }
        self.conductor.state.column_descriptions = {"t1": {"a": "col a"}}
        self.conductor.state.S = "result = 1"

        self.conductor.materializer.materialize_T = MagicMock(
            return_value=([], None, None, None, {"t1": pd.DataFrame({"a": [1]})})
        )
        self.conductor.action_set.execute_code = MagicMock(
            return_value=pd.DataFrame({"a": [1]})
        )

        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [{{"action":"{ActionNames.PYTHON_EXECUTOR.value}","args":{{}}}}]}}""",
            "chained done",
        ]

        responses = list(
            self.conductor.chat("run S", interaction_history=[], external_table_paths=[])
        )

        self.assertEqual(responses[-1].message, "chained done")
        self.assertEqual(
            self.conductor.action_set.execute_code.call_count,
            1,
            "S must execute exactly once despite the nested materializer auto-chain",
        )
        self.conductor.materializer.materialize_T.assert_called_once()
        self.assertTrue(self.conductor.state.is_T_materialized)
        self.assertTrue(self.conductor.state.is_S_executed)

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
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "info provided",
        ]
        gen = self.conductor.chat(
            user_message="check assumptions",
            interaction_history=[],
            external_table_paths=[],
        )
        responses = list(gen)
        self.assertIn("info provided", responses[-1].message)


    def test_max_steps_force_response(self):
        """When all conductor steps are exhausted without a response, a fallback is force-produced."""
        self.conductor.config.MAX_CONDUCTOR_STEPS = 1
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"still thinking"}}}}]}}""",
            "Fallback answer",  # consumed by the force-produce LLM call
        ]
        responses = list(
            self.conductor.chat("test", interaction_history=[], external_table_paths=[])
        )
        self.assertEqual(responses[-1].message, "Fallback answer")

    def test_llm_exception_propagates(self):
        """An exception raised by the LLM during planning is re-raised, not swallowed."""
        with patch.object(
            self.conductor.language_model_api,
            "chat",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                list(
                    self.conductor.chat(
                        "test", interaction_history=[], external_table_paths=[]
                    )
                )

    def test_plan_parse_error_retries_with_feedback(self):
        """A structurally invalid LLM plan is fed back as an error and the conductor retries."""
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            '{"plan": "not a list"}',  # plan must be a list — triggers retry
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "recovered",
        ]
        responses = list(
            self.conductor.chat("test", interaction_history=[], external_table_paths=[])
        )
        self.assertIn("recovered", responses[-1].message)

    def test_validate_plan_strips_user_facing_with_execution_action(self):
        """_validate_plan removes user_facing_communication when an execution action is present."""
        plan = [
            {"action": ActionNames.PYTHON_EXECUTOR.value, "args": {}},
            {"action": ActionNames.USER_FACING_COMMUNICATION.value, "args": {"message": "done"}},
        ]
        result = self.conductor._validate_plan(plan)
        action_names = [p["action"] for p in result]
        self.assertNotIn(ActionNames.USER_FACING_COMMUNICATION.value, action_names)
        self.assertIn(ActionNames.PYTHON_EXECUTOR.value, action_names)
        self.assertIsNotNone(
            self.conductor._pending_plan_feedback,
            "Stripping user_facing_communication must surface feedback for the next planning turn",
        )

    def test_validate_plan_preserves_user_facing_without_execution_action(self):
        """_validate_plan does NOT strip user_facing_communication when no execution action is present."""
        plan = [
            {"action": ActionNames.SITUATIONAL_ANALYSIS.value, "args": {"message": "thinking"}},
            {"action": ActionNames.USER_FACING_COMMUNICATION.value, "args": {"message": "done"}},
        ]
        result = self.conductor._validate_plan(plan)
        action_names = [p["action"] for p in result]
        self.assertIn(ActionNames.USER_FACING_COMMUNICATION.value, action_names)
        self.assertIsNone(self.conductor._pending_plan_feedback)

    def test_ds_skeptic_pushback_aborts_remaining_plan(self):
        """DS-Skeptic pushback causes actions after state_manipulation to be skipped in that step."""
        self.conductor.config.ENABLE_DS_SKEPTIC = True
        self.conductor.config.MAX_DS_SKEPTIC_ROUNDS = 1
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
                {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result=1"}}}},
                {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}
            ]}}""",
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "reconsidered",
        ]
        self.conductor.ds_skeptic.review = MagicMock(return_value=(True, "Skeptic has concerns"))

        responses = list(
            self.conductor.chat("test", interaction_history=[], external_table_paths=[])
        )

        self.assertEqual(responses[-1].message, "reconsidered")
        self.conductor.ds_skeptic.review.assert_called_once()

    def test_action_error_aborts_remaining_plan(self):
        """An ordinary action error aborts remaining actions in the same step, mirroring
        the existing DS-Skeptic pushback behavior."""
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
                {{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{}}}},
                {{"action":"{ActionNames.WEB_SEARCH.value}","args":{{"prompt":"x"}}}}
            ]}}""",
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "recovered after error",
        ]
        self.conductor.action_set.retrieve_documents = MagicMock(return_value=[])

        responses = list(
            self.conductor.chat("test", interaction_history=[], external_table_paths=[])
        )

        self.conductor.action_set.retrieve_documents.assert_not_called()
        self.assertEqual(responses[-1].message, "recovered after error")
        abort_messages = [
            m["content"]
            for m in self.conductor.llm_messages
            if isinstance(m["content"], str) and "were skipped because" in m["content"]
        ]
        self.assertTrue(
            abort_messages, "Expected an abort note in llm_messages after the error"
        )
        self.assertNotIn(
            ActionNames.WEB_SEARCH.value,
            self.conductor.actions[0],
            "The recorded action log for the failed step must not include the "
            "action that was never reached.",
        )

    def test_accelerated_actions_recorded_in_actions_log(self):
        """_accelerate_plan expands the plan before it's recorded, so 'recent actions'
        shown to Conductor next turn already reflects PYTHON_EXECUTOR and
        USER_FACING_COMMUNICATION even though the LLM only planned up to MATERIALIZER —
        Conductor doesn't have to infer what happened from environment state alone."""
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
                {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a","b"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result = pd.DataFrame({{'sum': [tables['t1']['a'].sum()]}})"}}}},
                {{"action":"{ActionNames.MATERIALIZER.value}","args":{{"note":""}}}}
            ]}}""",
            "materialization and execution done",
        ]
        self.conductor.materializer.materialize_T = MagicMock(
            return_value=([], None, None, None, {"t1": pd.DataFrame({"a": [1, 2], "b": [3, 4]})})
        )
        self.conductor.action_set.execute_code = MagicMock(
            return_value=pd.DataFrame({"sum": [3]})
        )

        list(
            self.conductor.chat(
                user_message="materialize T",
                interaction_history=[],
                external_table_paths=[],
            )
        )

        recorded = self.conductor.actions[0]
        self.assertIn(ActionNames.MATERIALIZER.value, recorded)
        self.assertIn(
            ActionNames.PYTHON_EXECUTOR.value,
            recorded,
            "Accelerated PYTHON_EXECUTOR must appear in the recorded action log even "
            "though the LLM never planned it explicitly.",
        )
        self.assertIn(
            ActionNames.USER_FACING_COMMUNICATION.value,
            recorded,
            "Accelerated USER_FACING_COMMUNICATION must appear in the recorded action "
            "log even though the LLM never planned it explicitly.",
        )

    def test_state_manipulation_clears_s_description_when_S_updated(self):
        """When STATE_MANIPULATION updates S, s_description must be cleared."""
        self.conductor.state.s_description = "Old explanation."

        self.conductor.language_model_api.llm._responses = [
            f"""{{"plan": [
            {{"action":"state_manipulation","args":{{"S":"result = 42"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "S updated",
        ]
        list(self.conductor.chat(
            user_message="update S",
            interaction_history=[],
            external_table_paths=[],
        ))

        self.assertEqual(self.conductor.state.S, "result = 42")
        self.assertEqual(
            self.conductor.state.s_description,
            "",
            "s_description must be cleared when S is updated",
        )

    def test_state_manipulation_clears_s_description_when_S_and_T_updated(self):
        """When STATE_MANIPULATION updates both S and T, s_description must be cleared."""
        self.conductor.state.s_description = "Old explanation."

        self.conductor.language_model_api.llm._responses = [
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}},"S":"result = 1"}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "both updated",
        ]
        list(self.conductor.chat(
            user_message="update S and T",
            interaction_history=[],
            external_table_paths=[],
        ))

        self.assertEqual(self.conductor.state.s_description, "")

    def test_state_manipulation_preserves_s_description_when_only_T_updated(self):
        """When STATE_MANIPULATION updates only T (not S), s_description must be preserved."""
        self.conductor.state.s_description = "Existing explanation."

        self.conductor.language_model_api.llm._responses = [
            f"""{{"plan": [
            {{"action":"{ActionNames.STATE_MANIPULATION.value}","args":{{"T":{{"t1":["a"]}},"column_descriptions":{{"t1":{{"a":"col a"}}}}}}}},
            {{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args": {{}}}}
        ]}}""",
            "T only",
        ]
        list(self.conductor.chat(
            user_message="update T only",
            interaction_history=[],
            external_table_paths=[],
        ))

        self.assertEqual(
            self.conductor.state.s_description,
            "Existing explanation.",
            "s_description must not be cleared when only T is updated",
        )

    def test_context_extraction_new_uncertainties_path(self):
        """Context extraction with the uncertainties format calls run_context_extraction."""
        self.conductor.retrieved_tables = [
            Table(
                doc_id="table1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=pd.DataFrame({"A": [1, 2]}),
                metadata={},
            )
        ]
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f"""{{"plan": [
                {{"action":"{ActionNames.CONTEXT_EXTRACTION.value}","args":{{"uncertainties":[{{"table_ids":["table1"],"question":"what is the range of A?"}}]}}}}
            ]}}""",
            f"""{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}""",
            "done",
        ]
        self.conductor.action_set.run_context_extraction = MagicMock(
            return_value=("A ranges from 1 to 2", [])
        )

        responses = list(
            self.conductor.chat("check assumptions", interaction_history=[], external_table_paths=[])
        )
        self.assertIn("done", responses[-1].message)
        self.conductor.action_set.run_context_extraction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
