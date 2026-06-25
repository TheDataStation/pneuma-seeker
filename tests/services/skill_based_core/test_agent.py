# tests/services/skill_based_core/test_agent.py
#
# Mirror of tests/services/core/conductor/test_main.py adapted for SkillsAgent.
# Goal: measure token usage and runtime of the flat skills-based architecture vs
# the two-level Conductor + Materializer architecture.
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

import unittest
from unittest.mock import MagicMock

import pandas as pd

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.skill_based_core.agent import SkillsAgent
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    RetrieverType,
    Table,
    Text,
)


class SkillsAgentTests(unittest.TestCase):
    def setUp(self):
        config = Config(".env.test")
        config.ENABLE_WEB_SEARCH = True
        config.ENABLE_WEB_CRAWL = True
        config.LLM_PATH = "mock"
        config.EMBED_MODEL_PATH = "mock"

        self.logger = logging.getLogger("test_skills_agent")
        self.logger.setLevel(logging.ERROR)

        self.tmpdir = tempfile.mkdtemp()
        dataset_db_path = Path(os.path.join(self.tmpdir, "datasets"))
        workspace_db_path = Path(os.path.join(self.tmpdir, "workspaces"))

        self.agent = SkillsAgent(
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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def test_retrieve_tables_updates_retrieved_tables(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "retrieve_tables", "args": {"prompts": ["find tables"]}}',
            '{"skill": "respond", "args": {"message": "done"}}',
        ]
        self.agent.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[
                Table(
                    doc_id="table1",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
                    metadata={},
                )
            ]
        )

        responses = list(
            self.agent.chat(
                user_input="Find relevant tables",
                interaction_history=[],
                external_table_paths=[],
            )
        )

        self.assertIn("done", responses[-1].message)
        self.assertTrue(len(self.agent.retrieved_tables) > 0)

    def test_web_search_sets_web_search_result(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "web_search", "args": {"query": "some query"}}',
            '{"skill": "respond", "args": {"message": "web done"}}',
        ]
        self.agent.action_set.retrieve_documents = MagicMock(
            return_value=[
                Text(
                    doc_id="web_result_1",
                    retriever_type=RetrieverType.WEB_SEARCH,
                    content="Web result.",
                    metadata={},
                )
            ]
        )

        responses = list(
            self.agent.chat(
                user_input="search web",
                interaction_history=[],
                external_table_paths=[],
            )
        )

        self.assertIn("web done", responses[-1].message)
        self.assertIsNotNone(self.agent.web_search_result)

    def test_web_crawl_sets_web_crawl_result(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "web_crawl", "args": {"url": "http://example.com"}}',
            '{"skill": "respond", "args": {"message": "crawl done"}}',
        ]
        self.agent.action_set.retrieve_documents = MagicMock(
            return_value=[
                Text(
                    doc_id="crawl_1",
                    retriever_type=RetrieverType.WEB_CRAWL,
                    content="Crawled content.",
                    metadata={},
                )
            ]
        )

        responses = list(
            self.agent.chat(
                user_input="crawl",
                interaction_history=[],
                external_table_paths=[],
            )
        )

        self.assertIn("crawl done", responses[-1].message)
        self.assertIsNotNone(self.agent.web_crawl_result)

    def test_enumerate_tables_populates_enumerated_tables(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "enumerate_tables", "args": {"patterns": ["sales_*"]}}',
            '{"skill": "respond", "args": {"message": "enum done"}}',
        ]
        self.agent.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[
                Table(
                    doc_id="sales_2024",
                    retriever_type=RetrieverType.ENUMERATOR,
                    content=pd.DataFrame({"A": [1]}),
                    metadata={},
                )
            ]
        )

        responses = list(
            self.agent.chat(
                "enumerate", interaction_history=[], external_table_paths=[]
            )
        )

        self.assertIn("enum done", responses[-1].message)
        self.assertTrue(len(self.agent.enumerated_tables) > 0)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def test_run_python_creates_workspace_table(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "run_python", "args": {"code": "pass", "result_table_id": "my_result"}}',
            '{"skill": "respond", "args": {"message": "python done"}}',
        ]
        self.agent.action_set.execute_code = MagicMock(
            return_value=pd.DataFrame({"sum": [3]})
        )

        responses = list(
            self.agent.chat("compute", interaction_history=[], external_table_paths=[])
        )

        self.assertIn("python done", responses[-1].message)
        self.assertIn("my_result", self.agent.workspace_table_ids)

    def test_run_sql_creates_workspace_table(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "run_sql", "args": {"query": "SELECT 1 AS val", "result_table_id": "sql_out"}}',
            '{"skill": "respond", "args": {"message": "sql done"}}',
        ]
        self.agent.action_set.execute_query = MagicMock(
            return_value=pd.DataFrame({"val": [1]})
        )

        responses = list(
            self.agent.chat("run sql", interaction_history=[], external_table_paths=[])
        )

        self.assertIn("sql done", responses[-1].message)
        self.assertIn("sql_out", self.agent.workspace_table_ids)

    def test_project_table_registers_workspace_table(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "project_table", "args": {"src_table_id": "src", "target_table_id": "proj_out", "column_mapping": {"A": "a"}}}',
            '{"skill": "respond", "args": {"message": "project done"}}',
        ]
        self.agent.action_set.project_table = MagicMock(
            return_value=pd.DataFrame({"a": [1, 2]})
        )

        responses = list(
            self.agent.chat("project", interaction_history=[], external_table_paths=[])
        )

        self.assertIn("project done", responses[-1].message)
        self.assertIn("proj_out", self.agent.workspace_table_ids)

    def test_join_tables_registers_workspace_table(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "join_tables", "args": {"left_table_id": "l", "right_table_id": "r", "left_keys": ["id"], "right_keys": ["id"], "output_table_id": "joined"}}',
            '{"skill": "respond", "args": {"message": "join done"}}',
        ]
        self.agent.action_set.join_equality = MagicMock(
            return_value=pd.DataFrame({"id": [1], "val": [10]})
        )

        responses = list(
            self.agent.chat("join", interaction_history=[], external_table_paths=[])
        )

        self.assertIn("join done", responses[-1].message)
        self.assertIn("joined", self.agent.workspace_table_ids)

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def test_probe_table_returns_summary(self):
        self.agent.retrieved_tables = [
            Table(
                doc_id="table1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=pd.DataFrame({"A": [1, 2, 3], "B": ["x", "x", "y"]}),
                metadata={},
            )
        ]
        # run_context_extraction calls the LLM internally; mock to avoid consuming
        # responses reserved for the outer skills loop.
        self.agent.action_set.run_context_extraction = MagicMock(
            return_value=("Mean of A is 2.0", [])
        )

        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "probe_table", "args": {"uncertainties": [{"table_ids": ["table1"], "question": "What is the mean of A?"}]}}',
            '{"skill": "respond", "args": {"message": "probe done"}}',
        ]

        responses = list(
            self.agent.chat("probe", interaction_history=[], external_table_paths=[])
        )

        self.assertIn("probe done", responses[-1].message)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def test_respond_terminates_loop(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "respond", "args": {"message": "hello user"}}',
            '{"skill": "respond", "args": {"message": "should not appear"}}',
        ]

        responses = list(
            self.agent.chat("say hi", interaction_history=[], external_table_paths=[])
        )

        self.assertEqual(responses[-1].message, "hello user")
        self.assertEqual(
            len(self.agent.language_model_api.llm._responses),  # type: ignore
            1,
            "Loop must stop on respond — second response must remain unconsumed",
        )

    def test_situational_analysis_does_not_terminate(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "situational_analysis", "args": {"message": "thinking"}}',
            '{"skill": "respond", "args": {"message": "final"}}',
        ]

        responses = list(
            self.agent.chat(
                "think then respond", interaction_history=[], external_table_paths=[]
            )
        )

        self.assertIn("final", responses[-1].message)

    def test_unknown_skill_does_not_crash(self):
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "nonexistent_skill", "args": {}}',
            '{"skill": "respond", "args": {"message": "recovered"}}',
        ]

        responses = list(
            self.agent.chat(
                "trigger unknown", interaction_history=[], external_table_paths=[]
            )
        )

        self.assertIn("recovered", responses[-1].message)

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def test_token_profiling_silent_without_counters(self):
        """MockLLM has no token counters; hasattr guards must be silent."""
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "respond", "args": {"message": "ok"}}',
        ]

        responses = list(
            self.agent.chat("anything", interaction_history=[], external_table_paths=[])
        )

        self.assertEqual(responses[-1].message, "ok")
        self.assertFalse(
            hasattr(self.agent.language_model_api.llm, "total_input_tokens")
        )

    def test_token_profiling_reads_counters_when_present(self):
        llm = self.agent.language_model_api.llm
        llm.total_input_tokens = 100  # type: ignore
        llm.total_output_tokens = 50  # type: ignore
        llm.total_llm_time = 1.5  # type: ignore
        llm.reset_metrics = lambda: None  # type: ignore

        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "respond", "args": {"message": "counted"}}',
        ]

        responses = list(
            self.agent.chat("count", interaction_history=[], external_table_paths=[])
        )

        self.assertIn("counted", responses[-1].message)

    # ------------------------------------------------------------------
    # External table upload
    # ------------------------------------------------------------------

    def test_external_table_upload_creates_provenance_node(self):
        import tempfile as tmp_mod

        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        tmp = tmp_mod.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()
        df.to_csv(tmp_path, index=False)

        from pneuma_seeker.shared.schemas.core.ir_system import Table

        uploaded = Table(
            doc_id="uploaded_table_1",
            retriever_type=RetrieverType.USER,
            content=pd.DataFrame(),
            metadata={},
            path=tmp_path,
        )
        self.agent.table_reader.process_external_tables = MagicMock(
            return_value=[uploaded]
        )

        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "respond", "args": {"message": "External data read successfully."}}'
        ]

        list(
            self.agent.chat(
                "upload", interaction_history=[], external_table_paths=[tmp_path]
            )
        )

        self.assertEqual(len(self.agent.prov_graph.nodes), 2)

    # ------------------------------------------------------------------
    # Relational reification (define_target)
    # ------------------------------------------------------------------

    def test_define_target_sets_T_and_S_on_state(self):
        """define_target populates agent.state.T and agent.state.S (relational reification)."""
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "define_target", "args": {"T": {"summary": ["category", "total"]}, "column_descriptions": {"summary": {"category": "product category", "total": "revenue total"}}, "S": "result = tables[\'summary\'].groupby(\'category\').sum()"}}',
            '{"skill": "respond", "args": {"message": "target set"}}',
        ]

        responses = list(
            self.agent.chat(
                "set target", interaction_history=[], external_table_paths=[]
            )
        )

        self.assertIn("target set", responses[-1].message)
        self.assertIn("summary", self.agent.state.T)
        self.assertEqual(
            list(self.agent.state.T["summary"].content.columns),
            ["category", "total"],
        )
        self.assertIn(
            "category", self.agent.state.column_descriptions.get("summary", {})
        )
        self.assertIn("tables['summary']", self.agent.state.S)
        self.assertFalse(self.agent.state.is_T_materialized)
        self.assertFalse(self.agent.state.is_S_executed)

    def test_define_target_S_only(self):
        """define_target can set only S without redefining T."""
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "define_target", "args": {"S": "result = tables[\'t1\'].head(5)"}}',
            '{"skill": "respond", "args": {"message": "S set"}}',
        ]

        responses = list(
            self.agent.chat(
                "set S only", interaction_history=[], external_table_paths=[]
            )
        )

        self.assertIn("S set", responses[-1].message)
        self.assertIn("tables['t1']", self.agent.state.S)
        self.assertFalse(self.agent.state.T)  # T untouched (still empty)

    def test_define_target_requires_column_descriptions_with_T(self):
        """define_target must reject T without column_descriptions."""
        self.agent.language_model_api.llm._responses = [  # type: ignore
            '{"skill": "define_target", "args": {"T": {"t1": ["a", "b"]}}}',
            '{"skill": "respond", "args": {"message": "recovered"}}',
        ]

        responses = list(
            self.agent.chat(
                "bad define_target", interaction_history=[], external_table_paths=[]
            )
        )

        # Loop must continue after the error and reach respond
        self.assertIn("recovered", responses[-1].message)
        # T must not have been set
        self.assertFalse(self.agent.state.T)

    def test_state_restored_via_set_prov_graph(self):
        """set_prov_graph propagates to action_set (ChatSession session-restore path)."""
        from pneuma_seeker.provenance.graph import ProvenanceGraph

        new_graph = ProvenanceGraph(self.logger)
        self.agent.set_prov_graph(new_graph)
        self.assertIs(self.agent.prov_graph, new_graph)
        self.assertIs(self.agent.action_set.prov_graph, new_graph)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def test_all_expected_skills_discovered(self):
        """Sanity-check that the discovery mechanism loaded all built-in skills."""
        expected = {
            "situational_analysis",
            "retrieve_tables",
            "enumerate_tables",
            "probe_table",
            "web_search",
            "web_crawl",
            "define_target",
            "project_table",
            "join_tables",
            "union_tables",
            "run_sql",
            "run_python",
            "respond",
        }
        self.assertEqual(expected, set(self.agent._skill_registry.keys()))


if __name__ == "__main__":
    unittest.main()
