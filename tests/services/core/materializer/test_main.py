# tests/pneuma_seeker/core/materializer/test_main.py
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
from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.materializer.main import Materializer
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table, Text


class MaterializerTests(unittest.TestCase):
    def setUp(self):
        self.user_id = "uX"
        self.chat_id = "cX"

        self.logger = logging.getLogger("test_materializer")
        self.logger.setLevel(logging.ERROR)
        self.prov_graph = ProvenanceGraph(self.logger)

        self.config = Config(".env.test")
        self.config.DATA_SOURCES = ["test_ds"]
        self.config.ENABLE_WEB_SEARCH = True
        self.config.ENABLE_WEB_CRAWL = True
        self.config.LLM_PATH = "mock"
        self.config.EMBED_MODEL_PATH = "mock"

        self.tmpdir = tempfile.mkdtemp()
        self.db_api = DBAPI(
            self.config,
            self.logger,
            str(Path(os.path.join(self.tmpdir, "datasets"))),
            str(Path(os.path.join(self.tmpdir, "workspaces"))),
        )
        self.lm_api = LanguageModelAPI(self.config, self.logger)

        self.action_set = ActionSet(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.prov_graph,
            self.db_api,
            self.lm_api,
        )

        self.materializer = Materializer(
            user_id=self.user_id,
            chat_id=self.chat_id,
            config=self.config,
            logger=self.logger,
            prov_graph=self.prov_graph,
            action_set=self.action_set,
            db_api=self.db_api,
            language_model_api=self.lm_api,
        )

    def tearDown(self):
        patch.stopall()

    def test_table_retrieve_and_table_projection_materializes_T(self):
        # LLM will ask to call table_retrieve then table_projection to materialize t1
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f"""{{"plan": [{plan1}, {plan2}]}}"""]  # type: ignore

        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )

        # TABLE_RETRIEVE uses multi-topic retrieval.
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        # Define target T (schema only) so materializer knows it needs t1
        T = {"t1": pd.DataFrame(columns=["a", "b"])}

        result = self.materializer.materialize_T(T=T, column_descriptions={}, S="")[-1]

        self.assertIn("t1", result)
        pd.testing.assert_frame_equal(result["t1"].reset_index(drop=True), table_df)

        self.assertTrue(len(self.materializer.prov_graph.nodes) == 3)
        prov_graph_code_lines = [
            self.materializer.prov_graph.ROOT_NODE_CODE,
            self.action_set.generate_pandas_read_csv_code(table_doc),
            self.action_set.generate_table_select_code(
                "t1", "test_ds.table_1", ["a", "b"]
            ),
        ]
        self.assertEqual(
            "\n\n".join(prov_graph_code_lines),
            self.materializer.prov_graph.get_graph_code(),
        )

    def test_web_search_sets_web_search_result(self):
        # LLM will call table_retrieve, web_search, then table_projection to finish
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = (
            f'{{"action":"{ActionNames.WEB_SEARCH.value}","args":{{"prompt":"query"}}}}'
        )
        plan3 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}, {plan3}]}}']  # type: ignore

        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )

        web_text = Text(
            doc_id="web_result_1",
            retriever_type=RetrieverType.WEB_SEARCH,
            content="This is a web search result.",
            metadata={},
        )

        # First call returns the table, second call returns the web result
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )
        self.action_set.retrieve_documents = MagicMock(return_value=[web_text])

        T = {"t1": pd.DataFrame(columns=["a", "b"])}

        result = self.materializer.materialize_T(T=T, column_descriptions={}, S="")[-1]

        # After web_search operation the state should have web_search_result set
        self.assertIsNotNone(self.materializer.state.web_search_result)
        assert self.materializer.state.web_search_result is not None
        self.assertEqual(
            self.materializer.state.web_search_result.content,
            "This is a web search result.",
        )

        # Ensure final materialized result still contains t1
        self.assertIn("t1", result)

        self.assertTrue(len(self.materializer.prov_graph.nodes) == 4)
        prov_graph_code_lines_1 = [
            self.materializer.prov_graph.ROOT_NODE_CODE,
            self.action_set.generate_pandas_read_csv_code(table_doc),
            self.action_set.generate_view_textual_document_code(web_text),
            self.action_set.generate_table_select_code(
                "t1", "test_ds.table_1", ["a", "b"]
            ),
        ]
        prov_graph_code_lines_2 = [
            self.materializer.prov_graph.ROOT_NODE_CODE,
            self.action_set.generate_view_textual_document_code(web_text),
            self.action_set.generate_pandas_read_csv_code(table_doc),
            self.action_set.generate_table_select_code(
                "t1", "test_ds.table_1", ["a", "b"]
            ),
        ]
        self.assertIn(
            self.materializer.prov_graph.get_graph_code(),
            (
                "\n\n".join(prov_graph_code_lines_1),
                "\n\n".join(prov_graph_code_lines_2),
            ),
        )

    def test_web_crawl_sets_web_crawl_result(self):
        # LLM will call table_retrieve, web_crawl, then table_projection to finish
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.WEB_CRAWL.value}","args":{{"url":"http://example.com"}}}}'
        plan3 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}, {plan3}]}}']  # type: ignore

        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )

        web_text = Text(
            doc_id="web_result_1",
            retriever_type=RetrieverType.WEB_CRAWL,
            content="This is a web crawl result.",
            metadata={},
        )

        # First call returns the table, second call returns the web result
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )
        self.action_set.retrieve_documents = MagicMock(return_value=[web_text])

        T = {"t1": pd.DataFrame(columns=["a", "b"])}

        result = self.materializer.materialize_T(T=T, column_descriptions={}, S="")[-1]

        # After web_crawl operation the state should have web_crawl_result set
        self.assertIsNotNone(self.materializer.state.web_crawl_result)
        assert self.materializer.state.web_crawl_result is not None
        self.assertEqual(
            self.materializer.state.web_crawl_result.content,
            "This is a web crawl result.",
        )

        # Ensure final materialized result still contains t1
        self.assertIn("t1", result)

        self.assertTrue(len(self.materializer.prov_graph.nodes) == 4)
        prov_graph_code_lines_1 = [
            self.materializer.prov_graph.ROOT_NODE_CODE,
            self.action_set.generate_pandas_read_csv_code(table_doc),
            self.action_set.generate_view_textual_document_code(web_text),
            self.action_set.generate_table_select_code(
                "t1", "test_ds.table_1", ["a", "b"]
            ),
        ]
        prov_graph_code_lines_2 = [
            self.materializer.prov_graph.ROOT_NODE_CODE,
            self.action_set.generate_view_textual_document_code(web_text),
            self.action_set.generate_pandas_read_csv_code(table_doc),
            self.action_set.generate_table_select_code(
                "t1", "test_ds.table_1", ["a", "b"]
            ),
        ]
        self.assertIn(
            self.materializer.prov_graph.get_graph_code(),
            (
                "\n\n".join(prov_graph_code_lines_1),
                "\n\n".join(prov_graph_code_lines_2),
            ),
        )

    def test_semantic_column_generator_adds_column(self):
        self.config.ENABLE_SEMANTIC_COL_GEN = True
        # LLM will call table_retrieve, table_projection, then semantic_column_generator
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        plan3 = f'{{"action":"{ActionNames.SEMANTIC_COLUMN_GENERATION.value}","args":{{"table_id":"t1","new_column_name":"newcol","relevant_columns":["b"],"instruction":"make new"}}}}'

        # Mock the LLM responses using llm._responses (the internal list used by the LLM mock)
        # First response: planning response with the actions
        planning_response_1 = f'{{"plan": [{plan1}, {plan2}, {plan3}]}}'
        # Second response: the semantic column generation LLM call returns the transformed values
        sem_col_response = (
            "['Result for: name: Alice; age: 30', 'Result for: name: Alice; age: 30']"
        )

        self.lm_api.llm._responses = [  # type: ignore
            planning_response_1,  # Planning call
            sem_col_response,  # Semantic column generation call
        ]  # type: ignore

        table_df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df.copy(),
            metadata={},
        )

        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        T = {"t1": pd.DataFrame(columns=["a", "b", "newcol"])}

        result = self.materializer.materialize_T(T=T, column_descriptions={}, S="")[-1]

        self.assertIn("t1", result)
        res_df = result["t1"].reset_index(drop=True)
        self.assertIn("newcol", res_df.columns)
        # The semantic column generation returns the LLM responses
        self.assertEqual(
            list(res_df["newcol"]),
            ["Result for: name: Alice; age: 30", "Result for: name: Alice; age: 30"],
        )

        self.assertTrue(len(self.materializer.prov_graph.nodes) == 4)
        t1_doc = next(
            doc
            for doc in self.materializer.state.intermediate_tables
            if doc.doc_id == "t1"
        )
        prov_graph_code_lines_1 = [
            self.materializer.prov_graph.ROOT_NODE_CODE,
            self.action_set.generate_pandas_read_csv_code(table_doc),
            self.action_set.generate_table_select_code(
                "t1", "test_ds.table_1", ["a", "b"]
            ),
            self.action_set.generate_semantic_col_generator_code(
                ["b"],
                t1_doc,
                "newcol",
                [
                    "Result for: name: Alice; age: 30",
                    "Result for: name: Alice; age: 30",
                ],
            ),
        ]
        self.assertEqual(
            self.materializer.prov_graph.get_graph_code(),
            "\n\n".join(prov_graph_code_lines_1),
        )

    # ------------------------------------------------------------------ update-mode tests

    def _run_fresh_t1_materialization(self) -> None:
        """Helper: run a fresh materialize_T that produces t1 with columns [a, b]."""
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore

        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=[table_doc])

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        result = self.materializer.materialize_T(T=T, column_descriptions={}, S="")
        self.assertIn("t1", result[-1])

    def test_saved_intermediates_populated_after_fresh_run(self):
        """_saved_intermediate_tables is populated with t1 after a fresh run."""
        self._run_fresh_t1_materialization()
        saved_ids = {doc.doc_id for doc in self.materializer._saved_intermediate_tables}
        self.assertIn("t1", saved_ids)

    def test_update_mode_preloads_saved_intermediates_into_state(self):
        """In update mode, _saved_intermediate_tables is pre-loaded into state.intermediate_tables
        before the LLM loop, so the LLM sees them as already-existing intermediates."""
        self._run_fresh_t1_materialization()

        # Verify t1 is in saved intermediates
        self.assertIn("t1", {d.doc_id for d in self.materializer._saved_intermediate_tables})

        # Second run in update mode: LLM just sees t1 already there and accepts it
        plan_update = f'{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"t1 already present, no extra work needed"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan_update}]}}']  # type: ignore

        T_updated = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(T=T_updated, column_descriptions={}, S="", update_mode=True)

        # state.intermediate_tables must have included t1 at some point — check via saved
        saved_ids_after = {d.doc_id for d in self.materializer._saved_intermediate_tables}
        self.assertIn("t1", saved_ids_after)

    def test_update_mode_preserves_intermediate_tables_in_db(self):
        """In update mode, prior intermediate tables are not dropped from the database."""
        self._run_fresh_t1_materialization()

        tables_before = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")["name"].tolist()
        )
        self.assertIn("t1", tables_before)

        # Update run: LLM does nothing extra (t1 already satisfies T schema)
        plan_noop = f'{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"nothing to do"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan_noop}]}}']  # type: ignore

        T_same = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(T=T_same, column_descriptions={}, S="", update_mode=True)

        tables_after = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")["name"].tolist()
        )
        # t1 must still be present
        self.assertIn("t1", tables_after)

    def test_fresh_mode_drops_previous_intermediate_tables(self):
        """A fresh materialize_T call drops all intermediate tables from the prior run."""
        self._run_fresh_t1_materialization()

        tables_after_first = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")["name"].tolist()
        )
        self.assertIn("t1", tables_after_first)

        # Second fresh run targeting t2 — t1 must be dropped
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2 tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        # retrieval mock already set from first run; reset to same table with new side_effect
        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_doc = Table(doc_id="table_1", retriever_type=RetrieverType.PNEUMA_RETRIEVER, content=table_df, metadata={})
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=[table_doc])

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(T=T2, column_descriptions={}, S="", update_mode=False)

        tables_after_second = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")["name"].tolist()
        )
        self.assertIn("t2", tables_after_second)
        self.assertNotIn("t1", tables_after_second)

    def test_reset_mode_is_identical_to_fresh(self):
        """update_mode=False (reset) drops previous intermediates, same as fresh."""
        self._run_fresh_t1_materialization()

        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        table_df = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        table_doc = Table(doc_id="table_1", retriever_type=RetrieverType.PNEUMA_RETRIEVER, content=table_df, metadata={})
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=[table_doc])

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        # update_mode=False is reset behavior
        self.materializer.materialize_T(T=T2, column_descriptions={}, S="", update_mode=False)

        tables_after = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")["name"].tolist()
        )
        self.assertIn("t2", tables_after)
        self.assertNotIn("t1", tables_after)

    def test_saved_intermediates_updated_after_each_run(self):
        """_saved_intermediate_tables reflects the most recently completed run."""
        self._run_fresh_t1_materialization()
        self.assertIn("t1", {d.doc_id for d in self.materializer._saved_intermediate_tables})
        self.assertNotIn("t2", {d.doc_id for d in self.materializer._saved_intermediate_tables})

        # Second fresh run targeting t2
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_doc = Table(doc_id="table_1", retriever_type=RetrieverType.PNEUMA_RETRIEVER, content=table_df, metadata={})
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=[table_doc])
        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(T=T2, column_descriptions={}, S="", update_mode=False)

        # Now _saved_intermediate_tables should reflect t2, not t1
        saved_ids = {d.doc_id for d in self.materializer._saved_intermediate_tables}
        self.assertIn("t2", saved_ids)
        self.assertNotIn("t1", saved_ids)

    def test_update_mode_with_empty_saved_intermediates_completes_normally(self):
        """update_mode=True with no prior run pre-loads nothing and still completes."""
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore

        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        table_doc = Table(doc_id="table_1", retriever_type=RetrieverType.PNEUMA_RETRIEVER, content=table_df, metadata={})
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=[table_doc])

        # No prior run — _saved_intermediate_tables is empty
        self.assertEqual(len(self.materializer._saved_intermediate_tables), 0)

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        result = self.materializer.materialize_T(T=T, column_descriptions={}, S="", update_mode=True)
        self.assertIn("t1", result[-1])

    def test_provenance_not_reset_in_update_mode(self):
        """In update mode, reset_materialization_nodes() is NOT called; prior prov nodes are kept."""
        self._run_fresh_t1_materialization()
        node_ids_after_first = set(self.prov_graph.nodes.keys())
        # There should be materializer nodes from the first run
        mat_nodes_first = [n for n in self.prov_graph.nodes.values() if n.source_retriever == RetrieverType.MATERIALIZER]
        self.assertTrue(len(mat_nodes_first) > 0)

        # Update run: LLM does a situational analysis only (t1 already satisfies T)
        plan_noop = f'{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"t1 already present"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan_noop}]}}']  # type: ignore

        T_same = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(T=T_same, column_descriptions={}, S="", update_mode=True)

        # All prior nodes must still be in the graph
        for node_id in node_ids_after_first:
            self.assertIn(node_id, self.prov_graph.nodes)

    def test_provenance_reset_in_fresh_mode(self):
        """In fresh mode, reset_materialization_nodes() IS called; prior MATERIALIZER prov nodes gone."""
        self._run_fresh_t1_materialization()
        mat_nodes_first = [n for n in self.prov_graph.nodes.values() if n.source_retriever == RetrieverType.MATERIALIZER]
        first_mat_node_ids = {n.id for n in mat_nodes_first}
        self.assertTrue(len(first_mat_node_ids) > 0)

        # Second fresh run targeting t2
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_doc = Table(doc_id="table_1", retriever_type=RetrieverType.PNEUMA_RETRIEVER, content=table_df, metadata={})
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=[table_doc])
        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(T=T2, column_descriptions={}, S="", update_mode=False)

        # The MATERIALIZER nodes from the first run must be gone
        for node_id in first_mat_node_ids:
            self.assertNotIn(node_id, self.prov_graph.nodes)

    def test_update_mode_applies_delta_to_existing_table(self):
        """Full update workflow: fresh run produces t1[a,b]; update run adds column c via SQL."""
        self._run_fresh_t1_materialization()

        # t1 with [a, b] is now in DB and in _saved_intermediate_tables
        t1_before = self.db_api.execute_query(self.user_id, self.chat_id, 'SELECT * FROM "t1";')
        self.assertListEqual(sorted(t1_before.columns.tolist()), ["a", "b"])

        # Update run: LLM uses query_executor to add column c.
        # QueryExecutor wraps the query as CREATE OR REPLACE TABLE "t1" AS <query>,
        # so pass only the SELECT part.
        add_col_query = "SELECT a, b, 99 AS c FROM t1"
        plan_update = (
            f'{{"action":"{ActionNames.QUERY_EXECUTOR.value}",'
            f'"args":{{"query":"{add_col_query}","assign_to":"t1"}}}}'
        )
        self.lm_api.llm._responses = [f'{{"plan": [{plan_update}]}}']  # type: ignore

        T_updated = {"t1": pd.DataFrame(columns=["a", "b", "c"])}
        result = self.materializer.materialize_T(T=T_updated, column_descriptions={}, S="", update_mode=True)

        self.assertIn("t1", result[-1])
        result_df = result[-1]["t1"]
        self.assertIn("c", result_df.columns)
        self.assertIn("a", result_df.columns)
        self.assertIn("b", result_df.columns)

    def test_rematerialize_cleans_previous_intermediate_tables(self):
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        plan3 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find other tables"]}}}}'
        plan4 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_2","columns":{{"a":"a","b":"b"}}}}}}}}'

        self.lm_api.llm._responses = [  # type: ignore
            f'{{"plan": [{plan1}, {plan2}]}}',
            f'{{"plan": [{plan3}, {plan4}]}}',
        ]

        table_df_1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_df_2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df_1.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        table_df_2.to_csv(Path(self.tmpdir) / "test_ds" / "table_2.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))

        table_doc_1 = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df_1,
            metadata={},
        )
        table_doc_2 = Table(
            doc_id="table_2",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df_2,
            metadata={},
        )

        self.action_set.retrieve_multi_topic_documents = MagicMock(
            side_effect=[[table_doc_1], [table_doc_2]]
        )

        baseline_tables = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )

        T1 = {"t1": pd.DataFrame(columns=["a", "b"])}
        result_1 = self.materializer.materialize_T(T=T1, column_descriptions={}, S="")
        self.assertIn("t1", result_1[-1])
        tables_after_first = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t1", tables_after_first)

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        result_2 = self.materializer.materialize_T(T=T2, column_descriptions={}, S="")
        self.assertIn("t2", result_2[-1])
        tables_after_second = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t2", tables_after_second)
        self.assertNotIn("t1", tables_after_second)
        self.assertEqual(tables_after_second - baseline_tables, {"t2"})


if __name__ == "__main__":
    unittest.main()
