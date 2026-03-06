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
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")
            ["name"]
            .tolist()
        )

        T1 = {"t1": pd.DataFrame(columns=["a", "b"])}
        result_1 = self.materializer.materialize_T(T=T1, column_descriptions={}, S="")
        self.assertIn("t1", result_1[-1])
        tables_after_first = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")
            ["name"]
            .tolist()
        )
        self.assertIn("t1", tables_after_first)

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        result_2 = self.materializer.materialize_T(T=T2, column_descriptions={}, S="")
        self.assertIn("t2", result_2[-1])
        tables_after_second = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")
            ["name"]
            .tolist()
        )
        self.assertIn("t2", tables_after_second)
        self.assertNotIn("t1", tables_after_second)
        self.assertEqual(tables_after_second - baseline_tables, {"t2"})


if __name__ == "__main__":
    unittest.main()
