# tests/pneuma_seeker/core/materializer/test_main.py
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from pneuma_seeker.provenance.graph import NodeKind, ProvenanceGraph
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
        self.dataset_name = "test_ds"
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

    def _assert_script_reproduces(self, target_table_ids: list[str]) -> str:
        """Runs the end-to-end script next to the source CSVs (in a fresh
        interpreter) and checks it rebuilds the workspace's target tables."""
        script = self.prov_graph.get_graph_code(target_table_ids)
        script_path = Path(self.tmpdir) / "e2e_script.py"
        script_path.write_text(script)
        run = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=self.tmpdir,
        )
        self.assertEqual(run.returncode, 0, f"{run.stderr}\n---\n{script}")
        for table_id in target_table_ids:
            expected_csv = Path(self.tmpdir) / f"expected_{table_id}.csv"
            self.db_api.execute_query(
                self.user_id, self.chat_id, f'SELECT * FROM "{table_id}"'
            ).to_csv(expected_csv, index=False)
            actual = pd.read_csv(Path(self.tmpdir) / "output" / f"{table_id}.csv")
            expected = pd.read_csv(expected_csv)
            # Row order of unordered SQL results is not part of the contract.
            pd.testing.assert_frame_equal(
                actual.sort_values(list(actual.columns)).reset_index(drop=True),
                expected.sort_values(list(expected.columns)).reset_index(drop=True),
            )
        return script

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

        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )[-1]

        self.assertIn("t1", result)
        pd.testing.assert_frame_equal(result["t1"].reset_index(drop=True), table_df)

        # root -> load test_ds.table_1 -> projection into t1
        self.assertTrue(len(self.materializer.prov_graph.nodes) == 3)
        script = self._assert_script_reproduces(["t1"])
        self.assertIn("table_1.csv  (table `table_1` of dataset `test_ds`)", script)
        self.assertIn(
            self.action_set.generate_load_table_code("table_1", "test_ds"), script
        )
        self.assertIn('CREATE OR REPLACE TABLE "t1" AS SELECT', script)
        steps = self.prov_graph.get_graph_explanation(["t1"])
        self.assertEqual(len(steps), 2)
        self.assertIn("Loads the table `table_1` of dataset `test_ds`", steps[0])
        self.assertIn("Projects a table", steps[1])

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

        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )[-1]

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
        context_nodes = [
            n for n in self.prov_graph.nodes.values() if n.kind == NodeKind.CONTEXT
        ]
        self.assertEqual(len(context_nodes), 1)
        self.assertEqual(
            context_nodes[0].python_code,
            self.action_set.generate_view_textual_document_code(
                "web_search_result", 'This is a web search result.'
            ),
        )
        # The web result does not feed t1, so the script leaves it out.
        script = self._assert_script_reproduces(["t1"])
        self.assertNotIn('This is a web search result.', script)

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

        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )[-1]

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
        context_nodes = [
            n for n in self.prov_graph.nodes.values() if n.kind == NodeKind.CONTEXT
        ]
        self.assertEqual(len(context_nodes), 1)
        self.assertEqual(
            context_nodes[0].python_code,
            self.action_set.generate_view_textual_document_code(
                "web_crawl_result", 'This is a web crawl result.'
            ),
        )
        # The web result does not feed t1, so the script leaves it out.
        script = self._assert_script_reproduces(["t1"])
        self.assertNotIn('This is a web crawl result.', script)

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

        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )[-1]

        self.assertIn("t1", result)
        res_df = result["t1"].reset_index(drop=True)
        self.assertIn("newcol", res_df.columns)
        # The semantic column generation returns the LLM responses
        self.assertEqual(
            list(res_df["newcol"]),
            ["Result for: name: Alice; age: 30", "Result for: name: Alice; age: 30"],
        )

        self.assertTrue(len(self.materializer.prov_graph.nodes) == 4)
        # The script replays the generated values instead of calling the LLM.
        script = self._assert_script_reproduces(["t1"])
        self.assertIn(
            self.action_set.generate_semantic_column_replay_code(
                "t1",
                ["b"],
                "newcol",
                [
                    ("10", "Result for: name: Alice; age: 30"),
                    ("20", "Result for: name: Alice; age: 30"),
                ],
            ),
            script,
        )

    def test_steps_using_values_from_a_web_result_are_linked_to_it(self):
        """A web search result pre-fetched by the Conductor (no provenance node
        yet) gets a node, and steps transcribing its values are linked to it."""
        web_text = Text(
            doc_id="web_search",
            retriever_type=RetrieverType.WEB_SEARCH,
            content="Paris has 2,140,526 inhabitants and Marseille has 873,076.",
            metadata={"query": "population of French cities"},
        )
        plan = [
            {
                "action": ActionNames.QUERY_EXECUTOR.value,
                "args": {
                    "query": "SELECT * FROM (VALUES ('Paris', 2140526), ('Marseille', 873076)) AS v(city, population)",
                    "assign_to": "cities",
                },
            },
            {
                "action": ActionNames.QUERY_EXECUTOR.value,
                "args": {
                    "query": "SELECT city FROM cities WHERE population > 1000 ORDER BY city",
                    "assign_to": "t1",
                },
            },
        ]
        self.lm_api.llm._responses = [json.dumps({"plan": plan})]  # type: ignore
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        pd.DataFrame({"a": [1]}).to_csv(Path(self.tmpdir) / "test_ds" / "unused.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))

        self.materializer.materialize_T(
            T={"t1": pd.DataFrame(columns=["city"])},
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            prefetched_web_search_result=web_text,
        )

        web_node = self.prov_graph.get_node_by_id(web_text.last_node_id or "")
        assert web_node is not None
        self.assertEqual(web_node.kind, NodeKind.CONTEXT)
        self.assertIn("population of French cities", web_node.description)
        cities_node = web_node.children[0]
        self.assertEqual(cities_node.output_table_id, "cities")
        self.assertIn("Uses values from the web search result", cities_node.description)
        t1_node = next(n for n in self.prov_graph.nodes.values() if n.output_table_id == "t1")
        # t1 only reads `cities`; it transcribes nothing from the web result.
        self.assertEqual(t1_node.parents, [cities_node])

        script = self._assert_script_reproduces(["t1"])
        self.assertIn("web_search_result = r", script)
        steps = self.prov_graph.get_graph_explanation(["t1"])
        self.assertEqual(len(steps), 3)
        self.assertIn("population of French cities", steps[0])

    def test_failed_replay_recording_does_not_fail_the_step(self):
        """If recording a model-driven step for replay fails, materialization
        still succeeds; the script stops at that step with an explanation."""
        self.config.ENABLE_SEMANTIC_COL_GEN = True
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        plan3 = f'{{"action":"{ActionNames.SEMANTIC_COLUMN_GENERATION.value}","args":{{"table_id":"t1","new_column_name":"newcol","relevant_columns":["b"],"instruction":"make new"}}}}'
        self.lm_api.llm._responses = [  # type: ignore
            f'{{"plan": [{plan1}, {plan2}, {plan3}]}}',
            "['x', 'y']",
        ]
        table_df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        table_df.to_csv(Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False)
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[
                Table(
                    doc_id="table_1",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=table_df,
                    metadata={},
                )
            ]
        )
        self.materializer._fetch_rows = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore

        result = self.materializer.materialize_T(
            T={"t1": pd.DataFrame(columns=["a", "b", "newcol"])},
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
        )[-1]

        self.assertEqual(list(result["t1"]["newcol"]), ["x", "y"])
        script = self.prov_graph.get_graph_code(["t1"])
        compile(script, "e2e_script.py", "exec")
        self.assertIn("raise RuntimeError(", script)
        self.assertIn("could not be recorded for replay (boom)", script)

    def _materialize_full_pipeline(self) -> list[str]:
        """Materializes a pipeline chaining every replayable step type from raw
        CSV files; returns the target table IDs."""
        self.config.ENABLE_SEMANTIC_COL_GEN = True
        self.config.ENABLE_SEMANTIC_JOIN = True

        # Raw CSV names/headers are normalized on ingestion (e.g. "Sales 2023.csv"
        # -> table sales_2023, "Amount ($)" -> amount_usd); the script must match.
        raw_dir = Path(self.tmpdir) / "test_ds"
        os.makedirs(raw_dir, exist_ok=True)
        raw_tables = {
            "Orders.csv": pd.DataFrame(
                {"Order ID": [3, 1, 2], "Customer ID": [20, 10, 10], "Amount ($)": [5.5, 10.0, 7.25]}
            ),
            "customers.csv": pd.DataFrame(
                {"id": [10, 20], "Customer Name": ["Alice Smith", "Bob Jones"]}
            ),
            "Sales 2023.csv": pd.DataFrame({"Region": ["N", "S"], "Revenue": [1000, 2000]}),
            "Sales 2024.csv": pd.DataFrame({"Region": ["N"], "Revenue": [4500]}),
            "unused.csv": pd.DataFrame({"x": [1]}),
        }
        for file_name, df in raw_tables.items():
            df.to_csv(raw_dir / file_name, index=False)
        self.db_api.ingest_dataset("test_ds", str(raw_dir))
        self.db_api.link_dataset_tables(self.user_id, self.chat_id, "test_ds")

        retrieved = [
            Table(
                doc_id=table_id,
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=self.db_api.execute_query(
                    self.user_id, self.chat_id, f'SELECT * FROM test_ds."{table_id}"'
                ),
                metadata={"dataset_name": "test_ds", "topic": "sales data"},
                path=f'test_ds."{table_id}"',
            )
            for table_id in ["orders", "customers", "sales_2023", "sales_2024", "unused"]
        ]
        self.action_set.retrieve_multi_topic_documents = MagicMock(return_value=retrieved)

        python_code = """df = db_api.execute_query(user_id, chat_id, 'SELECT year, SUM(revenue) AS revenue FROM all_sales GROUP BY year ORDER BY year')
df["revenue_k"] = df["revenue"] / 1000
db_api.register_temporary_df(user_id, chat_id, df, "t2_tmp")
db_api.execute_query(user_id, chat_id, 'CREATE OR REPLACE TABLE "t2" AS SELECT * FROM t2_tmp')"""
        plan = [
            {"action": ActionNames.TABLE_RETRIEVE.value, "args": {"prompts": ["sales"]}},
            {
                "action": ActionNames.TABLE_PROJECTION.value,
                "args": {
                    "t5": {
                        "id": "test_ds.customers",
                        "columns": {"cust_id": "id", "name": "customer_name"},
                    }
                },
            },
            {
                "action": ActionNames.EQUALITY_JOIN.value,
                "args": {
                    "left_table_id": "test_ds.orders",
                    "right_table_id": "t5",
                    "left_table_column_keys": ["customer_id"],
                    "right_table_column_keys": ["cust_id"],
                    "result_table_id": "orders_customers",
                },
            },
            {
                "action": ActionNames.TABLE_UNION.value,
                "args": {
                    "table_ids": ["re:^sales_\\d{4}$"],
                    "result_table_id": "all_sales",
                    "provenance_column_name": "year",
                    "provenance_regex": "(\\d{4})",
                },
            },
            {
                "action": ActionNames.QUERY_EXECUTOR.value,
                "args": {
                    "query": "SELECT left_order_id AS order_id, right_name AS name, left_amount_usd AS amount FROM orders_customers ORDER BY order_id",
                    "assign_to": "t1",
                },
            },
            {
                "action": ActionNames.PYTHON_EXECUTOR.value,
                "args": {"code": python_code, "assign_to": "t2"},
            },
            {
                "action": ActionNames.SEMANTIC_COLUMN_GENERATION.value,
                "args": {
                    "table_id": "t2",
                    "new_column_name": "label",
                    "relevant_columns": ["year"],
                    "instruction": "fiscal year label",
                },
            },
            {
                "action": ActionNames.SEMANTIC_JOIN.value,
                "args": {
                    "left_table_id": "t1",
                    "right_table_id": "customers",
                    "relevant_left_cols": ["name"],
                    "relevant_right_cols": ["customer_name"],
                    "joined_table_id": "t3",
                },
            },
            {
                "action": ActionNames.ENTITY_RESOLUTION.value,
                "args": {
                    "source_table_id": "t1",
                    "target_column": "name",
                    "output_mapping_table_id": "name_map",
                    "canonical_entities": ["Alice Smith", "Bob Jones"],
                },
            },
            {
                "action": ActionNames.QUERY_EXECUTOR.value,
                "args": {
                    "query": "SELECT t1.order_id, m.canonical_value AS canonical_name FROM t1 JOIN name_map AS m ON t1.name = m.original_value ORDER BY t1.order_id",
                    "assign_to": "t4",
                },
            },
        ]
        self.lm_api.llm._responses = [  # type: ignore
            json.dumps({"plan": plan}),
            "['FY23', 'FY24']",  # semantic column generation
        ]

        T = {
            "t1": pd.DataFrame(columns=["order_id", "name", "amount"]),
            "t2": pd.DataFrame(columns=["year", "revenue", "revenue_k", "label"]),
            "t3": pd.DataFrame(columns=["left_name", "right_customer_name", "similarity_score"]),
            "t4": pd.DataFrame(columns=["order_id", "canonical_name"]),
            "t5": pd.DataFrame(columns=["cust_id", "name"]),
        }
        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )[-1]
        self.assertEqual(set(result), set(T))
        self.assertEqual(
            list(result["t2"]["label"]), ["FY23", "FY24"]
        )
        return list(T)

    def test_e2e_script_reproduces_full_pipeline(self):
        """Every replayable step type, chained: the downloaded script rebuilds
        all target tables from the raw CSV files, offline."""
        targets = self._materialize_full_pipeline()

        script = self._assert_script_reproduces(targets)

        # Only the source tables the targets depend on are required.
        for table_id in ["orders", "customers", "sales_2023", "sales_2024"]:
            self.assertIn(f"#     - {table_id}.csv", script)
        self.assertNotIn("unused", script)

        steps = self.prov_graph.get_graph_explanation(targets)
        # 4 loads + 9 materialization steps (table_retrieve adds no step)
        self.assertEqual(len(steps), 13)
        for step_md in steps:
            self.assertRegex(step_md, r"^# Step \d+\n\n\S")
        self.assertEqual(self.prov_graph.get_snapshot_table_ids(targets), [])

    def test_e2e_script_loads_snapshots_of_steps_over_the_replay_cap(self):
        """Model-driven steps with more decisions than the cap are loaded from
        CSV snapshots (shipped in "Download views"), and the inputs that only
        fed those steps are no longer required."""
        self.config.PROVENANCE_REPLAY_MAX_ROWS = 1
        targets = self._materialize_full_pipeline()

        snapshots = self.prov_graph.get_snapshot_table_ids(targets)
        self.assertEqual(sorted(snapshots), ["name_map", "t2", "t3"])
        # Simulate unzipping "Download views" next to the script.
        for table_id in snapshots:
            self.db_api.execute_query(
                self.user_id, self.chat_id, f'SELECT * FROM "{table_id}"'
            ).to_csv(Path(self.tmpdir) / f"{table_id}.csv", index=False)

        script = self._assert_script_reproduces(targets)

        for table_id in snapshots:
            self.assertIn(f"load_snapshot_table({table_id!r})", script)
            self.assertIn(f"#     - {table_id}.csv\n", script)
        # Sales tables only fed t2, which is now loaded from its snapshot.
        for table_id in ["orders", "customers"]:
            self.assertIn(f"#     - {table_id}.csv  (", script)
        for table_id in ["sales_2023", "sales_2024"]:
            self.assertNotIn(f"#     - {table_id}.csv", script)

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
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )
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
        self.assertIn(
            "t1", {d.doc_id for d in self.materializer._saved_intermediate_tables}
        )

        # Second run in update mode: LLM just sees t1 already there and accepts it
        plan_update = f'{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"t1 already present, no extra work needed"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan_update}]}}']  # type: ignore

        T_updated = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T_updated,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=True,
        )

        # state.intermediate_tables must have included t1 at some point — check via saved
        saved_ids_after = {
            d.doc_id for d in self.materializer._saved_intermediate_tables
        }
        self.assertIn("t1", saved_ids_after)

    def test_update_mode_preserves_intermediate_tables_in_db(self):
        """In update mode, prior intermediate tables are not dropped from the database."""
        self._run_fresh_t1_materialization()

        tables_before = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t1", tables_before)

        # Update run: LLM does nothing extra (t1 already satisfies T schema)
        plan_noop = f'{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"nothing to do"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan_noop}]}}']  # type: ignore

        T_same = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T_same,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=True,
        )

        tables_after = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        # t1 must still be present
        self.assertIn("t1", tables_after)

    def test_fresh_mode_drops_previous_intermediate_tables(self):
        """A fresh materialize_T call drops all intermediate tables from the prior run."""
        self._run_fresh_t1_materialization()

        tables_after_first = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t1", tables_after_first)

        # Second fresh run targeting t2 — t1 must be dropped
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2 tables"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        # retrieval mock already set from first run; reset to same table with new side_effect
        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T2,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=False,
        )

        tables_after_second = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
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
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        # update_mode=False is reset behavior
        self.materializer.materialize_T(
            T=T2,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=False,
        )

        tables_after = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t2", tables_after)
        self.assertNotIn("t1", tables_after)

    def test_saved_intermediates_updated_after_each_run(self):
        """_saved_intermediate_tables reflects the most recently completed run."""
        self._run_fresh_t1_materialization()
        self.assertIn(
            "t1", {d.doc_id for d in self.materializer._saved_intermediate_tables}
        )
        self.assertNotIn(
            "t2", {d.doc_id for d in self.materializer._saved_intermediate_tables}
        )

        # Second fresh run targeting t2
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )
        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T2,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=False,
        )

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
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        # No prior run — _saved_intermediate_tables is empty
        self.assertEqual(len(self.materializer._saved_intermediate_tables), 0)

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        result = self.materializer.materialize_T(
            T=T,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=True,
        )
        self.assertIn("t1", result[-1])

    def test_provenance_not_reset_in_update_mode(self):
        """In update mode, reset_materialization_nodes() is NOT called; prior prov nodes are kept."""
        self._run_fresh_t1_materialization()
        node_ids_after_first = set(self.prov_graph.nodes.keys())
        # There should be materializer nodes from the first run
        mat_nodes_first = [
            n
            for n in self.prov_graph.nodes.values()
            if n.source_retriever == RetrieverType.MATERIALIZER
        ]
        self.assertTrue(len(mat_nodes_first) > 0)

        # Update run: LLM does a situational analysis only (t1 already satisfies T)
        plan_noop = f'{{"action":"{ActionNames.SITUATIONAL_ANALYSIS.value}","args":{{"message":"t1 already present"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan_noop}]}}']  # type: ignore

        T_same = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T_same,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=True,
        )

        # All prior nodes must still be in the graph
        for node_id in node_ids_after_first:
            self.assertIn(node_id, self.prov_graph.nodes)

    def test_provenance_reset_in_fresh_mode(self):
        """In fresh mode, reset_materialization_nodes() IS called; prior MATERIALIZER prov nodes gone."""
        self._run_fresh_t1_materialization()
        mat_nodes_first = [
            n
            for n in self.prov_graph.nodes.values()
            if n.source_retriever == RetrieverType.MATERIALIZER
        ]
        first_mat_node_ids = {n.id for n in mat_nodes_first}
        self.assertTrue(len(first_mat_node_ids) > 0)

        # Second fresh run targeting t2
        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find t2"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t2":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        table_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table_doc = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_df,
            metadata={},
        )
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )
        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T2,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=False,
        )

        # The MATERIALIZER nodes from the first run must be gone
        for node_id in first_mat_node_ids:
            self.assertNotIn(node_id, self.prov_graph.nodes)

    def test_update_mode_applies_delta_to_existing_table(self):
        """Full update workflow: fresh run produces t1[a,b]; update run adds column c via SQL."""
        self._run_fresh_t1_materialization()

        # t1 with [a, b] is now in DB and in _saved_intermediate_tables
        t1_before = self.db_api.execute_query(
            self.user_id, self.chat_id, 'SELECT * FROM "t1";'
        )
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
        result = self.materializer.materialize_T(
            T=T_updated,
            column_descriptions={},
            S="",
            dataset_name=self.dataset_name,
            update_mode=True,
        )

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
        result_1 = self.materializer.materialize_T(
            T=T1, column_descriptions={}, S="", dataset_name=self.dataset_name
        )
        self.assertIn("t1", result_1[-1])
        tables_after_first = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t1", tables_after_first)

        T2 = {"t2": pd.DataFrame(columns=["a", "b"])}
        result_2 = self.materializer.materialize_T(
            T=T2, column_descriptions={}, S="", dataset_name=self.dataset_name
        )
        self.assertIn("t2", result_2[-1])
        tables_after_second = set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")[
                "name"
            ].tolist()
        )
        self.assertIn("t2", tables_after_second)
        self.assertNotIn("t1", tables_after_second)
        self.assertEqual(tables_after_second - baseline_tables, {"t2"})

    def test_check_completion_extra_columns_allowed(self):
        """Extra columns in a materialized table do not prevent completion."""
        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.state.T = T
        self.materializer.state.add_intermediate_table(
            Table(
                doc_id="t1",
                retriever_type=RetrieverType.MATERIALIZER,
                content=pd.DataFrame({"a": [1], "b": [2], "extra": [3]}),
                metadata={},
            )
        )
        self.assertTrue(self.materializer._check_completion(T))

    def test_check_completion_missing_required_columns_not_complete(self):
        """A materialized table missing required columns does not satisfy completion."""
        T = {"t1": pd.DataFrame(columns=["a", "b", "required_col"])}
        self.materializer.state.T = T
        self.materializer.state.add_intermediate_table(
            Table(
                doc_id="t1",
                retriever_type=RetrieverType.MATERIALIZER,
                content=pd.DataFrame({"a": [1], "b": [2]}),  # missing required_col
                metadata={},
            )
        )
        self.assertFalse(self.materializer._check_completion(T))

    def test_plan_parse_error_consumes_a_step_and_still_recovers(self):
        """A malformed LLM plan now consumes a step (so a run of invalid JSON can't stall
        the loop indefinitely), but the materialization still recovers and completes as
        long as enough steps remain in the budget."""
        self.materializer.config.MAX_MATERIALIZER_STEPS = 2  # 1 wasted + 1 real step

        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{"prompts":["find"]}}}}'
        plan2 = f'{{"action":"{ActionNames.TABLE_PROJECTION.value}","args":{{"t1":{{"id":"test_ds.table_1","columns":{{"a":"a","b":"b"}}}}}}}}'
        self.lm_api.llm._responses = [  # type: ignore
            '{"plan": "not a list"}',  # bad — consumes step 1
            f'{{"plan": [{plan1}, {plan2}]}}',  # good — completes within step 2
        ]

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
        self.action_set.retrieve_multi_topic_documents = MagicMock(
            return_value=[table_doc]
        )

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        result = self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )
        self.assertIn("t1", result[-1])

    def test_plan_parse_error_terminates_within_step_budget(self):
        """Repeated invalid JSON must not stall the loop: each parse failure now consumes
        a step, guaranteeing termination within MAX_MATERIALIZER_STEPS."""
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        pd.DataFrame({"a": [1]}).to_csv(
            Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False
        )
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))

        self.materializer.config.MAX_MATERIALIZER_STEPS = 1
        self.lm_api.llm._responses = ['{"plan": "not a list"}'] * 5  # type: ignore

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )

        self.assertEqual(
            len(self.lm_api.llm._responses),
            4,
            "Only one planning call should have been made before the step budget was exhausted",
        )

    def test_action_error_aborts_remaining_plan(self):
        """An action error aborts remaining actions in the same step, so a follow-up
        action is never executed against a failed prerequisite."""
        os.makedirs(Path(self.tmpdir) / "test_ds", exist_ok=True)
        pd.DataFrame({"a": [1]}).to_csv(
            Path(self.tmpdir) / "test_ds" / "table_1.csv", index=False
        )
        self.db_api.ingest_dataset("test_ds", str(Path(self.tmpdir) / "test_ds"))

        plan1 = f'{{"action":"{ActionNames.TABLE_RETRIEVE.value}","args":{{}}}}'  # missing 'prompts' -> ERROR
        plan2 = f'{{"action":"{ActionNames.WEB_SEARCH.value}","args":{{"prompt":"x"}}}}'
        self.lm_api.llm._responses = [f'{{"plan": [{plan1}, {plan2}]}}']  # type: ignore
        self.materializer.config.MAX_MATERIALIZER_STEPS = 1

        self.action_set.retrieve_documents = MagicMock(return_value=[])

        T = {"t1": pd.DataFrame(columns=["a", "b"])}
        self.materializer.materialize_T(
            T=T, column_descriptions={}, S="", dataset_name=self.dataset_name
        )

        self.action_set.retrieve_documents.assert_not_called()
        abort_messages = [
            m["content"]
            for m in self.materializer.llm_messages
            if isinstance(m["content"], str) and "were skipped because" in m["content"]
        ]
        self.assertTrue(
            abort_messages, "Expected an abort note in llm_messages after the error"
        )


if __name__ == "__main__":
    unittest.main()
