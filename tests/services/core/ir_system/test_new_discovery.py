# tests/services/core/ir_system/test_new_discovery.py
import json
import logging
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import duckdb

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.ir_system.retriever.impl.new_discovery import (
    NewDiscovery,
    build_candidates,
    cell,
    table_text,
)
from pneuma_seeker.services.core.ir_system.retriever.impl.pneuma_retriever import PneumaRetriever
from pneuma_seeker.services.core.ir_system.retriever.retriever_factory import RetrieverFactory
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table


class JudgeLLM:
    """A candidate table is relevant if its representation mentions `marker`."""

    def __init__(self, marker: str = "rain", fail: bool = False):
        self.marker = marker
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()

    def chat(self, messages, option=None):
        if self.fail:
            raise RuntimeError("model unavailable")
        with self._lock:
            self.calls += 1
        body = messages[1]["content"].split("Candidates:", 1)[1]
        parts = re.split(r"\n?\[(c\d+)\] ", body)[1:]
        verdict = {parts[i]: self.marker in parts[i + 1].lower() for i in range(0, len(parts), 2)}
        yield json.dumps(verdict)


def make_dataset(root: Path, name: str, tables: dict[str, str]) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(root / name / f"{name}.db"))
    for sql in tables.values():
        con.execute(sql)
    con.close()


TABLES = {
    "rain_boston": "CREATE TABLE rain_boston AS SELECT * FROM (VALUES ('2020', 1.95), ('2021', 2.10)) t(year, jul)",
    "rain_chatham": "CREATE TABLE rain_chatham AS SELECT * FROM (VALUES ('2020', 3.10)) t(year, jul)",
    "beach_status": "CREATE TABLE beach_status AS SELECT * FROM (VALUES ('Carson', 'closed')) t(beach, status)",
    "aqi_county": "CREATE TABLE aqi_county AS SELECT * FROM (VALUES ('Suffolk', 192)) t(county, good_days)",
}


class RepresentationTests(unittest.TestCase):
    def test_r2_text_and_cell_truncation(self):
        table = {
            "name": "t", "dataset_name": "d",
            "columns": [{"name": "a"}, {"name": "b"}],
            "sample_rows": [["x" * 60, None]],
        }
        text = table_text(table, max_columns=50)
        self.assertEqual(text.splitlines()[:2], ["Table: t (dataset d)", "Schema: a | b"])
        self.assertEqual(len(cell("x" * 60)), 40)
        self.assertTrue(cell("x" * 60).endswith("…"))

    def test_identical_columns_form_one_family(self):
        cols = [{"name": "year", "type": "VARCHAR"}, {"name": "jul", "type": "DOUBLE"}]
        tables = [
            {"name": "rain_boston", "dataset_name": "env", "columns": cols, "sample_rows": []},
            {"name": "rain_chatham", "dataset_name": "env", "columns": cols, "sample_rows": []},
            {"name": "beach", "dataset_name": "env", "columns": [{"name": "b", "type": "VARCHAR"}], "sample_rows": []},
        ]
        candidates = build_candidates("env", tables, 50, 100)
        self.assertEqual([c["tables"] for c in candidates], [["rain_boston", "rain_chatham"], ["beach"]])
        self.assertTrue(candidates[0]["content"].startswith("Family of 2 tables with the same columns"))


class NewDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.config = Config(".env.test")
        self.config.LLM_PATH = "mock"
        self.config.EMBED_MODEL_PATH = "mock"
        self.config.RETRIEVAL_INDEX_ROOT = str(self.tmp / "indices")
        self.logger = logging.getLogger("test_new_discovery")
        make_dataset(self.tmp / "datasets", "env", TABLES)
        self.db_api = DBAPI(
            self.config, self.logger,
            dataset_db_path=str(self.tmp / "datasets"), workspace_db_path=str(self.tmp / "ws"),
        )

    def retriever(self, llm) -> NewDiscovery:
        return NewDiscovery("u1", "c1", self.config, self.db_api, llm)

    def current_version(self) -> str:
        return (self.tmp / "indices" / "discovery" / "env" / "CURRENT").read_text()

    def test_returns_relevant_tables_as_internal_table_documents(self):
        docs = self.retriever(JudgeLLM("rain")).retrieve("July rainfall in Boston", "env", 10, True, 3)
        # The family (identical columns) is one candidate, returned as all its tables.
        self.assertEqual([d.doc_id for d in docs], ["rain_boston", "rain_chatham"])
        doc = docs[0]
        self.assertIsInstance(doc, Table)
        self.assertEqual(doc.retriever_type, RetrieverType.PNEUMA_RETRIEVER)
        self.assertEqual(doc.path, 'env."rain_boston"')
        self.assertEqual(doc.metadata["dataset_name"], "env")
        self.assertEqual(list(doc.content.columns), ["year", "jul"])
        self.assertLessEqual(len(doc.content), 3)

    def test_index_is_persisted_reused_and_rebuilt_when_the_dataset_changes(self):
        retriever = self.retriever(JudgeLLM("air"))
        retriever.retrieve("good air days", "env", 10, True, 3)
        version = self.current_version()

        # Same dataset: a new retriever (a new chat, a restart) reuses the index on disk.
        self.retriever(JudgeLLM("air")).retrieve("good air days", "env", 10, True, 3)
        self.assertEqual(self.current_version(), version)

        # The dataset changes: the index is rebuilt and sees the new table.
        self.db_api.detach_datasets_except("u1", "c1", set())
        con = duckdb.connect(str(self.tmp / "datasets" / "env" / "env.db"))
        con.execute("CREATE TABLE air_quality AS SELECT * FROM (VALUES ('Suffolk', 'good')) t(county, air)")
        con.close()
        docs = self.retriever(JudgeLLM("air")).retrieve("good air days", "env", 10, True, 3)
        self.assertNotEqual(self.current_version(), version)
        self.assertIn("air_quality", [d.doc_id for d in docs])

    def test_falls_back_to_bm25_top_k_when_the_llm_fails(self):
        docs = self.retriever(JudgeLLM(fail=True)).retrieve("beach status closed", "env", 2, True, 3)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].doc_id, "beach_status")  # best BM25 match first

    def test_nothing_relevant(self):
        llm = JudgeLLM("no-such-marker")
        self.assertEqual(self.retriever(llm).retrieve("rainfall", "env", 10, True, 3), [])
        self.assertGreater(llm.calls, 0)


class MethodSettingTests(unittest.TestCase):
    def test_defaults_and_unknown_values(self):
        saved = {k: os.environ.pop(k, None) for k in ("TABLE_DISCOVERY_METHOD", "TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD")}
        try:
            config = Config(".env.test")
            self.assertEqual(config.TABLE_DISCOVERY_METHOD, "pneuma_retriever")
            self.assertEqual(config.TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD, "m1")
            for name, bad in (("TABLE_DISCOVERY_METHOD", "pneuma"), ("TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD", "hybrid")):
                os.environ[name] = bad
                with self.assertRaises(ValueError):
                    Config(".env.test")
                del os.environ[name]
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(".env.test")
        self.config.LLM_PATH = "mock"
        self.config.EMBED_MODEL_PATH = "mock"
        tmp = tempfile.mkdtemp()
        self.db_api = DBAPI(self.config, logging.getLogger("t"), dataset_db_path=f"{tmp}/d", workspace_db_path=f"{tmp}/w")

    def factory(self, method: str) -> RetrieverFactory:
        self.config.TABLE_DISCOVERY_METHOD = method
        from pneuma_seeker.services.core.api.language_model import LanguageModelAPI

        return RetrieverFactory(
            "u1", "c1", self.config, self.db_api, LanguageModelAPI(self.config, logging.getLogger("t"))
        )

    def test_default_uses_the_pneuma_retriever(self):
        factory = self.factory("pneuma_retriever")
        self.assertIsInstance(factory.get_retriever_for_retrieval(RetrieverType.PNEUMA_RETRIEVER), PneumaRetriever)

    def test_m1_routes_retrieval_but_not_indexing(self):
        factory = self.factory("m1")
        self.assertIsInstance(factory.get_retriever_for_retrieval(RetrieverType.PNEUMA_RETRIEVER), NewDiscovery)
        self.assertIsInstance(factory.get_retriever(RetrieverType.PNEUMA_RETRIEVER), PneumaRetriever)
        # Other retrievers are unaffected.
        self.assertIs(
            factory.get_retriever_for_retrieval(RetrieverType.ENUMERATOR),
            factory.get_retriever(RetrieverType.ENUMERATOR),
        )


if __name__ == "__main__":
    unittest.main()
