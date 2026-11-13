# tests/services/core/conductor/test_harness.py
"""The Conductor/Materializer contract: T's column sources, need resolutions,
append-only retrieval with a spotlight, and materialization that either
completes or reports why not."""
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import json
import unittest
from unittest.mock import MagicMock

import pandas as pd

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.main import Conductor
from pneuma_seeker.services.core.materializer.main import MaterializationIncomplete
from pneuma_seeker.services.core.retrieved_tables import render_retrieved_tables, source_tables
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionExecutionStatus, ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table


def _table(name: str, dataset: str = "env", columns=("year", "jul"), topic: str = "") -> Table:
    metadata = {"dataset_name": dataset, "description": f"{name} data"}
    if topic:
        metadata["topic"] = topic
    return Table(
        doc_id=name,
        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
        content=pd.DataFrame({c: [f"{c}-value"] for c in columns}),
        metadata=metadata,
    )


class RenderingTests(unittest.TestCase):
    def test_spotlight_tables_get_rows_others_one_line(self):
        docs = [_table("rain"), _table("beach", columns=("beach", "status"))]
        text = render_retrieved_tables(docs, {"env.rain"}, {"env.rain": ["July rainfall"], "env.beach": ["beaches"]})
        rain_line, beach_line = text.split("\n- ")
        self.assertIn("jul-value", rain_line)  # sample rows
        self.assertIn("(found by: July rainfall)", rain_line)
        self.assertNotIn("status-value", beach_line)
        self.assertIn("env.beach: beach data; columns: beach, status (found by: beaches) [no sample rows shown]", beach_line)

    def test_tables_with_identical_columns_are_shown_once(self):
        docs = [_table("rain_boston", topic="q"), _table("rain_chatham"), _table("beach", columns=("b",))]
        text = render_retrieved_tables(docs, {"env.rain_chatham"})
        self.assertEqual(text.count("jul-value"), 1)
        self.assertIn("Also 1 more table(s) with the same columns: env.rain_chatham", text)
        compact = render_retrieved_tables(docs, set())
        self.assertIn("2 tables with the same columns (env.rain_boston, env.rain_chatham)", compact)

    def test_source_patterns(self):
        docs = [_table("rain_boston"), _table("rain_chatham"), _table("beach", columns=("b",))]
        used, unmatched = source_tables(
            {"t": {"jul": ["env.rain_*.jul"], "beach": ["beach.b"], "x": ["env.nowhere.c"], "y": []}}, docs
        )
        self.assertEqual(used, {"env.rain_boston", "env.rain_chatham", "env.beach"})
        self.assertEqual(unmatched, ["env.nowhere.c"])


class ConductorHarnessTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(".env.test")
        self.config.LLM_PATH = "mock"
        self.config.EMBED_MODEL_PATH = "mock"
        logger = logging.getLogger("test_harness")
        logger.setLevel(logging.ERROR)
        tmp = tempfile.mkdtemp()
        self.conductor = Conductor(
            "u1", "c1", self.config, logger, ProvenanceGraph(logger),
            DBAPI(self.config, logger, dataset_db_path=str(Path(tmp, "d")), workspace_db_path=str(Path(tmp, "w"))),
            LanguageModelAPI(self.config, logger),
            frontend_callback=lambda _: None,
        )
        self.conductor.action_set.discover_join_paths = MagicMock(return_value="")  # type: ignore
        self.conductor._needs = ["What unit is rainfall in?", "Which months are summer?"]

    def manipulate(self, **args):
        return self.conductor._handle_state_manipulation("q", args, "env")

    T = {"summer": ["region", "jul"]}
    DESCRIPTIONS = {"summer": {"region": "station", "jul": "July rainfall"}}

    # -- T's column sources -------------------------------------------------

    def test_T_requires_column_sources_for_every_column(self):
        msg, status = self.manipulate(T=self.T, column_descriptions=self.DESCRIPTIONS)
        self.assertEqual(status, ActionExecutionStatus.ERROR)
        self.assertIn("column_sources", msg)
        msg, status = self.manipulate(
            T=self.T, column_descriptions=self.DESCRIPTIONS, column_sources={"summer": {"jul": ["env.rain.jul"]}}
        )
        self.assertEqual(status, ActionExecutionStatus.ERROR)
        self.assertIn("summer.region", msg)

    def test_empty_sources_are_allowed_and_unmatched_ones_warned(self):
        self.conductor.retrieved_tables = [_table("rain")]
        msg, status = self.manipulate(
            T=self.T, column_descriptions=self.DESCRIPTIONS,
            column_sources={"summer": {"region": [], "jul": ["env.rain.jul", "env.stations.name"]}},
        )
        self.assertEqual(status, ActionExecutionStatus.SUCCESS)
        self.assertIn("match no retrieved or external table: env.stations.name", msg)
        self.assertEqual(self.conductor.state.column_sources["summer"]["jul"], ["env.rain.jul", "env.stations.name"])

    # -- Need resolutions ---------------------------------------------------

    def test_T_and_S_need_every_need_resolved(self):
        sources = {"summer": {"region": [], "jul": []}}
        msg, status = self.manipulate(T=self.T, column_descriptions=self.DESCRIPTIONS, column_sources=sources)
        self.assertEqual(status, ActionExecutionStatus.SUCCESS)  # T alone: needs may stay open

        msg, status = self.manipulate(S="result = 1", need_resolutions={"1": "evidence: rain.jul is in inches"})
        self.assertEqual(status, ActionExecutionStatus.ERROR)
        self.assertIn("Unresolved information needs: 2", msg)
        self.assertEqual(self.conductor.state.S, "")  # rejected as a whole

        msg, status = self.manipulate(
            S="result = 1",
            need_resolutions={"1": "evidence: rain.jul is in inches", "2": "assumption: June-August"},
        )
        self.assertEqual(status, ActionExecutionStatus.SUCCESS)
        self.assertEqual(self.conductor._need_resolutions[2], "assumption: June-August")

    def test_resolutions_accumulate_and_are_validated(self):
        _, status = self.manipulate(S="x", need_resolutions={"3": "nope"})
        self.assertEqual(status, ActionExecutionStatus.ERROR)  # there are only 2 needs
        self.manipulate(S="x", need_resolutions={"1": "evidence"})  # T missing: allowed
        _, status = self.manipulate(
            T=self.T, column_descriptions=self.DESCRIPTIONS,
            column_sources={"summer": {"region": [], "jul": []}}, need_resolutions={"2": "assumption"},
        )
        self.assertEqual(status, ActionExecutionStatus.SUCCESS)

    # -- Append-only retrieval and the spotlight ----------------------------

    def test_retrieval_accumulates_and_records_queries(self):
        batches = iter([[_table("rain", topic="July rainfall")], [_table("beach", columns=("b",), topic="beaches")]])
        self.conductor.action_set.retrieve_multi_topic_documents = MagicMock(side_effect=lambda *a: next(batches))  # type: ignore
        self.conductor._handle_table_retrieve("q", {"prompts": ["July rainfall"]}, "env")
        self.conductor._handle_table_retrieve("q", {"prompts": ["beaches"]}, "env")

        self.assertEqual([t.doc_id for t in self.conductor.retrieved_tables], ["rain", "beach"])
        self.assertEqual(self.conductor.found_by, {"env.rain": ["July rainfall"], "env.beach": ["beaches"]})
        self.assertEqual(self.conductor.latest_retrieval, {"env.beach"})
        self.assertEqual(self.conductor._spotlight(), {"env.beach"})

        # Once T's sources use the earlier table, it is back in the spotlight.
        self.manipulate(T=self.T, column_descriptions=self.DESCRIPTIONS,
                        column_sources={"summer": {"region": [], "jul": ["env.rain.jul"]}})
        self.assertEqual(self.conductor._spotlight(), {"env.beach", "env.rain"})

    def test_exact_names_load_without_search(self):
        self.conductor.dataset_ctx = self.conductor.dataset_ctx.single("env")
        search = MagicMock(return_value=[])
        self.conductor.action_set.retrieve_multi_topic_documents = search  # type: ignore
        self.conductor.action_set.load_tables_by_name = MagicMock(return_value=([_table("rain")], ["ghost"]))  # type: ignore
        msg, status = self.conductor._handle_table_retrieve("q", {"tables": ["env.rain", "ghost"]}, "env")

        self.assertEqual(status, ActionExecutionStatus.SUCCESS)
        search.assert_not_called()
        self.assertIn("Not found: ghost", msg)
        self.assertEqual(self.conductor.found_by["env.rain"], ["by name"])

    # -- The Materializer contract -------------------------------------------

    def test_materializer_gets_only_the_tables_T_names(self):
        self.conductor.retrieved_tables = [_table("rain"), _table("beach", columns=("b",))]
        self.manipulate(T=self.T, column_descriptions=self.DESCRIPTIONS,
                        column_sources={"summer": {"region": [], "jul": ["env.rain.jul"]}})
        materialize = MagicMock(return_value=([], None, None, None, {"summer": pd.DataFrame({"region": ["x"], "jul": [1]})}))
        self.conductor.materializer.materialize_T = materialize  # type: ignore

        _, status = self.conductor._handle_materializer("q", {}, "env")
        self.assertEqual(status, ActionExecutionStatus.SUCCESS)
        provided = materialize.call_args.args[6]
        self.assertEqual([t.doc_id for t in provided], ["rain"])
        self.assertEqual(materialize.call_args.kwargs["column_sources"]["summer"]["jul"], ["env.rain.jul"])
        self.assertTrue(self.conductor.state.is_T_materialized)

    def test_incomplete_materialization_is_an_error_and_T_stays_unmaterialized(self):
        self.manipulate(T=self.T, column_descriptions=self.DESCRIPTIONS,
                        column_sources={"summer": {"region": [], "jul": []}})
        self.conductor.materializer.materialize_T = MagicMock(  # type: ignore
            side_effect=MaterializationIncomplete("The Materializer reported a blocker: no jul column")
        )
        msg, status = self.conductor._handle_materializer("q", {}, "env")
        self.assertEqual(status, ActionExecutionStatus.ERROR)
        self.assertIn("no jul column", msg)
        self.assertIn("Revise T or its column sources", msg)
        self.assertFalse(self.conductor.state.is_T_materialized)

    def test_discovery_actions_belong_to_the_conductor(self):
        registry = self.conductor.action_set.registry
        materializer_actions = {a.action_name for a in registry.get_for_agent(AgentType.MATERIALIZER)}
        self.assertTrue({ActionNames.TABLE_RETRIEVE, ActionNames.TABLE_ENUMERATION}.isdisjoint(materializer_actions))
        self.assertIn(ActionNames.REPORT_BLOCKER, materializer_actions)

    def test_drop_relation_drops_tables_and_views_by_their_type(self):
        db = self.conductor.db_api
        db.execute_query("u1", "c1", "CREATE TABLE tmp_t AS SELECT 1 AS x")
        db.execute_query("u1", "c1", "CREATE VIEW tmp_v AS SELECT 1 AS x")
        for name in ("tmp_t", "tmp_v", "never_existed"):
            db.drop_relation("u1", "c1", name)  # raises on a type mismatch, which used to be logged as a warning
        names = db.execute_query("u1", "c1", "SELECT table_name FROM information_schema.tables")["table_name"].tolist()
        self.assertFalse({"tmp_t", "tmp_v"} & set(names))

    def test_timing_never_reports_negative_cpu(self):
        from pneuma_seeker.shared.logger import format_timing

        self.assertEqual(format_timing(10.0, 4.0), "10.00s (CPU: 6.00s, LLM: 4.00s)")
        self.assertEqual(format_timing(11.72, 20.17), "11.72s (LLM calls: 20.17s in total, partly in parallel)")

    def test_table_enumeration_only_with_the_pneuma_retriever(self):
        self.config.ENABLE_TABLE_ENUMERATION = True  # TABLE_DISCOVERY_METHOD=pneuma_retriever
        self.assertTrue(self.conductor.action_set.is_valid_conductor_action(ActionNames.TABLE_ENUMERATION.value))
        self.config.ENABLE_TABLE_ENUMERATION = False  # TABLE_DISCOVERY_METHOD=m1
        self.assertFalse(self.conductor.action_set.is_valid_conductor_action(ActionNames.TABLE_ENUMERATION.value))
        prompt = self.conductor.prompt_factory.get_sys_prompt()
        self.assertNotIn(ActionNames.TABLE_ENUMERATION.value, prompt)
        self.assertIn(ActionNames.TABLE_RETRIEVE.value, prompt)


class LoadByNameTests(unittest.TestCase):
    def test_exact_names_and_patterns(self):
        import re
        from types import SimpleNamespace
        from pneuma_seeker.services.core.action_set.main import ActionSet

        names = {"env": ["rain_2001", "rain_2002", "rainfall", "beach"], "wild": ["rain_2001"]}

        def retrieve(regex, _type, ds, _k, _sample_only, _size):
            return [_table(n, ds) for n in names[ds] if re.match(regex, n)]

        fake = SimpleNamespace(retrieve_documents=retrieve)
        docs, missing = ActionSet.load_tables_by_name(fake, ["rain_*", "rain_2001", "beach", "ghost_*"], ["env"], 3)
        self.assertEqual([d.doc_id for d in docs], ["rain_2001", "rain_2002", "beach"])
        self.assertEqual(missing, ["ghost_*"])
        docs, _ = ActionSet.load_tables_by_name(fake, ["rain_2001"], ["env", "wild"], 3)
        self.assertEqual([d.metadata["dataset_name"] for d in docs], ["env"])
        docs, _ = ActionSet.load_tables_by_name(fake, ["rain_*"], ["env", "wild"], 3)
        self.assertEqual(len(docs), 3)


if __name__ == "__main__":
    unittest.main()
