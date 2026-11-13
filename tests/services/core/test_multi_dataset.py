# tests/services/core/test_multi_dataset.py
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
from pneuma_seeker.services.core.conductor.main import Conductor
from pneuma_seeker.services.core.dataset_context import (
    MULTI_DATASET_MODE,
    DatasetContext,
    in_scope,
    merge_retrieved,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table


def _table(doc_id: str, dataset: str | None) -> Table:
    return Table(
        doc_id=doc_id,
        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
        content=pd.DataFrame({"a": [1]}),
        metadata={"dataset_name": dataset} if dataset else {},
    )


class DatasetContextTests(unittest.TestCase):
    def test_single_mode_ignores_requested_datasets(self):
        ctx = DatasetContext.single("ds1")
        self.assertEqual(ctx.resolve(None), ["ds1"])
        self.assertEqual(ctx.resolve(["other", "ds2"]), ["ds1"])
        self.assertEqual(ctx.describe(), "")

    def test_multi_mode_defaults_and_validates(self):
        ctx = DatasetContext(default="a", accessible=("a", "b"), multi=True)
        self.assertEqual(ctx.resolve(None), ["a"])
        self.assertEqual(ctx.resolve(["b", "a", "b"]), ["b", "a"])
        self.assertEqual(ctx.resolve("b"), ["b"])
        with self.assertRaises(ValueError):
            ctx.resolve(["secret"])
        with self.assertRaises(ValueError):
            ctx.resolve([1])

    def test_multi_mode_without_default_requires_datasets(self):
        ctx = DatasetContext(default="", accessible=("a", "b"), multi=True)
        with self.assertRaises(ValueError):
            ctx.resolve(None)
        self.assertIn("a, b", ctx.describe())

    def test_merge_replaces_only_searched_datasets(self):
        existing = [_table("t1", "a"), _table("t2", "b")]
        merged, notes = merge_retrieved(existing, [_table("t3", "b")], ["b"])
        self.assertEqual(notes, [])
        self.assertEqual(
            [(t.metadata["dataset_name"], t.doc_id) for t in merged],
            [("a", "t1"), ("b", "t3")],
        )

    def test_merge_with_one_dataset_is_full_replacement(self):
        existing = [_table("t1", "a"), _table("t2", "a")]
        new = [_table("t3", "a"), _table("t3", "a")]
        self.assertEqual(
            [t.doc_id for t in merge_retrieved(existing, new, ["a"])[0]], ["t3"]
        )

    def test_merge_keeps_table_names_unique_newest_wins(self):
        merged, notes = merge_retrieved([_table("t", "a"), _table("u", "a")], [_table("t", "b")], ["b"])
        self.assertEqual(
            sorted((t.metadata["dataset_name"], t.doc_id) for t in merged),
            [("a", "u"), ("b", "t")],
        )
        self.assertEqual(len(notes), 1)
        self.assertIn("`t` from `a` was replaced by the one from `b`", notes[0])

    def test_merge_same_name_in_one_call_keeps_first(self):
        merged, notes = merge_retrieved([], [_table("t", "a"), _table("t", "b")], ["a", "b"])
        self.assertEqual([t.metadata["dataset_name"] for t in merged], ["a"])
        self.assertIn("exists in both `a` and `b`", notes[0])

    def test_merge_same_dataset_replacement_is_silent(self):
        merged, notes = merge_retrieved([_table("t", "a")], [_table("t", "a")], [])
        self.assertEqual((len(merged), notes), (1, []))

    def test_in_scope_drops_out_of_scope_tables(self):
        docs = [_table("t", "a"), _table("u", "b"), _table("ext", None)]
        self.assertEqual([t.doc_id for t in in_scope(docs, ["a"])], ["t", "ext"])

    def test_default_must_be_accessible(self):
        ctx = DatasetContext(default="unindexed", accessible=("a",), multi=True)
        with self.assertRaises(ValueError):
            ctx.resolve(None)


class MultiDatasetConductorTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(".env.test")
        self.config.LLM_PATH = "mock"
        self.config.EMBED_MODEL_PATH = "mock"
        self.config.ENABLE_MULTI_DATASET = True

        logger = logging.getLogger("test_multi_dataset")
        logger.setLevel(logging.ERROR)
        tmpdir = tempfile.mkdtemp()
        self.conductor = Conductor(
            user_id="uM",
            chat_id="cM",
            config=self.config,
            logger=logger,
            prov_graph=ProvenanceGraph(logger),
            db_api=DBAPI(
                self.config,
                logger,
                dataset_db_path=str(Path(tmpdir, "datasets")),
                workspace_db_path=str(Path(tmpdir, "workspaces")),
            ),
            language_model_api=LanguageModelAPI(self.config, logger),
            frontend_callback=lambda _: None,
        )
        self.conductor._plan_information_needs = MagicMock()  # type: ignore
        self.conductor.action_set.discover_join_paths = MagicMock(return_value="")  # type: ignore
        # Each (prompt list, dataset) returns one table named after the dataset.
        self.retrieve = MagicMock(
            side_effect=lambda prompts, _rt, ds, *a: [_table(f"{ds}_t", ds)]
        )
        self.conductor.action_set.retrieve_multi_topic_documents = self.retrieve  # type: ignore

    def _run(self, plans: list[str], **chat_kwargs):
        self.conductor.language_model_api.llm._responses = plans + [  # type: ignore
            f'{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}',
            "done",
        ]
        return list(
            self.conductor.chat(
                "q", interaction_history=[], external_table_paths=[], **chat_kwargs
            )
        )

    def _retrieve_plan(self, datasets: list[str] | None) -> str:
        ds_arg = f', "datasets": {datasets!r}'.replace("'", '"') if datasets else ""
        return (
            f'{{"plan": [{{"action":"{ActionNames.TABLE_RETRIEVE.value}",'
            f'"args":{{"prompts":["p"]{ds_arg}}}}}]}}'
        )

    def _outcomes(self) -> str:
        return "\n".join(str(m["content"]) for m in self.conductor.llm_messages)

    # --- off is unchanged ------------------------------------------------

    def test_prompts_unchanged_when_mode_not_requested(self):
        registry = self.conductor.action_set.registry
        factory = self.conductor.prompt_factory

        registry.set_active_modes(set())
        flag_on_prompt = factory.get_sys_prompt()
        self.config.ENABLE_MULTI_DATASET = False
        flag_off_prompt = factory.get_sys_prompt()

        self.assertEqual(flag_on_prompt, flag_off_prompt)
        self.assertNotIn(ActionNames.DATASET_LISTING.value, flag_on_prompt)
        self.assertNotIn('"datasets"', flag_on_prompt)

    def test_single_mode_ignores_datasets_arg_and_rejects_listing(self):
        self._run(
            [
                f'{{"plan": [{{"action":"{ActionNames.DATASET_LISTING.value}","args":{{}}}}]}}',
                self._retrieve_plan(["other"]),
            ],
            dataset_name="ds1",
        )
        self.assertFalse(self.conductor.dataset_ctx.multi)
        self.assertIn(
            f"Action specified is not valid: {ActionNames.DATASET_LISTING.value}",
            self._outcomes(),
        )
        self.retrieve.assert_called_once()
        self.assertEqual(self.retrieve.call_args.args[2], "ds1")

    def test_config_flag_gates_request_flag(self):
        self.config.ENABLE_MULTI_DATASET = False
        self._run([], dataset_name="ds1", multi_dataset=True, accessible_datasets=["ds1", "ds2"])
        self.assertFalse(self.conductor.dataset_ctx.multi)
        self.assertFalse(
            self.conductor.action_set.is_valid_conductor_action(
                ActionNames.DATASET_LISTING.value
            )
        )

    # --- multi mode ------------------------------------------------------

    def test_multi_mode_exposes_actions_and_datasets_arg(self):
        self._run([], dataset_name="", multi_dataset=True, accessible_datasets=["a", "b"])
        registry = self.conductor.action_set.registry
        names = [a.action_name for a in registry.get_for_agent(AgentType.CONDUCTOR)]
        self.assertEqual(names[0], ActionNames.DATASET_LISTING)
        self.assertIn(
            ActionNames.DATASET_LISTING,
            [a.action_name for a in registry.get_for_agent(AgentType.MATERIALIZER)],
        )
        self.assertIn('"datasets"', self.conductor.prompt_factory.get_sys_prompt())

    def test_retrieve_across_datasets_keeps_other_datasets_tables(self):
        self._run(
            [self._retrieve_plan(["a", "b"]), self._retrieve_plan(["b"])],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        self.assertEqual(
            [c.args[2] for c in self.retrieve.call_args_list], ["a", "b", "b"]
        )
        self.assertEqual(
            sorted(t.doc_id for t in self.conductor.retrieved_tables), ["a_t", "b_t"]
        )

    def test_retrieve_rejects_inaccessible_dataset(self):
        self._run(
            [self._retrieve_plan(["secret"])],
            dataset_name="a",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        self.retrieve.assert_not_called()
        self.assertIn("not accessible", self._outcomes())

    def test_retrieve_without_default_requires_datasets(self):
        self._run(
            [self._retrieve_plan(None)],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        self.retrieve.assert_not_called()
        self.assertIn("No dataset to search", self._outcomes())

    def test_list_datasets_lists_datasets_without_searching(self):
        db_api = self.conductor.action_set.registry.get(ActionNames.DATASET_LISTING).db_api
        db_api.list_dataset_tables = MagicMock(  # type: ignore
            side_effect=lambda ds: [f"{ds}_t{i}" for i in range(7)]
        )
        self._run(
            [f'{{"plan": [{{"action":"{ActionNames.DATASET_LISTING.value}","args":{{}}}}]}}'],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        outcomes = self._outcomes()
        self.assertIn("- a (7 tables; e.g. a_t0, a_t1, a_t2, a_t3, a_t4, ...)", outcomes)
        self.assertIn("- b (7 tables;", outcomes)
        self.retrieve.assert_not_called()  # purely dataset-level: no table search

    def test_list_datasets_uses_descriptions(self):
        db_api = self.conductor.action_set.registry.get(ActionNames.DATASET_LISTING).db_api
        db_api.list_dataset_tables = MagicMock(side_effect=lambda ds: [f"{ds}_t"])  # type: ignore
        db_api.get_dataset_description = MagicMock(  # type: ignore
            side_effect=lambda ds: "Boston rainfall\nand beaches" if ds == "a" else ""
        )
        self._run(
            [f'{{"plan": [{{"action":"{ActionNames.DATASET_LISTING.value}","args":{{}}}}]}}'],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        outcomes = self._outcomes()
        self.assertIn("- a (1 table): Boston rainfall and beaches", outcomes)
        self.assertIn("- b (1 table; e.g. b_t)", outcomes)  # no description: examples

    def _make_dataset(self, root: Path, name: str) -> None:
        import duckdb

        (root / name).mkdir(parents=True)
        con = duckdb.connect(str(root / name / f"{name}.db"))
        con.execute("CREATE TABLE t AS SELECT 1 AS x")
        con.close()

    def test_turn_detaches_datasets_outside_its_scope(self):
        """A multi-dataset turn attaches several datasets on the cached workspace
        connection; a later single-dataset turn must not be able to query them."""
        db_api = self.conductor.db_api
        root = Path(db_api.pneuma_db.dataset_manager.dataset_db_path)
        for name in ("a", "b"):
            self._make_dataset(root, name)
            db_api.link_dataset_tables("uM", "cM", name)

        self._run([], dataset_name="a")

        attached = db_api.execute_query("uM", "cM", "PRAGMA database_list")["name"].tolist()
        self.assertIn("a", attached)
        self.assertNotIn("b", attached)
        self.assertEqual(db_api.execute_query("uM", "cM", 'SELECT x FROM a."t"')["x"].tolist(), [1])
        with self.assertRaises(Exception):
            db_api.execute_query("uM", "cM", 'SELECT x FROM b."t"')

    def test_multi_turn_keeps_accessible_datasets_attached(self):
        db_api = self.conductor.db_api
        root = Path(db_api.pneuma_db.dataset_manager.dataset_db_path)
        for name in ("a", "b"):
            self._make_dataset(root, name)
            db_api.link_dataset_tables("uM", "cM", name)

        self._run([], dataset_name="", multi_dataset=True, accessible_datasets=["a", "b"])

        attached = db_api.execute_query("uM", "cM", "PRAGMA database_list")["name"].tolist()
        self.assertTrue({"a", "b"} <= set(attached))

    def test_turn_drops_tables_from_datasets_outside_scope(self):
        """Tables carried over from a multi-dataset turn (or restored after access
        was revoked) must not reach the next turn's agents."""
        self._run(
            [self._retrieve_plan(["a", "b"])],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        self.assertEqual(len(self.conductor.retrieved_tables), 2)

        self._run([], dataset_name="a")  # single-dataset turn
        self.assertEqual([t.doc_id for t in self.conductor.retrieved_tables], ["a_t"])

        self._run([], dataset_name="", multi_dataset=True, accessible_datasets=["b"])
        self.assertEqual(self.conductor.retrieved_tables, [])

    def test_same_named_tables_across_datasets_stay_unique(self):
        self.retrieve.side_effect = lambda prompts, _rt, ds, *a: [_table("orders", ds)]
        self._run(
            [self._retrieve_plan(["a"]), self._retrieve_plan(["b"])],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        self.assertEqual(
            [(t.metadata["dataset_name"], t.doc_id) for t in self.conductor.retrieved_tables],
            [("b", "orders")],
        )
        self.assertIn("`orders` from `a` was replaced by the one from `b`", self._outcomes())

    def test_tribal_knowledge_follows_newly_searched_datasets(self):
        """The first retrieval from a dataset adds its tribal knowledge (matched
        to this turn's needs), without repeating entries already shown."""
        shared = {"memory_id": "m0", "memory_type": "tribal_knowledge", "content": "Untagged rule.", "query": "q"}
        by_ds = {
            "": [shared],
            "b": [shared, {"memory_id": "m1", "memory_type": "tribal_knowledge", "content": "B rule.", "query": "q"}],
        }
        tk = MagicMock(side_effect=lambda ds, *_, **__: by_ds.get(ds, []))
        self.conductor.tribal_knowledge_retriever.retrieve = tk  # type: ignore
        self._run(
            [self._retrieve_plan(["b"]), self._retrieve_plan(["b"])],
            dataset_name="",
            multi_dataset=True,
            accessible_datasets=["a", "b"],
        )
        self.assertEqual([c.args[0] for c in tk.call_args_list], ["", "b"])  # b fetched once
        self.assertEqual(tk.call_args.args[2], ["q"])  # same queries as the turn ("q" = user message)
        self.assertEqual([e["memory_id"] for e in self.conductor.used_memory_entries], ["m0", "m1"])
        self.assertEqual(self._outcomes().count("B rule."), 1)

    def test_mode_does_not_leak_into_next_turn(self):
        self._run([], dataset_name="a", multi_dataset=True, accessible_datasets=["a", "b"])
        self._run([], dataset_name="a")
        self.assertFalse(self.conductor.dataset_ctx.multi)
        self.assertNotIn(
            MULTI_DATASET_MODE,
            self.conductor.action_set.registry.get(ActionNames.TABLE_RETRIEVE).active_modes,
        )


if __name__ == "__main__":
    unittest.main()
