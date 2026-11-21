import json
import logging
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table

LLM_LATENCY_S = 0.5


class ContextExtractionTests(unittest.TestCase):
    """Runs context extraction against a real workspace DB with a stub LLM."""

    def setUp(self) -> None:
        self.user_id, self.chat_id = "u1", "c1"
        logger = logging.getLogger("test_context_extraction")
        logger.setLevel(logging.ERROR)
        self.config = Config()
        self.config.ENABLE_CONTEXT_EXTRACTION = True
        tmpdir = tempfile.mkdtemp()
        self.db_api = DBAPI(
            self.config,
            logger,
            str(Path(tmpdir) / "datasets"),
            str(Path(tmpdir) / "workspaces"),
        )
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            "CREATE TABLE numbers AS SELECT range AS n FROM range(10);",
        )
        self.initial_tables = self._workspace_tables()

        lm_api = MagicMock()
        lm_api.chat.side_effect = self._fake_chat
        self.action_set = ActionSet(
            self.user_id,
            self.chat_id,
            self.config,
            logger,
            ProvenanceGraph(logger),
            self.db_api,
            lm_api,
        )

    @staticmethod
    def _fake_chat(messages, option=None):
        """Answers "How many numbers are below K?" with code writing the count
        into the result table named in the prompt, after a fixed latency."""
        prompt = messages[-1]["content"]
        limit = re.search(r"below (\d+)", prompt).group(1)  # type: ignore[union-attr]
        table = re.search(r'into a table named "([^"]+)"', prompt).group(1)  # type: ignore[union-attr]
        time.sleep(LLM_LATENCY_S)
        yield (
            "db_api.execute_query(user_id, chat_id, "
            f"'CREATE OR REPLACE TABLE \"{table}\" AS "
            f"SELECT COUNT(*) AS below FROM numbers WHERE n < {limit}')"
        )

    def _workspace_tables(self) -> set[str]:
        return set(
            self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")["name"]
        )

    def test_questions_are_resolved_concurrently_in_order(self):
        limits = [3, 7, 5, 9]
        start = time.monotonic()
        summary, logs = self.action_set.run_context_extraction(
            [
                {"table_ids": ["numbers"], "question": f"How many numbers are below {k}?"}
                for k in limits
            ],
            [],
            "check",
        )
        elapsed = time.monotonic() - start

        # Each answer matches its own question, in the original order.
        for i, k in enumerate(limits, start=1):
            self.assertRegex(
                summary, rf"Q{i}: How many numbers are below {k}\?\nA{i}: \s*below\s+{k}\b"
            )
        self.assertEqual(
            [m for m in logs if m.startswith("[Context Extraction ")],
            [
                f"[Context Extraction {i}/4] Resolving: How many numbers are below {k}?"
                for i, k in enumerate(limits, start=1)
            ],
        )
        # Sequential resolution would take len(limits) * LLM_LATENCY_S.
        self.assertLess(elapsed, (len(limits) - 1) * LLM_LATENCY_S)
        # Per-question result tables are cleaned up.
        self.assertEqual(self._workspace_tables(), self.initial_tables)

    def test_questions_without_text_are_skipped(self):
        summary, _ = self.action_set.run_context_extraction(
            [
                {"table_ids": ["numbers"], "question": ""},
                {"table_ids": ["numbers"], "question": "How many numbers are below 4?"},
            ],
            [],
            "check",
        )
        self.assertRegex(summary, r"Q1: How many numbers are below 4\?\nA1: \s*below\s+4\b")
        self.assertNotIn("Q2", summary)
        self.assertEqual(self._workspace_tables(), self.initial_tables)


class ContextExtractionReuseTests(unittest.TestCase):
    """Reuse of a chat's earlier answers, with a stub LLM for both the reuse check and the code."""

    def setUp(self) -> None:
        self.user_id, self.chat_id = "u1", "c1"
        logger = logging.getLogger("test_context_extraction")
        logger.setLevel(logging.ERROR)
        self.config = Config()
        self.config.ENABLE_CONTEXT_EXTRACTION = True
        self.config.CONTEXT_EXTRACTION_REUSE_TOP_K = 3
        tmpdir = tempfile.mkdtemp()
        self.db_api = DBAPI(
            self.config, logger, str(Path(tmpdir) / "datasets"), str(Path(tmpdir) / "workspaces")
        )
        self.db_api.execute_query(
            self.user_id, self.chat_id, "CREATE TABLE numbers AS SELECT range AS n FROM range(10);"
        )
        self.numbers = Table("numbers", RetrieverType.PNEUMA_RETRIEVER, pd.DataFrame({"n": range(10)}), {})
        self.reuse_prompts: list[str] = []
        self.reuse_response = json.dumps({"decisions": []})
        self.code_calls = 0
        self.code_prompts: list[str] = []
        lm_api = MagicMock()
        lm_api.chat.side_effect = self._fake_chat
        self.action_set = ActionSet(
            self.user_id, self.chat_id, self.config, logger, ProvenanceGraph(logger), self.db_api, lm_api
        )

    def _fake_chat(self, messages, option=None):
        if messages[0]["content"].startswith("You decide"):
            self.reuse_prompts.append(messages[-1]["content"])
            yield self.reuse_response
            return
        self.code_calls += 1
        prompt = messages[-1]["content"]
        self.code_prompts.append(prompt)
        limit = re.search(r"below (\d+)", prompt).group(1)  # type: ignore[union-attr]
        table = re.search(r'into a table named "([^"]+)"', prompt).group(1)  # type: ignore[union-attr]
        yield (
            "db_api.execute_query(user_id, chat_id, "
            f"'CREATE OR REPLACE TABLE \"{table}\" AS "
            f"SELECT COUNT(*) AS below FROM numbers WHERE n < {limit}')"
        )

    def ask(self, *questions: str, tables=None, asked_by: str = "check") -> str:
        summary, _ = self.action_set.run_context_extraction(
            [{"table_ids": ["numbers"], "question": q} for q in questions],
            tables if tables is not None else [self.numbers],
            f"{asked_by}_assumption_check",
        )
        return summary

    def test_an_answer_the_reuse_check_picks_is_returned_without_new_code(self):
        self.ask("How many numbers are below 3?")
        self.reuse_response = json.dumps({"decisions": [
            {"question": 1, "reuse": 1}, {"question": 2, "reuse": None},
        ]})

        summary = self.ask("How many numbers are below 3 in this table?", "How many numbers are below 8?")

        self.assertEqual(self.code_calls, 2)  # the first question, then only the second one
        self.assertEqual(len(self.reuse_prompts), 1)  # one check for both questions
        self.assertIn("Earlier answer 1\nQuestion: How many numbers are below 3?", self.reuse_prompts[0])
        self.assertRegex(
            summary,
            r"A1 \(reused, answered earlier for: How many numbers are below 3\?\): \s*below\s+3\b",
        )
        self.assertRegex(summary, r"Q2: How many numbers are below 8\?\nA2: \s*below\s+8\b")

    def test_a_rebuilt_table_makes_earlier_answers_about_it_ineligible(self):
        built = Table("numbers", RetrieverType.MATERIALIZER, pd.DataFrame({"n": range(10)}), {}, last_node_id="n1")
        self.ask("How many numbers are below 3?", tables=[built])
        built.last_node_id = "n2"

        self.ask("How many numbers are below 3?", tables=[built])

        self.assertEqual(self.reuse_prompts, [])
        self.assertEqual(self.code_calls, 2)

    def test_a_source_table_loaded_into_provenance_keeps_its_answers(self):
        self.ask("How many numbers are below 3?")
        self.numbers.last_node_id = "load-node"

        self.ask("How many numbers are below 3?")

        self.assertEqual(len(self.reuse_prompts), 1)

    def test_answers_about_other_tables_are_not_candidates(self):
        other = Table("letters", RetrieverType.PNEUMA_RETRIEVER, pd.DataFrame({"c": ["a", "b"]}), {})
        self.action_set.run_context_extraction(
            [{"table_ids": ["letters"], "question": "How many numbers are below 3?"}], [other], "check"
        )

        self.ask("How many numbers are below 3?")

        self.assertEqual(self.reuse_prompts, [])

    def test_a_failed_or_invalid_reuse_check_resolves_every_question(self):
        self.ask("How many numbers are below 3?")
        for response in ("not json", json.dumps({"decisions": [{"question": 1, "reuse": 7}]})):
            self.reuse_response = response
            summary = self.ask("How many numbers are below 3?")
            self.assertRegex(summary, r"Q1: How many numbers are below 3\?\nA1: \s*below\s+3\b")
        self.assertEqual(self.code_calls, 3)

    def test_a_partly_covered_question_starts_from_the_earlier_code(self):
        self.ask("How many numbers are below 3?")
        self.reuse_response = json.dumps({"decisions": [{"question": 1, "reuse": None, "start_from": 1}]})

        summary = self.ask("How many numbers are below 5, and how many below 3?")

        self.assertEqual(self.code_calls, 2)
        self.assertIn("## Earlier Check in This Chat", self.code_prompts[-1])
        self.assertIn("WHERE n < 3", self.code_prompts[-1])  # the earlier code
        self.assertNotIn("reused", summary)
        self.assertNotIn("## Earlier Check in This Chat", self.code_prompts[0])

    def test_established_facts_are_the_conductors_checks_on_unchanged_tables(self):
        built = Table("built", RetrieverType.MATERIALIZER, pd.DataFrame({"n": range(10)}), {}, last_node_id="n1")
        self.ask("How many numbers are below 3?", asked_by="conductor")
        self.ask("How many numbers are below 4?", asked_by="materializer")
        self.ask("How many numbers are below 6?", tables=[built], asked_by="conductor")

        facts = self.action_set.established_facts("conductor", [self.numbers, built])
        self.assertIn("Q: How many numbers are below 3?", facts)
        self.assertRegex(facts, r"A: below\s+3")  # padding collapsed
        self.assertNotIn("below 4", facts)  # the Materializer's own check
        # A table that was rebuilt since: its facts no longer hold.
        built.last_node_id = "n2"
        self.assertNotIn("below 6", self.action_set.established_facts("conductor", [self.numbers, built]))
        self.assertEqual(self.action_set.established_facts("conductor", []), "")

    def test_reuse_is_off_with_top_k_zero(self):
        self.config.CONTEXT_EXTRACTION_REUSE_TOP_K = 0
        self.ask("How many numbers are below 3?")
        self.ask("How many numbers are below 3?")

        self.assertEqual(self.reuse_prompts, [])
        self.assertEqual(self.code_calls, 2)


if __name__ == "__main__":
    unittest.main()
