import logging
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config

LLM_LATENCY_S = 0.5


class ContextExtractionTests(unittest.TestCase):
    """Runs context extraction against a real workspace DB with a stub LLM."""

    def setUp(self) -> None:
        self.user_id, self.chat_id = "u1", "c1"
        logger = logging.getLogger("test_context_extraction")
        logger.setLevel(logging.ERROR)
        self.config = Config()
        self.config.ENABLE_CONTEXT_EXTRACTION = True
        self.config.ENABLE_MEMORY_LAYER = False
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


if __name__ == "__main__":
    unittest.main()
