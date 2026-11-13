# tests/services/core/conductor/test_tribal_knowledge.py
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import unittest
from unittest.mock import MagicMock

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.main import (
    Conductor,
    split_information_needs,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames


class SplitInformationNeedsTests(unittest.TestCase):
    def test_numbered_list_with_continuations(self):
        needs = """1. What unit is rainfall in?
2) Which regions are in scope —
   Boston, Chatham, Amherst?

3. Which months count as summer?"""
        self.assertEqual(
            split_information_needs(needs),
            [
                "What unit is rainfall in?",
                "Which regions are in scope — Boston, Chatham, Amherst?",
                "Which months count as summer?",
            ],
        )

    def test_unnumbered_text_is_one_need(self):
        self.assertEqual(split_information_needs("  rainfall\n in Boston "), ["rainfall in Boston"])
        self.assertEqual(split_information_needs(""), [])


class ConductorTribalKnowledgeTests(unittest.TestCase):
    def setUp(self):
        config = Config(".env.test")
        config.LLM_PATH = "mock"
        config.EMBED_MODEL_PATH = "mock"
        logger = logging.getLogger("test_tribal_knowledge")
        logger.setLevel(logging.ERROR)
        tmp = tempfile.mkdtemp()
        self.conductor = Conductor(
            "u1", "c1", config, logger, ProvenanceGraph(logger),
            DBAPI(config, logger, dataset_db_path=str(Path(tmp, "d")), workspace_db_path=str(Path(tmp, "w"))),
            LanguageModelAPI(config, logger),
            frontend_callback=lambda _: None,
        )
        self.retrieve = MagicMock(return_value=[])
        self.conductor.tribal_knowledge_retriever.retrieve = self.retrieve  # type: ignore

    def _run(self, needs: str | None, **kwargs):
        def plan_needs(*_):
            self.conductor._information_needs = needs

        self.conductor._plan_information_needs = MagicMock(side_effect=plan_needs)  # type: ignore
        self.conductor.language_model_api.llm._responses = [  # type: ignore
            f'{{"plan": [{{"action":"{ActionNames.USER_FACING_COMMUNICATION.value}","args":{{}}}}]}}',
            "done",
        ]
        list(self.conductor.chat("How rainy was Boston?", [], [], dataset_name="env",
                                 memory_group_ids=["g1"], **kwargs))

    def _system_messages(self) -> list[str]:
        return [m["content"] for m in self.conductor.llm_messages if m["role"] == "system"]

    def test_searches_once_per_planned_need(self):
        self._run("1. What unit is rainfall in?\n2. Which months are summer?")
        self.retrieve.assert_called_once_with(
            "env", ["g1"], ["What unit is rainfall in?", "Which months are summer?"], chat_id="c1"
        )

    def test_falls_back_to_user_message_without_needs(self):
        self._run(None)
        self.assertEqual(self.retrieve.call_args.args[2], ["How rainy was Boston?"])

    def test_injects_matches_grouped_by_need(self):
        self.retrieve.return_value = [
            {"memory_id": "m1", "memory_type": "tribal_knowledge", "content": "Inches.", "query": "What unit?"},
            {"memory_id": "m2", "memory_type": "tribal_knowledge", "content": "Jun-Aug.", "query": "Which months?"},
        ]
        self._run("1. What unit?\n2. Which months?")
        (tk,) = [m for m in self._system_messages() if m.startswith("# Tribal Knowledge")]
        self.assertIn("Relevant to: What unit?\n- Inches.", tk)
        self.assertIn("Relevant to: Which months?\n- Jun-Aug.", tk)
        # The Materializer never sees this message, so the Conductor is told to
        # carry what matters into T's column descriptions or the note.
        self.assertIn("does not see this tribal knowledge", tk)
        self.assertIn("T's column descriptions or in the `note`", tk)
        self.assertEqual([e["memory_id"] for e in self.conductor.used_memory_entries], ["m1", "m2"])

    def test_no_message_when_nothing_matches(self):
        """Without matches the prompts are exactly as without tribal knowledge."""
        self._run("1. What unit?")
        self.assertEqual(
            self._system_messages(), [self.conductor.prompt_factory.get_sys_prompt()]
        )
        self.assertFalse(
            any("tribal knowledge" in str(m["content"]).lower() for m in self.conductor.llm_messages)
        )
        self.assertEqual(self.conductor.used_memory_entries, [])

    def test_retrieval_failure_does_not_break_the_turn(self):
        self.retrieve.side_effect = RuntimeError("postgres down")
        self._run("1. What unit?")
        self.assertEqual(self.conductor.user_facing_response, "done")
        self.assertEqual(self.conductor.used_memory_entries, [])


if __name__ == "__main__":
    unittest.main()
