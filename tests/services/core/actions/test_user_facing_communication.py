import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

from pneuma_seeker.services.core.action_set.impl.user_facing_communication import (
    UserFacingCommunication,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.role import Role


class UserFacingCommunicationTests(unittest.TestCase):
    """Unit tests for UserFacingCommunication's plan-mode/normal-mode prompt dispatch."""

    def setUp(self) -> None:
        self.user_id = "u1"
        self.chat_id = "c1"
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.captured_messages = []

        def fake_chat(messages, option=None):
            self.captured_messages.append(messages)
            yield "the response"

        self.lm_api.chat.side_effect = fake_chat

        self.action = UserFacingCommunication(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.lm_api,
        )

    def _system_prompt_from_last_call(self) -> str:
        messages = self.captured_messages[-1]
        system_messages = [m for m in messages if m["role"] == Role.SYSTEM.value]
        return system_messages[-1]["content"]

    def test_normal_mode_uses_normal_response_prompt(self):
        result = self.action.execute(
            {
                "planning_messages": [],
                "user_message": "how many orders?",
                "interaction_history": [],
                "forced": False,
                "plan_mode": False,
            }
        )
        self.assertEqual(result, "the response")
        prompt = self._system_prompt_from_last_call()
        self.assertIn("Respond to the user directly now", prompt)
        self.assertNotIn("Present your proposal to the user now", prompt)

    def test_plan_mode_uses_plan_mode_response_prompt(self):
        result = self.action.execute(
            {
                "planning_messages": [],
                "user_message": "how many orders?",
                "interaction_history": [],
                "forced": False,
                "plan_mode": True,
            }
        )
        self.assertEqual(result, "the response")
        prompt = self._system_prompt_from_last_call()
        self.assertNotIn("Respond to the user directly now", prompt)
        self.assertIn("Proposed data", prompt)
        self.assertIn("Proposed analysis", prompt)
        self.assertIn("Open questions", prompt)

    def test_plan_mode_prompt_offers_unanswerable_alternative(self):
        """The plan-mode prompt must not force every response into the 4-part
        proposal structure — it needs a real alternative for when nothing
        retrieved plausibly relates to the question at all."""
        self.action.execute(
            {
                "planning_messages": [],
                "user_message": "how many orders?",
                "interaction_history": [],
                "forced": False,
                "plan_mode": True,
            }
        )
        prompt = self._system_prompt_from_last_call()
        self.assertIn("does not support this request", prompt)
        self.assertIn("subject-matter mismatch", prompt)
        self.assertIn("not just an imperfect proxy", prompt)

    def test_plan_mode_defaults_to_false_when_omitted(self):
        self.action.execute(
            {
                "planning_messages": [],
                "user_message": "hi",
                "interaction_history": [],
            }
        )
        prompt = self._system_prompt_from_last_call()
        self.assertIn("Respond to the user directly now", prompt)

    def test_plan_mode_forced_uses_forced_clause(self):
        self.action.execute(
            {
                "planning_messages": [],
                "user_message": "hi",
                "interaction_history": [],
                "forced": True,
                "plan_mode": True,
            }
        )
        prompt = self._system_prompt_from_last_call()
        self.assertIn("step budget was exhausted", prompt)
        self.assertIn("best-effort proposal", prompt)


if __name__ == "__main__":
    unittest.main()
