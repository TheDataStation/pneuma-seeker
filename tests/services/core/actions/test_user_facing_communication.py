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
from pneuma_seeker.shared.schemas.core.action import PLAN_INSTRUCTION
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


class PlanLeakGuardTests(unittest.TestCase):
    """Regression coverage for the plan-JSON-leak fix: planning turns must be
    summarized (not replayed) and any plan-shaped output must be caught."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.captured_messages = []
        self.responses: list[str] = []

        def fake_chat(messages, option=None):
            self.captured_messages.append(messages)
            response = self.responses.pop(0) if self.responses else "the response"
            yield response

        self.lm_api.chat.side_effect = fake_chat

        self.action = UserFacingCommunication(
            "u1", "c1", self.config, self.logger, self.db_api, self.lm_api
        )

    def _run(self, planning_messages=None, **overrides):
        payload = {
            "planning_messages": planning_messages or [],
            "user_message": "how many orders?",
            "interaction_history": [],
            "forced": False,
            "plan_mode": False,
        }
        payload.update(overrides)
        return self.action.execute(payload)

    # -- message shape sent to the LLM ------------------------------------

    def test_only_one_message_sent_no_raw_turn_replay(self):
        """planning_messages must never be replayed verbatim as chat turns —
        that's exactly what conditioned the model to keep emitting JSON."""
        planning_messages = [
            {"role": Role.SYSTEM.value, "content": "conductor sys prompt"},
            {
                "role": Role.USER.value,
                "content": f"Step 1 (out of maximum 10 steps)\n{PLAN_INSTRUCTION}",
            },
            {
                "role": Role.ASSISTANT.value,
                "content": '{"plan": [{"action": "situational_analysis", "args": {}}]}',
            },
            {"role": Role.USER.value, "content": "You did a situational analysis: hi"},
        ]
        self._run(planning_messages=planning_messages)
        sent = self.captured_messages[0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["role"], Role.SYSTEM.value)

    def test_planning_transcript_embedded_in_prompt(self):
        planning_messages = [
            {
                "role": Role.ASSISTANT.value,
                "content": '{"plan": [{"action": "situational_analysis", "args": {"message": "checking schema"}}]}',
            },
            {
                "role": Role.USER.value,
                "content": "You did a situational analysis: checking schema",
            },
        ]
        self._run(planning_messages=planning_messages)
        prompt = self.captured_messages[0][0]["content"]
        self.assertIn("Step 1 plan:", prompt)
        self.assertIn("checking schema", prompt)
        self.assertIn("Step 1 output:", prompt)

    def test_skeleton_marker_excluded_from_transcript(self):
        planning_messages = [
            {
                "role": Role.USER.value,
                "content": f"Step 1 (out of maximum 10 steps)\n{PLAN_INSTRUCTION}",
            },
            {"role": Role.ASSISTANT.value, "content": '{"plan": []}'},
        ]
        self._run(planning_messages=planning_messages)
        prompt = self.captured_messages[0][0]["content"]
        self.assertNotIn(PLAN_INSTRUCTION, prompt)

    def test_empty_planning_messages_uses_placeholder(self):
        self._run(planning_messages=[])
        prompt = self.captured_messages[0][0]["content"]
        self.assertIn("no planning steps were recorded", prompt)

    def test_multiple_steps_numbered_correctly(self):
        planning_messages = [
            {
                "role": Role.ASSISTANT.value,
                "content": '{"plan": [{"action": "situational_analysis"}]}',
            },
            {"role": Role.USER.value, "content": "result one"},
            {
                "role": Role.ASSISTANT.value,
                "content": '{"plan": [{"action": "table_retrieve"}]}',
            },
            {"role": Role.USER.value, "content": "result two"},
        ]
        self._run(planning_messages=planning_messages)
        prompt = self.captured_messages[0][0]["content"]
        self.assertIn("Step 1 plan:", prompt)
        self.assertIn("Step 1 output: result one", prompt)
        self.assertIn("Step 2 plan:", prompt)
        self.assertIn("Step 2 output: result two", prompt)

    # -- retry guard --------------------------------------------------------

    def test_retries_once_when_response_is_plan_json(self):
        leaked = (
            '{"plan": [{"action": "situational_analysis", "args": {"message": "x"}}]}'
        )
        self.responses = [leaked, "Here is your answer in plain language."]
        result = self._run()
        self.assertEqual(result, "Here is your answer in plain language.")
        self.assertEqual(self.lm_api.chat.call_count, 2)
        retry_messages = self.captured_messages[1]
        self.assertEqual(retry_messages[1]["role"], Role.ASSISTANT.value)
        self.assertEqual(retry_messages[1]["content"], leaked)
        self.assertEqual(retry_messages[2]["role"], Role.SYSTEM.value)

    def test_no_retry_when_response_is_clean(self):
        self.responses = ["A perfectly normal answer."]
        result = self._run()
        self.assertEqual(result, "A perfectly normal answer.")
        self.assertEqual(self.lm_api.chat.call_count, 1)

    def test_does_not_retry_more_than_once(self):
        """A second leaked response must not trigger a retry loop — it's
        returned as-is rather than spending a third LLM call."""
        leaked = '{"plan": [{"action": "situational_analysis"}]}'
        self.responses = [leaked, leaked]
        result = self._run()
        self.assertEqual(result, leaked)
        self.assertEqual(self.lm_api.chat.call_count, 2)

    def test_regression_malformed_leak_example(self):
        """Reproduces the originally reported leak: JSON-ish text missing its
        opening brace and using mismatched brackets."""
        leaked = (
            'plan": ["action":"situational_analysis","args": ("message": '
            '"All program_name values came back NULL in marketo_program_definitions."]],'
            '[action":"assumption_check", "args": ("uncertainties": ["table_ids":'
            '[marketo_program_definitions"]. "question": "What columns does this table have?"]]}'
        )
        self.responses = [
            leaked,
            "The program_name column is null for all rows in that table.",
        ]
        result = self._run()
        self.assertEqual(
            result, "The program_name column is null for all rows in that table."
        )
        self.assertEqual(self.lm_api.chat.call_count, 2)

    # -- _looks_like_plan_json unit coverage --------------------------------

    def test_looks_like_plan_json_valid_json_with_plan_key(self):
        self.assertTrue(
            self.action._looks_like_plan_json('{"plan": [{"action": "foo"}]}')
        )

    def test_looks_like_plan_json_valid_json_with_action_key_only(self):
        self.assertTrue(
            self.action._looks_like_plan_json('{"action": "foo", "args": {}}')
        )

    def test_looks_like_plan_json_prose_is_clean(self):
        self.assertFalse(
            self.action._looks_like_plan_json("Revenue grew 12% quarter over quarter.")
        )

    def test_looks_like_plan_json_prose_mentioning_word_plan_is_clean(self):
        self.assertFalse(
            self.action._looks_like_plan_json(
                "Here's the plan going forward: keep monitoring."
            )
        )

    def test_looks_like_plan_json_empty_string_is_clean(self):
        self.assertFalse(self.action._looks_like_plan_json(""))

    # -- single source of truth with prompt_factory --------------------------

    def test_prompt_factory_skeleton_is_filtered_from_transcript(self):
        """End-to-end guard: if prompt_factory.py ever reverts to a hardcoded
        string instead of PLAN_INSTRUCTION, its skeleton prompt would stop
        being recognized and filtered here — this test would catch that."""
        from pneuma_seeker.services.core.conductor.prompt_factory import (
            ConductorPromptFactory,
        )

        factory = ConductorPromptFactory(self.config, MagicMock())
        skeleton = factory.get_skeleton_curr_state_prompt(1)
        self.assertIn(PLAN_INSTRUCTION, skeleton)

        transcript = self.action._format_planning_transcript(
            [{"role": Role.USER.value, "content": skeleton}]
        )
        self.assertEqual(transcript, "")


if __name__ == "__main__":
    unittest.main()
