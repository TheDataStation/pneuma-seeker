import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pneuma_seeker.session_manager import SessionManager
from pneuma_seeker.shared.config import Config


class SessionManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.logger = MagicMock()
        # Mock the ChatSession class imported in the session_manager module
        import pneuma_seeker.session_manager as sm_mod

        self._orig_chat_session = getattr(sm_mod, "ChatSession", None)
        sm_mod.ChatSession = MagicMock(side_effect=lambda *a, **k: MagicMock())

        self.pneuma_db = MagicMock()
        self.sm = SessionManager(self.cfg, self.logger, self.pneuma_db)

    def test_get_chat_session_returns_same_instance_for_same_keys(self):
        s1 = self.sm.get_chat_session("user1", "chat1")
        s2 = self.sm.get_chat_session("user1", "chat1")
        self.assertIs(s1, s2)

    def test_get_chat_session_creates_different_instances_for_different_keys(self):
        a = self.sm.get_chat_session("u1", "c1")
        b = self.sm.get_chat_session("u1", "c2")
        c = self.sm.get_chat_session("u2", "c1")
        self.assertIsNot(a, b)
        self.assertIsNot(a, c)
        self.assertIsNot(b, c)
    
    def test_chat_session_is_created_only_once_per_key(self):
        import pneuma_seeker.session_manager as sm_mod
        cs_mock = sm_mod.ChatSession

        self.sm.get_chat_session("user1", "chat1")
        self.sm.get_chat_session("user1", "chat1")

        self.assertEqual(cs_mock.call_count, 1) # type: ignore

    def tearDown(self) -> None:
        import pneuma_seeker.session_manager as sm_mod

        if self._orig_chat_session is not None:
            sm_mod.ChatSession = self._orig_chat_session


if __name__ == "__main__":
    unittest.main()
