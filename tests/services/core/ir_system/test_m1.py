# tests/services/core/ir_system/test_m1.py
import json
import os
import re
import sys
import threading
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

from pneuma_seeker.services.core.ir_system.m1 import M1, count_tokens, seeded_rng


class JudgeLLM:
    """Scripted classifier: a candidate is relevant if its text contains `marker`.
    Records each call's candidate ids (by their text) for inspection."""

    def __init__(self, marker: str = "relevant", fail: bool = False):
        self.marker = marker
        self.fail = fail
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def chat(self, messages, option=None):
        if self.fail:
            raise RuntimeError("model unavailable")
        candidates = re.findall(r"\[(c\d+)\] ([^\n]*)", messages[1]["content"].split("Candidates:", 1)[1])
        with self._lock:
            self.calls.append([text for _, text in candidates])
        yield json.dumps({label: self.marker in text for label, text in candidates})


def ranked(texts):
    return [{"id": f"e{i}", "content": t} for i, t in enumerate(texts)]


def select(llm, candidates, *, tau=20, window_batches=2, empty_windows=1, budget=10_000, need="need"):
    return M1(llm).select(
        need, iter(candidates), seeded_rng("chat", need),
        tau=tau, window_batches=window_batches, empty_windows=empty_windows,
        budget=budget, instructions="judge",
    )


class M1Tests(unittest.TestCase):
    def test_accepts_judged_relevant_in_rank_order(self):
        candidates = ranked(["alpha relevant", "beta", "gamma relevant", "delta"])
        result = select(JudgeLLM(), candidates)
        self.assertEqual([c["id"] for c in result.accepted], ["e0", "e2"])
        self.assertFalse(result.failed)

    def test_stops_after_an_empty_window(self):
        # Each candidate is a few tokens, so a window (2 batches of 20 tokens) holds several.
        candidates = ranked([f"item {i} relevant" for i in range(3)] + [f"filler {i}" for i in range(200)])
        llm = JudgeLLM()
        result = select(llm, candidates)
        self.assertEqual([c["id"] for c in result.accepted], ["e0", "e1", "e2"])
        # It read the first window (with the hits) and one empty window, then stopped.
        self.assertLess(result.read, 40)
        self.assertGreater(result.read, 3)

    def test_more_empty_windows_read_further(self):
        candidates = ranked([f"filler {i}" for i in range(30)] + ["late relevant"] + [f"x {i}" for i in range(30)])
        self.assertEqual(select(JudgeLLM(), candidates, empty_windows=1).accepted, [])
        late = select(JudgeLLM(), candidates, empty_windows=10)
        self.assertEqual([c["content"] for c in late.accepted], ["late relevant"])

    def test_token_budget_caps_what_is_judged(self):
        candidates = ranked([f"relevant {i}" for i in range(100)])
        result = select(JudgeLLM(), candidates, budget=30)
        judged_tokens = sum(count_tokens(c["content"]) for c in result.accepted)
        self.assertLessEqual(judged_tokens, 30)
        self.assertLess(len(result.accepted), 100)

    def test_long_candidate_is_its_own_batch(self):
        long_text = "relevant " + "word " * 200
        llm = JudgeLLM()
        result = select(llm, ranked([long_text, "short relevant"]), tau=20)
        self.assertEqual(len(result.accepted), 2)
        self.assertIn([long_text.strip()], [[t.strip() for t in call] for call in llm.calls])

    def test_failed_call_is_recorded(self):
        result = select(JudgeLLM(fail=True), ranked(["a relevant", "b"]))
        self.assertTrue(result.failed)
        self.assertEqual(result.accepted, [])

    def test_same_need_gets_the_same_batches(self):
        candidates = ranked([f"c{i} relevant" if i % 3 == 0 else f"c{i}" for i in range(20)])
        first, second = JudgeLLM(), JudgeLLM()
        select(first, candidates)
        select(second, candidates)
        self.assertEqual(sorted(map(tuple, first.calls)), sorted(map(tuple, second.calls)))

    def test_empty_ranking(self):
        result = select(JudgeLLM(), [])
        self.assertEqual((result.accepted, result.read, result.calls), ([], 0, 0))


if __name__ == "__main__":
    unittest.main()
