import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.actions.operators.semantic_column_generation import (
    SemanticColumnGeneration,
)
from pneuma_seeker.shared.config import Config


class SemanticColumnGenerationTests(unittest.TestCase):
    """Unit tests for SemanticColumnGeneration."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.semantic_col_gen = SemanticColumnGeneration(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_apply_invalid_inputs_raise(self):
        table = pd.DataFrame({"a": [1]})

        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "table": "not a df",
                    "column_name": "new_col",
                    "description": "desc",
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {"table": table, "column_name": 123, "description": "desc"}
            )

        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {"table": table, "column_name": "new_col", "description": 456}
            )

    def test_apply_empty_table_returns_unchanged(self):
        table = pd.DataFrame({"a": []})

        result = self.semantic_col_gen.apply(
            {
                "table": table,
                "column_name": "new_col",
                "description": "desc",
            }
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["a"])
        self.assertEqual(len(result), 0)

    def test_apply_happy_path_with_mocked_chat(self):
        table = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})

        def chat_side_effect(messages):
            content = "".join([m["content"] for m in messages])
            if "Values to transform" in content:
                return ["['A-30', 'B-25']"]
            return ["[]"]

        self.lm_api.chat = MagicMock(side_effect=chat_side_effect)

        result = self.semantic_col_gen.apply(
            {
                "table": table,
                "column_name": "profile",
                "description": "Combine name and age",
            }
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("profile", result.columns)
        self.assertEqual(result.iloc[0]["profile"], "A-30")
        self.assertEqual(result.iloc[1]["profile"], "B-25")

    def test_apply_deduplicates_formatted_values(self):
        table = pd.DataFrame({"name": ["Alice", "Alice"], "age": [30, 30]})

        self.lm_api.chat = MagicMock(return_value=["['A-30']"])

        result = self.semantic_col_gen.apply(
            {
                "table": table,
                "column_name": "profile",
                "description": "Combine name and age",
            }
        )

        self.assertEqual(list(result["profile"]), ["A-30", "A-30"])
        self.assertEqual(self.lm_api.chat.call_count, 1)
