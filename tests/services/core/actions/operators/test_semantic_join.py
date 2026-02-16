import os
import sys
import unittest
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import numpy as np
import pandas as pd

from pneuma_seeker.services.core.actions.operators.semantic_join import (
    SemanticJoin,
    SyntacticSimMetric,
)
from pneuma_seeker.shared.config import Config


class SemanticJoinTests(unittest.TestCase):
    """Unit tests for SemanticJoin."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.semantic_join = SemanticJoin(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_apply_invalid_inputs_raise(self):
        left_df = pd.DataFrame({"name": ["a"]})
        right_df = pd.DataFrame({"name": ["b"]})

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_df": "not df",
                    "right_df": right_df,
                    "left_cols": ["name"],
                    "right_cols": ["name"],
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_df": left_df,
                    "right_df": "not df",
                    "left_cols": ["name"],
                    "right_cols": ["name"],
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_df": left_df,
                    "right_df": right_df,
                    "left_cols": "name",
                    "right_cols": ["name"],
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_df": left_df,
                    "right_df": right_df,
                    "left_cols": ["name"],
                    "right_cols": "name",
                }
            )

    def test_join_empty_dataframe_returns_empty(self):
        left_df = pd.DataFrame({"name": []})
        right_df = pd.DataFrame({"name": ["x"]})

        result = self.semantic_join.join(
            left_df,
            right_df,
            left_cols=["name"],
            right_cols=["name"],
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            list(result.columns),
            ["left_name", "right_name", "similarity_score"],
        )
        self.assertEqual(len(result), 0)

    def test_join_basic_no_llm_with_mock_embeddings(self):
        left_df = pd.DataFrame({"name": ["apple", "banana"]})
        right_df = pd.DataFrame({"name": ["apple", "orange"]})

        left_texts = ["name: apple", "name: banana"]
        right_texts = ["name: apple", "name: orange"]

        left_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        right_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def encode_side_effect(texts):
            if texts == left_texts:
                return left_emb
            if texts == right_texts:
                return right_emb
            raise AssertionError("Unexpected texts passed to encode")

        self.lm_api.encode = MagicMock(side_effect=encode_side_effect)

        result = self.semantic_join.join(
            left_df,
            right_df,
            left_cols=["name"],
            right_cols=["name"],
            top_k=1,
            syntactic_sim_metric=SyntacticSimMetric.NONE,
            use_llm=False,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["left_name"], "apple")
        self.assertEqual(result.iloc[0]["right_name"], "apple")
        self.assertEqual(result.iloc[1]["left_name"], "banana")
        self.assertEqual(result.iloc[1]["right_name"], "orange")

    def test_join_with_llm_filter(self):
        left_df = pd.DataFrame({"name": ["apple"]})
        right_df = pd.DataFrame({"name": ["apple", "orange"]})

        left_texts = ["name: apple"]
        right_texts = ["name: apple", "name: orange"]

        left_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        right_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def encode_side_effect(texts):
            if texts == left_texts:
                return left_emb
            if texts == right_texts:
                return right_emb
            raise AssertionError("Unexpected texts passed to encode")

        self.lm_api.encode = MagicMock(side_effect=encode_side_effect)
        self.lm_api.chat = MagicMock(return_value=["[1, 0]"])

        result = self.semantic_join.join(
            left_df,
            right_df,
            left_cols=["name"],
            right_cols=["name"],
            top_k=2,
            syntactic_sim_metric=SyntacticSimMetric.NONE,
            use_llm=True,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["left_name"], "apple")
        self.assertEqual(result.iloc[0]["right_name"], "apple")
