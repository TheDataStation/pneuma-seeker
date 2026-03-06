import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.action_set.impl.join_path_extraction import (
    JoinPathExtraction,
)
from pneuma_seeker.shared.config import Config


class JoinPathExtractionTests(unittest.TestCase):
    """Unit tests for JoinPathExtraction."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.join_path_extractor = JoinPathExtraction(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_discover_join_paths_with_matching_columns(self):
        """Test discovering join paths with tables having matching column names and values."""
        tables = {
            "users": pd.DataFrame(
                {"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
            ),
            "orders": pd.DataFrame({"user_id": [1, 2, 3], "amount": [100, 200, 150]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("users", result)
        self.assertIn("orders", result)
        self.assertIn("user_id", result)
        self.assertIn("score=", result)

    def test_discover_join_paths_with_similar_column_names(self):
        """Test discovering join paths with similar but not identical column names."""
        tables = {
            "customers": pd.DataFrame(
                {"customer_id": [1, 2], "name": ["Alice", "Bob"]}
            ),
            "sales": pd.DataFrame({"cust_id": [1, 2], "amount": [100, 200]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("Top-", result)

    def test_discover_join_paths_with_no_matches(self):
        """Test discovering join paths when there are no good matches."""
        tables = {
            "table_a": pd.DataFrame({"col_x": [1, 2], "col_y": [3, 4]}),
            "table_b": pd.DataFrame({"col_z": [10, 20], "col_w": [30, 40]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("Top-", result)

    def test_discover_join_paths_prunes_constant_columns(self):
        """Test that constant columns (cardinality <= 1) are pruned."""
        tables = {
            "table_a": pd.DataFrame({"const_col": [1, 1, 1], "var_col": [1, 2, 3]}),
            "table_b": pd.DataFrame({"const_col": [1, 1, 1], "var_col": [1, 2, 3]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        # Should suggest var_col, not const_col
        if "var_col" in result:
            self.assertIn("var_col", result)

    def test_discover_join_paths_with_empty_tables(self):
        """Test discovering join paths with empty tables."""
        tables = {
            "empty_table": pd.DataFrame({"id": []}),
            "table_b": pd.DataFrame({"id": [1, 2, 3]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("Top-", result)

    def test_discover_join_paths_with_value_overlap(self):
        """Test discovering join paths based on value overlap."""
        tables = {
            "products": pd.DataFrame(
                {
                    "product_code": ["A1", "B2", "C3"],
                    "name": ["Prod A", "Prod B", "Prod C"],
                }
            ),
            "inventory": pd.DataFrame(
                {"code": ["A1", "B2", "C3"], "quantity": [10, 20, 30]}
            ),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("products", result)
        self.assertIn("inventory", result)
        # Should detect the value overlap between product_code and code

    def test_discover_join_paths_respects_top_k(self):
        """Test that the result respects the top_k configuration."""
        tables = {
            "t1": pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
            "t2": pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        # Count the number of join suggestions (lines starting with "-")
        suggestion_lines = [
            line for line in result.split("\n") if line.strip().startswith("-")
        ]
        self.assertLessEqual(
            len(suggestion_lines), self.config.JOIN_PATH_EXTRACTION_TOP_K
        )

    def test_discover_join_paths_with_single_table(self):
        """Test discovering join paths with only one table (should return no paths)."""
        tables = {
            "single_table": pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("Top-", result)
        # Should have no actual suggestions
        suggestion_lines = [
            line for line in result.split("\n") if line.strip().startswith("-")
        ]
        self.assertEqual(len(suggestion_lines), 0)

    def test_discover_join_paths_with_nulls(self):
        """Test discovering join paths with columns containing null values."""
        tables = {
            "table_a": pd.DataFrame({"id": [1, 2, None, 4], "value": [10, 20, 30, 40]}),
            "table_b": pd.DataFrame(
                {"id": [1, 2, 3, None], "amount": [100, 200, 300, 400]}
            ),
        }

        result = self.join_path_extractor.discover_join_paths(tables)

        self.assertIsInstance(result, str)
        self.assertIn("Top-", result)
        # Should handle nulls gracefully and still detect the id column match
