import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import duckdb
import pandas as pd

from pneuma_seeker.services.core.action_set.impl.join_path_extraction import (
    JoinPathExtraction,
)
from pneuma_seeker.services.core.ir_system.join_profiles import (
    SKETCH_SIZE,
    JoinProfiles,
    containment,
    profile_relation,
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

    def test_prefixed_ids_join_on_their_trailing_digits(self):
        books = pd.DataFrame({
            "book_id": [f"bookid_{i}" for i in range(1, 41)],
            "rating_number": [i % 5 + 1 for i in range(40)],
        })
        reviews = pd.DataFrame({
            "purchase_id": [f"purchaseid_{i % 40 + 1}" for i in range(120)],
            "rating": [i % 5 + 1 for i in range(120)],
        })

        result = self.join_path_extractor.discover_join_paths({"books_info": books, "review": reviews})

        first = result.split("\n")[1]
        self.assertIn("books_info (columns: [book_id]) <-> review (columns: [purchase_id])", first)
        self.assertIn("on their trailing digits", first)
        self.assertIn("regexp_extract", first)
        self.assertIn("100% of books_info rows and 100% of review rows have a match; 1:N", first)
        # Small numbers that overlap with no unique side rank below the key.
        self.assertIn("N:M", result.split("\n")[2])

    def test_float_ids_with_nulls_match_prefixed_ids(self):
        result = self.join_path_extractor.discover_join_paths({
            "a": pd.DataFrame({"id": [1, 2, None, 4]}),
            "b": pd.DataFrame({"ref": ["item-1", "item-2", "item-4", "item-4"]}),
        })

        self.assertIn('TRY_CAST(a."id" AS BIGINT) = TRY_CAST(regexp_extract(', result)
        self.assertIn("100% of a rows and 100% of b rows have a match; 1:N", result)

    def test_without_normalize_and_verify_values_are_compared_as_is(self):
        self.config.JOIN_PATH_EXTRACTION_NORMALIZE = False
        self.config.JOIN_PATH_EXTRACTION_VERIFY = False

        result = self.join_path_extractor.discover_join_paths({
            "books_info": pd.DataFrame({"book_id": ["bookid_1", "bookid_2"]}),
            "review": pd.DataFrame({"purchase_id": ["purchaseid_1", "purchaseid_2"]}),
        })

        self.assertNotIn("match", result.split("\n", 1)[1])


class JoinProfileTests(unittest.TestCase):
    """Join profiles built from tables in DuckDB, and join paths scored from them."""

    def setUp(self) -> None:
        self.con = duckdb.connect()
        self.con.execute("""
            CREATE TABLE books_info AS
            SELECT 'bookid_' || range AS book_id, range % 5 + 1 AS rating_number,
                   repeat('long description ', 10) AS description, range * 1.25 AS price,
                   CAST(range AS DOUBLE) AS whole FROM range(1, 401);
            CREATE TABLE review AS
            SELECT 'purchaseid_' || (range % 400 + 1) AS purchase_id, range % 5 + 1 AS rating FROM range(1200);
            -- More distinct ids than a sketch holds, half of them shared, formatted differently.
            CREATE TABLE name AS SELECT 'nm' || lpad(CAST(range AS VARCHAR), 5, '0') AS id FROM range(1, 5001);
            CREATE TABLE cast_info AS SELECT 'NM###' || (range % 5000 + 2501) AS person_id FROM range(20000);
        """)
        self.run = lambda sql: self.con.execute(sql).df()
        self.done: list[str] = []
        self.profiles = {t: profile_relation(self.run, t, on_column=self.done.append)
                         for t in ("books_info", "review", "name", "cast_info")}

    def test_columns_get_their_kind_and_counts(self):
        books = self.profiles["books_info"]
        self.assertEqual(books["rows"], 400)
        self.assertEqual(books["columns"]["book_id"]["kind"], "prefixed")
        self.assertEqual(books["columns"]["rating_number"]["kind"], "integer")
        self.assertEqual(books["columns"]["description"]["kind"], "other")  # long text is not a key
        self.assertEqual(books["columns"]["price"], {"kind": "other", "keys": {}})  # nor are decimals
        self.assertEqual(books["columns"]["whole"]["kind"], "integer")  # whole numbers stored as decimals are
        self.assertFalse(self.con.execute(
            "SELECT * FROM duckdb_tables() WHERE table_name LIKE '_join_profile_%'"
        ).fetchall())  # the copy is dropped
        digits = books["columns"]["book_id"]["keys"]["digits"]
        self.assertEqual((digits["nonnull"], digits["distinct"], len(digits["hashes"])), (400, 400, 400))

    def test_every_column_is_reported_done_once(self):
        self.assertEqual(sorted(self.done), sorted(
            ["book_id", "rating_number", "description", "price", "whole", "purchase_id", "rating", "id", "person_id"]
        ))

    def test_containment_is_exact_for_small_columns_and_estimated_for_large_ones(self):
        book = self.profiles["books_info"]["columns"]["book_id"]["keys"]["digits"]
        purchase = self.profiles["review"]["columns"]["purchase_id"]["keys"]["digits"]
        self.assertEqual(containment(purchase, book), (1.0, True))

        name = self.profiles["name"]["columns"]["id"]["keys"]["digits"]
        person = self.profiles["cast_info"]["columns"]["person_id"]["keys"]["digits"]
        self.assertEqual(len(name["hashes"]), SKETCH_SIZE)
        share, exact = containment(name, person)
        self.assertFalse(exact)
        self.assertAlmostEqual(share, 0.5, delta=0.1)  # names 2501..5000 of 1..5000 are cast

    def test_join_paths_come_from_stored_profiles_when_every_table_has_one(self):
        extractor = JoinPathExtraction("u", "c", Config(), MagicMock(), MagicMock(), MagicMock())
        three_rows = {
            "books_info": self.con.execute("SELECT * FROM books_info LIMIT 3").df(),
            "review": self.con.execute("SELECT * FROM review WHERE rating = 1 LIMIT 3").df(),
        }
        stored = {"books_info": self.profiles["books_info"], "review": self.profiles["review"]}
        with patch.object(JoinProfiles, "load", return_value=stored):
            result = extractor.discover_join_paths(three_rows, {"books_info": "books", "review": "books"})

        first = result.split("\n")[1]
        self.assertIn("books_info (columns: [book_id]) <-> review (columns: [purchase_id])", first)
        self.assertIn("on their trailing digits", first)
        self.assertIn("100% of books_info.book_id distinct values occur in review.purchase_id, 100% the other way; 1:N", first)

        with patch.object(JoinProfiles, "load", return_value=None):  # missing or stale
            fallback = extractor.discover_join_paths(three_rows, {"books_info": "books", "review": "books"})
        self.assertNotIn("distinct values occur", fallback)
