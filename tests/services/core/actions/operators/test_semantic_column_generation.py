import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.action_set.impl.semantic_column_generation import (
    SemanticColumnGeneration,
)
from pneuma_seeker.services.core.action_set.impl.table_projection import (
    TableProjection,
)
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.language_model.impl.mock_llm import MockLLM
from pneuma_seeker.shared.config import Config


class SemanticColumnGenerationTests(unittest.TestCase):
    """Unit tests for SemanticColumnGeneration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_db_path = Path(self.temp_dir) / "datasets"
        self.workspace_db_path = Path(self.temp_dir) / "workspaces"
        self.config = Config()
        self.config.DATA_SOURCES = ["test_ds"]
        self.config.ENABLE_SEMANTIC_COL_GEN = True
        self.logger = MagicMock()
        self.db_api = DBAPI(
            self.config,
            self.logger,
            str(self.dataset_db_path),
            str(self.workspace_db_path),
        )
        self.lm_api = MagicMock()
        self.semantic_col_gen = SemanticColumnGeneration(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )
        self.table_projection = TableProjection(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_apply_invalid_inputs_raise(self):
        table = pd.DataFrame({"a": [1]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"a": "a"},
            }
        )

        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "not a table id",
                    "src_table_columns": ["a"],
                    "new_column_name": "new_col",
                    "description": "desc",
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "table",
                    "src_table_columns": ["a"],
                    "new_column_name": 123,
                    "description": "desc",
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "table",
                    "src_table_columns": ["a"],
                    "new_column_name": "new_col",
                    "description": 456,
                }
            )

    def test_apply_empty_table_returns_unchanged(self):
        table = pd.DataFrame({"a": []})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"a": "a"},
            }
        )

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "src_table_columns": ["a"],
                "new_column_name": "new_col",
                "instruction": "desc",
            }
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["a", "new_col"])
        self.assertEqual(len(result), 0)

    def test_apply_deduplicates_formatted_values(self):
        table = pd.DataFrame({"name": ["Alice", "Alice"], "age": [30, 30]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"name": "name", "age": "age"},
            }
        )

        self.lm_api.chat = MagicMock(return_value=["['A-30']"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "profile",
                "src_table_columns": ["name", "age"],
                "instruction": "Combine name and age",
            }
        )

        self.assertEqual(list(result["profile"]), ["A-30", "A-30"])
        self.assertEqual(self.lm_api.chat.call_count, 1)

    def test_apply_missing_src_table_id_raises(self):
        """Test that missing src_table_id raises ValueError."""
        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_columns": ["a"],
                    "new_column_name": "new_col",
                    "instruction": "desc",
                }
            )

    def test_apply_missing_src_table_columns_raises(self):
        """Test that missing src_table_columns raises ValueError."""
        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "table",
                    "new_column_name": "new_col",
                    "instruction": "desc",
                }
            )

    def test_apply_empty_src_table_columns_raises(self):
        """Test that empty src_table_columns list raises ValueError."""
        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "table",
                    "src_table_columns": [],
                    "new_column_name": "new_col",
                    "instruction": "desc",
                }
            )

    def test_apply_non_string_columns_in_list_raises(self):
        """Test that non-string elements in src_table_columns raises ValueError."""
        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "table",
                    "src_table_columns": ["a", 123],
                    "new_column_name": "new_col",
                    "instruction": "desc",
                }
            )

    def test_apply_src_table_columns_not_list_raises(self):
        """Test that non-list src_table_columns raises ValueError."""
        with self.assertRaises(ValueError):
            self.semantic_col_gen.apply(
                {
                    "src_table_id": "table",
                    "src_table_columns": "a",
                    "new_column_name": "new_col",
                    "instruction": "desc",
                }
            )

    def test_apply_single_row_table(self):
        """Test semantic column generation on a single-row table."""
        table = pd.DataFrame({"city": ["Paris"], "country": ["France"]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"city": "city", "country": "country"},
            }
        )

        self.lm_api.chat = MagicMock(return_value=["['Paris, France']"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "location",
                "src_table_columns": ["city", "country"],
                "instruction": "Combine city and country with comma",
            }
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result["location"].iloc[0], "Paris, France")

    def test_apply_multiple_unique_values(self):
        """Test semantic column generation with multiple unique row combinations."""
        table = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"name": "name", "age": "age"},
            }
        )

        self.lm_api.chat = MagicMock(
            return_value=["['Adult-Alice', 'Adult-Bob', 'Adult-Charlie']"]
        )

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "profile",
                "src_table_columns": ["name", "age"],
                "instruction": "Prepend 'Adult-' to names",
            }
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(self.lm_api.chat.call_count, 1)

    def test_apply_special_characters_in_column_names(self):
        """Test handling of column names with special characters."""
        table = pd.DataFrame({"first-name": ["Alice"], "last-name": ["Smith"]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {
                    "first_name": "first_name",
                    "last_name": "last_name",
                },
            }
        )

        self.lm_api.chat = MagicMock(return_value=["['Alice Smith']"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "full_name",
                "src_table_columns": ["first_name", "last_name"],
                "instruction": "Combine first and last names",
            }
        )

        self.assertEqual(result["full_name"].iloc[0], "Alice Smith")

    def test_apply_numeric_and_string_values(self):
        """Test formatting with mixed numeric and string data types."""
        table = pd.DataFrame({"product": ["Widget"], "quantity": [5], "price": [19.99]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {
                    "product": "product",
                    "quantity": "quantity",
                    "price": "price",
                },
            }
        )

        self.lm_api.chat = MagicMock(return_value=["['Widget x5 @ $19.99']"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "order_summary",
                "src_table_columns": ["product", "quantity", "price"],
                "instruction": "Create order summary",
            }
        )

        self.assertEqual(result["order_summary"].iloc[0], "Widget x5 @ $19.99")

    def test_apply_returns_dataframe(self):
        """Test that apply returns a DataFrame."""
        table = pd.DataFrame({"a": [1, 2]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"a": "a"},
            }
        )

        self.lm_api.chat = MagicMock(return_value=["[2, 4]"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "doubled",
                "src_table_columns": ["a"],
                "instruction": "Double the value",
            }
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("doubled", result.columns)
        self.assertIn("a", result.columns)

    def test_apply_preserves_original_columns(self):
        """Test that original columns are preserved in the result."""
        table = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table.to_csv(Path(self.temp_dir) / "test_ds" / "table.csv", index=False)
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.table",
                "target_table_id": "table",
                "column_mapping": {"col1": "col1", "col2": "col2"},
            }
        )

        self.lm_api.chat = MagicMock(return_value=["['x', 'y']"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "new_col",
                "src_table_columns": ["col1", "col2"],
                "instruction": "Create something",
            }
        )

        self.assertIn("col1", result.columns)
        self.assertIn("col2", result.columns)
        self.assertIn("new_col", result.columns)
        self.assertEqual(list(result["col1"]), [1, 2])
        self.assertEqual(list(result["col2"]), ["a", "b"])

    def test_get_name(self):
        """Test get_name returns correct action name."""
        name = self.semantic_col_gen.get_name()
        self.assertIsNotNone(name)
        self.assertIsInstance(name, str)

    def test_get_description(self):
        """Test get_description returns a non-empty string."""
        desc = self.semantic_col_gen.get_description()
        self.assertIsNotNone(desc)
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 0)

    def test_get_input_schema(self):
        """Test get_input_schema returns expected structure."""
        schema = self.semantic_col_gen.get_input_schema()
        self.assertIsInstance(schema, dict)
        self.assertIn("table", schema)
        self.assertIn("column_name", schema)
        self.assertIn("description", schema)

    def test_get_notes(self):
        """Test get_notes returns a non-empty string."""
        notes = self.semantic_col_gen.get_notes()
        self.assertIsNotNone(notes)
        self.assertIsInstance(notes, str)
        self.assertGreater(len(notes), 0)


class SemanticColumnGenerationMockLLMTests(unittest.TestCase):
    """
    Exercises SemanticColumnGeneration against a real DuckDB-backed DBAPI and
    the project's own MockLLM (rather than a hand-rolled MagicMock), so the
    generator-based `chat()` contract and queued-response ordering used in
    production are actually what's under test.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_db_path = Path(self.temp_dir) / "datasets"
        self.workspace_db_path = Path(self.temp_dir) / "workspaces"
        self.config = Config()
        self.config.DATA_SOURCES = ["test_ds"]
        self.config.ENABLE_SEMANTIC_COL_GEN = True
        self.logger = MagicMock()
        self.db_api = DBAPI(
            self.config,
            self.logger,
            str(self.dataset_db_path),
            str(self.workspace_db_path),
        )
        # A MagicMock stand-in for LanguageModelAPI; .chat is swapped per-test
        # to a real MockLLM's bound method so `chat()` behaves like a real
        # queued-response generator instead of a MagicMock return_value list.
        self.lm_api = MagicMock()
        self.semantic_col_gen = SemanticColumnGeneration(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )
        self.table_projection = TableProjection(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def tearDown(self) -> None:
        self.db_api.pneuma_db.close_all_connections()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _use_mock_llm(self, responses: list[str]) -> None:
        mock_llm = MockLLM(self.config, self.logger, responses=responses)
        self.lm_api.chat = mock_llm.chat

    def _ingest_and_project(
        self, table: pd.DataFrame, table_name: str = "table"
    ) -> None:
        ds_dir = Path(self.temp_dir) / "test_ds"
        os.makedirs(ds_dir, exist_ok=True)
        table.to_csv(ds_dir / f"{table_name}.csv", index=False)
        self.db_api.ingest_dataset(self.config.DATA_SOURCES[0], str(ds_dir))
        self.table_projection.apply(
            {
                "src_table_id": f"test_ds.{table_name}",
                "target_table_id": table_name,
                "column_mapping": {c: c for c in table.columns},
            }
        )

    def _read_full_table(self, table_name: str) -> pd.DataFrame:
        return self.db_api.execute_query(
            "user_id", "chat_id", f'SELECT * FROM "{table_name}" ORDER BY name;'
        )

    def test_apply_against_real_mock_llm_generator_interface(self):
        """Sanity check: SemanticColumnGeneration works against MockLLM's actual
        generator-based chat(), not just a MagicMock configured to look like one."""
        table = pd.DataFrame({"name": ["Alice", "Bob"]})
        self._ingest_and_project(table)
        self._use_mock_llm(["['A', 'B']"])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "tag",
                "src_table_columns": ["name"],
                "instruction": "tag",
            }
        )

        self.assertEqual(list(result["tag"]), ["A", "B"])

    def test_apply_handles_apostrophes_in_generated_values(self):
        """Regression test: generated values containing apostrophes used to
        crash the materialization INSERT. The old code built the VALUES()
        clause with Python's repr(), which quotes strings containing a single
        quote with double-quotes (e.g. "O'Brien") — DuckDB parses a
        double-quoted token as an identifier, not a string literal, so the
        insert raised a BinderException for any such value."""
        table = pd.DataFrame({"name": ["Alice", "Bob"]})
        self._ingest_and_project(table)
        self._use_mock_llm(['["O\'Brien", "D\'Angelo"]'])

        result = self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "surname",
                "src_table_columns": ["name"],
                "instruction": "assign a surname",
            }
        )

        self.assertEqual(list(result["surname"]), ["O'Brien", "D'Angelo"])

    def test_apply_batches_100_row_table_with_small_batch_size(self):
        """Simulates a table too large to load in one shot: 100 unique rows
        with SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE set low, forcing
        both the DB row-loading loop and the LLM value-batching loop to run
        across many round trips. One MockLLM response is queued per row
        batch, matching the code's one-chat-call-per-row-batch behavior when
        every value in a batch is unique."""
        n = 100
        batch_size = 15
        self.config.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE = batch_size
        names = [f"Person_{i:03d}" for i in range(n)]
        table = pd.DataFrame({"name": names})
        self._ingest_and_project(table)

        responses = []
        for start in range(0, n, batch_size):
            chunk = names[start : start + batch_size]
            responses.append(str([f"Tag-{name}" for name in chunk]))
        self._use_mock_llm(responses)

        self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "tag",
                "src_table_columns": ["name"],
                "instruction": "tag each person",
            }
        )

        full = self._read_full_table("table")
        self.assertEqual(len(full), n)
        for _, row in full.iterrows():
            self.assertEqual(row["tag"], f"Tag-{row['name']}")

    def test_apply_missing_llm_responses_yield_null_for_unmatched_batches(self):
        """When more row batches are processed than responses were queued,
        MockLLM falls back to a default noop message with no '[...]' payload,
        so those rows' new-column value should gracefully become NULL rather
        than crashing the action."""
        n = 40
        batch_size = 15
        self.config.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE = batch_size
        names = [f"Row_{i:03d}" for i in range(n)]
        table = pd.DataFrame({"name": names})
        self._ingest_and_project(table)

        # Only queue a response for the first of the three row batches.
        first_chunk = names[:batch_size]
        self._use_mock_llm([str([f"Tag-{name}" for name in first_chunk])])

        self.semantic_col_gen.apply(
            {
                "src_table_id": "table",
                "new_column_name": "tag",
                "src_table_columns": ["name"],
                "instruction": "tag each row",
            }
        )

        full = self._read_full_table("table")
        self.assertEqual(len(full), n)
        tagged = full[full["name"].isin(first_chunk)]
        untagged = full[~full["name"].isin(first_chunk)]
        for _, row in tagged.iterrows():
            self.assertEqual(row["tag"], f"Tag-{row['name']}")
        for _, row in untagged.iterrows():
            self.assertIsNone(row["tag"])


if __name__ == "__main__":
    unittest.main()
