import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from pandas import DataFrame
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

from pneuma_seeker.services.core.action_set.impl.table_projection import (
    TableProjection,
)
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config


class TableProjectionTests(unittest.TestCase):
    """Unit tests for TableProjection."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_db_path = Path(self.temp_dir) / "datasets"
        self.workspace_db_path = Path(self.temp_dir) / "workspaces"
        self.config = Config()
        self.config.DATA_SOURCES = ["test_ds"]
        self.logger = MagicMock()
        self.lm_api = MagicMock()
        self.db_api = DBAPI(
            self.config,
            self.logger,
            dataset_db_path=str(self.dataset_db_path),
            workspace_db_path=str(self.workspace_db_path),
        )
        self.user_id = "user_id"
        self.chat_id = "chat_id"
        self.table_projection = TableProjection(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.lm_api,
        )

    def tearDown(self) -> None:
        self.db_api.pneuma_db.close_all_connections()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_src_table(self, table_name: str) -> None:
        self._create_custom_table(
            table_name,
            columns=["a", "b", "c"],
            rows=[(1, 3, 5), (2, 4, 6)],
        )

    def _create_custom_table(
        self, table_name: str, columns: list[str], rows: list[tuple]
    ):
        table_df = DataFrame(rows, columns=columns)
        os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
        table_df.to_csv(
            Path(self.temp_dir) / "test_ds" / f"{table_name}.csv", index=False
        )
        self.db_api.ingest_dataset(
            self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds")
        )

    def test_apply_projects_columns_and_returns_sample(self):
        self._create_src_table("src_table")

        result = self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "target_table",
                "column_mapping": {"c": "c", "a": "a"},
            }
        )

        self.assertEqual(list(result.columns), ["c", "a"])
        self.assertEqual(result.iloc[0]["c"], 5)
        self.assertEqual(result.iloc[1]["a"], 2)

        persisted = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            'SELECT * FROM "target_table" ORDER BY "a"',
        )
        self.assertEqual(list(persisted.columns), ["c", "a"])
        self.assertEqual(persisted.iloc[0]["c"], 5)

    def test_apply_overwrites_existing_target(self):
        self._create_src_table("src_table")
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            'CREATE TABLE "target_table" AS SELECT 1 AS "x";',
        )

        result = self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "target_table",
                "column_mapping": {"b": "b"},
            }
        )

        self.assertEqual(list(result.columns), ["b"])
        persisted = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            'SELECT * FROM "target_table"',
        )
        self.assertEqual(list(persisted.columns), ["b"])

    def test_apply_invalid_src_table_id_raises(self):
        with self.assertRaises(ValueError) as context:
            self.table_projection.apply(
                {
                    "src_table_id": None,
                    "target_table_id": "target_table",
                    "column_mapping": {"a": "a"},
                }
            )
        self.assertIn("Input 'src_table_id' must be a string", str(context.exception))

    def test_apply_invalid_target_table_id_raises(self):
        with self.assertRaises(ValueError) as context:
            self.table_projection.apply(
                {
                    "src_table_id": "test_ds.src_table",
                    "target_table_id": 123,
                    "column_mapping": {"a": "a"},
                }
            )
        self.assertIn(
            "Input 'target_table_id' must be a string", str(context.exception)
        )

    def test_apply_invalid_columns_type_raises(self):
        with self.assertRaises(ValueError) as context:
            self.table_projection.apply(
                {
                    "src_table_id": "test_ds.src_table",
                    "target_table_id": "target_table",
                    "column_mapping": "a",
                }
            )
        self.assertIn(
            "Input 'column_mapping' must be a dict[str, str]",
            str(context.exception),
        )

    def test_apply_invalid_columns_values_raises(self):
        with self.assertRaises(ValueError) as context:
            self.table_projection.apply(
                {
                    "src_table_id": "test_ds.src_table",
                    "target_table_id": "target_table",
                    "column_mapping": {"a": 1},
                }
            )
        self.assertIn(
            "Input 'column_mapping' must map strings to strings",
            str(context.exception),
        )

    def test_apply_empty_columns_raises(self):
        with self.assertRaises(ValueError) as context:
            self.table_projection.apply(
                {
                    "src_table_id": "test_ds.src_table",
                    "target_table_id": "target_table",
                    "column_mapping": {},
                }
            )
        self.assertIn(
            "column_mapping must contain at least one mapping",
            str(context.exception),
        )

    def test_apply_missing_column_raises(self):
        self._create_src_table("src_table")

        with pytest.raises(Exception):
            self.table_projection.apply(
                {
                    "src_table_id": "test_ds.src_table",
                    "target_table_id": "target_table",
                    "column_mapping": {"missing": "missing"},
                }
            )

    def test_apply_missing_src_table_raises(self):
        with pytest.raises(Exception):
            self.table_projection.apply(
                {
                    "src_table_id": "missing_table",
                    "target_table_id": "target_table",
                    "column_mapping": {"a": "a"},
                }
            )

    def test_apply_handles_special_column_names(self):
        self._create_custom_table(
            "src_table",
            columns=["col name", "x-y"],
            rows=[(1, 2), (3, 4)],
        )

        result = self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "target_table",
                "column_mapping": {"x_y": "x_y", "col_name": "col_name"},
            }
        )

        self.assertEqual(list(result.columns), ["x_y", "col_name"])
        self.assertEqual(result.iloc[0]["x_y"], 2)

    def test_apply_empty_source_table_preserves_schema(self):
        self._create_custom_table(
            "src_table",
            columns=["a", "b"],
            rows=[],
        )

        result = self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "target_table",
                "column_mapping": {"b": "b"},
            }
        )

        self.assertEqual(list(result.columns), ["b"])
        self.assertEqual(len(result), 0)

    def test_apply_returns_at_most_five_rows(self):
        self._create_custom_table(
            "src_table",
            columns=["a"],
            rows=[(1,), (2,), (3,), (4,), (5,), (6,), (7,)],
        )

        result = self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "target_table",
                "column_mapping": {"a": "a"},
            }
        )

        self.assertEqual(list(result.columns), ["a"])
        self.assertEqual(len(result), 5)

    def test_apply_projects_with_alias_mapping_and_returns_sample(self):
        self._create_src_table("src_table")

        result = self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "target_table",
                "column_mapping": {"c_alias": "c", "a_alias": "a"},
            }
        )

        self.assertEqual(list(result.columns), ["c_alias", "a_alias"])
        self.assertEqual(result.iloc[0]["c_alias"], 5)
        self.assertEqual(result.iloc[1]["a_alias"], 2)

        persisted = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            'SELECT * FROM "target_table" ORDER BY "a_alias"',
        )
        self.assertEqual(list(persisted.columns), ["c_alias", "a_alias"])
        self.assertEqual(persisted.iloc[0]["c_alias"], 5)

    def test_apply_alias_mapping_missing_source_column_raises(self):
        self._create_src_table("src_table")

        with pytest.raises(Exception):
            self.table_projection.apply(
                {
                    "src_table_id": "test_ds.src_table",
                    "target_table_id": "target_table",
                    "column_mapping": {"x": "missing"},
                }
            )

    def test_apply_can_project_from_workspace_table(self):
        self._create_src_table("src_table")

        # First projection creates an intermediate/workspace table.
        self.table_projection.apply(
            {
                "src_table_id": "test_ds.src_table",
                "target_table_id": "intermediate_table",
                "column_mapping": {"a": "a", "b": "b"},
            }
        )

        # Second projection uses that workspace table as the source.
        result = self.table_projection.apply(
            {
                "src_table_id": "intermediate_table",
                "target_table_id": "target_table",
                "column_mapping": {"b": "b"},
            }
        )

        self.assertEqual(list(result.columns), ["b"])
        persisted = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            'SELECT * FROM "target_table" ORDER BY "b"',
        )
        self.assertEqual(list(persisted.columns), ["b"])
        self.assertEqual(persisted.iloc[0]["b"], 3)

    def test_apply_accepts_quoted_dataset_table_ref(self):
        self._create_src_table("src_table")

        result = self.table_projection.apply(
            {
                "src_table_id": 'test_ds."src_table"',
                "target_table_id": "target_table",
                "column_mapping": {"a": "a"},
            }
        )

        self.assertEqual(list(result.columns), ["a"])
