
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pandas import DataFrame

sys.path.insert(
	0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

from pneuma_seeker.services.core.action_set.impl.equality_join import EqualityJoin
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config


class EqualityJoinTests(unittest.TestCase):
	"""Unit tests for EqualityJoin."""

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
		self.equality_join = EqualityJoin(
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

	def _create_custom_table(
		self, table_name: str, columns: list[str], rows: list[tuple]
	) -> None:
		table_df = DataFrame(rows, columns=columns)
		os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
		table_df.to_csv(Path(self.temp_dir) / "test_ds" / f"{table_name}.csv", index=False)
		self.db_api.ingest_dataset(self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds"))

	def test_apply_joins_on_single_key_and_returns_sample(self):
		self._create_custom_table(
			"left_table",
			columns=["id", "a"],
			rows=[(1, "L1"), (2, "L2"), (3, "L3")],
		)
		self._create_custom_table(
			"right_table",
			columns=["id", "b"],
			rows=[(2, "R2"), (3, "R3"), (4, "R4")],
		)

		result = self.equality_join.apply(
			{
				"left_table_id": "test_ds.left_table",
				"right_table_id": "test_ds.right_table",
				"left_table_column_keys": ["id"],
				"right_table_column_keys": ["id"],
				"result_table_id": "joined_table",
			}
		)

		self.assertIn("left_id", result.columns)
		self.assertIn("left_a", result.columns)
		self.assertIn("right_id", result.columns)
		self.assertIn("right_b", result.columns)

		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'SELECT * FROM "joined_table" ORDER BY "left_id"',
		)
		self.assertEqual(len(persisted), 2)
		self.assertEqual(persisted.iloc[0]["left_id"], 2)
		self.assertEqual(persisted.iloc[0]["left_a"], "L2")
		self.assertEqual(persisted.iloc[0]["right_b"], "R2")

	def test_apply_joins_on_multiple_keys(self):
		self._create_custom_table(
			"left_table",
			columns=["k1", "k2", "v"],
			rows=[("a", 1, "L"), ("a", 2, "L2"), ("b", 1, "L3")],
		)
		self._create_custom_table(
			"right_table",
			columns=["x", "y", "w"],
			rows=[("a", 1, "R"), ("a", 3, "R2"), ("b", 1, "R3")],
		)

		self.equality_join.apply(
			{
				"left_table_id": "test_ds.left_table",
				"right_table_id": "test_ds.right_table",
				"left_table_column_keys": ["k1", "k2"],
				"right_table_column_keys": ["x", "y"],
				"result_table_id": "joined_table",
			}
		)

		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'SELECT "left_k1", "left_k2", "right_w" FROM "joined_table" ORDER BY "left_k1", "left_k2"',
		)
		self.assertEqual(len(persisted), 2)
		self.assertEqual(persisted.iloc[0]["left_k1"], "a")
		self.assertEqual(persisted.iloc[0]["left_k2"], 1)
		self.assertEqual(persisted.iloc[0]["right_w"], "R")

	def test_apply_overwrites_existing_target(self):
		self._create_custom_table(
			"left_table",
			columns=["id"],
			rows=[(1,), (2,)],
		)
		self._create_custom_table(
			"right_table",
			columns=["id"],
			rows=[(2,), (3,)],
		)

		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE TABLE "joined_table" AS SELECT 1 AS "x";',
		)

		result = self.equality_join.apply(
			{
				"left_table_id": "test_ds.left_table",
				"right_table_id": "test_ds.right_table",
				"left_table_column_keys": ["id"],
				"right_table_column_keys": ["id"],
				"result_table_id": "joined_table",
			}
		)

		self.assertIn("left_id", result.columns)
		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'SELECT * FROM "joined_table"',
		)
		self.assertIn("left_id", persisted.columns)
		self.assertNotIn("x", persisted.columns)

	def test_apply_invalid_left_table_id_raises(self):
		with self.assertRaises(ValueError) as context:
			self.equality_join.apply(
				{
					"left_table_id": None,
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": ["id"],
					"right_table_column_keys": ["id"],
					"result_table_id": "joined_table",
				}
			)
		self.assertIn("Input 'left_table_id' must be a string", str(context.exception))

	def test_apply_invalid_right_table_id_raises(self):
		with self.assertRaises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": 123,
					"left_table_column_keys": ["id"],
					"right_table_column_keys": ["id"],
					"result_table_id": "joined_table",
				}
			)

	def test_apply_invalid_result_table_id_raises(self):
		with self.assertRaises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": ["id"],
					"right_table_column_keys": ["id"],
					"result_table_id": 1,
				}
			)

	def test_apply_invalid_key_types_raise(self):
		with self.assertRaises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": "id",
					"right_table_column_keys": ["id"],
					"result_table_id": "joined_table",
				}
			)

		with self.assertRaises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": ["id"],
					"right_table_column_keys": [1],
					"result_table_id": "joined_table",
				}
			)

	def test_apply_empty_keys_raise(self):
		with self.assertRaises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": [],
					"right_table_column_keys": [],
					"result_table_id": "joined_table",
				}
			)

	def test_apply_key_length_mismatch_raises(self):
		with self.assertRaises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": ["a", "b"],
					"right_table_column_keys": ["c"],
					"result_table_id": "joined_table",
				}
			)

	def test_apply_missing_column_raises(self):
		self._create_custom_table(
			"left_table",
			columns=["id"],
			rows=[(1,), (2,)],
		)
		self._create_custom_table(
			"right_table",
			columns=["id"],
			rows=[(2,), (3,)],
		)

		with pytest.raises(ValueError):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.left_table",
					"right_table_id": "test_ds.right_table",
					"left_table_column_keys": ["missing"],
					"right_table_column_keys": ["id"],
					"result_table_id": "joined_table",
				}
			)

	def test_apply_missing_table_raises(self):
		with pytest.raises(Exception):
			self.equality_join.apply(
				{
					"left_table_id": "test_ds.missing_left",
					"right_table_id": "test_ds.missing_right",
					"left_table_column_keys": ["id"],
					"right_table_column_keys": ["id"],
					"result_table_id": "joined_table",
				}
			)

	def test_apply_handles_special_column_names(self):
		# Column names are cleaned on ingestion: "col name" -> "col_name", "x-y" -> "x_y"
		self._create_custom_table(
			"left_table",
			columns=["col name", "v"],
			rows=[("k1", 1), ("k2", 2)],
		)
		self._create_custom_table(
			"right_table",
			columns=["x-y", "w"],
			rows=[("k2", 20), ("k3", 30)],
		)

		self.equality_join.apply(
			{
				"left_table_id": "test_ds.left_table",
				"right_table_id": "test_ds.right_table",
				"left_table_column_keys": ["col_name"],
				"right_table_column_keys": ["x_y"],
				"result_table_id": "joined_table",
			}
		)

		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'SELECT "left_col_name", "right_x_y" FROM "joined_table"',
		)
		self.assertEqual(len(persisted), 1)
		self.assertEqual(persisted.iloc[0]["left_col_name"], "k2")
