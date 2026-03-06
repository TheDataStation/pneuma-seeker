
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pandas as pd
from pandas import DataFrame

sys.path.insert(
	0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

from pneuma_seeker.services.core.action_set.impl.table_union import TableUnion
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config


class TableUnionTests(unittest.TestCase):
	"""Unit tests for TableUnion."""

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
		self.table_union = TableUnion(
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

	def _create_custom_dataset_table(
		self, table_name: str, columns: list[str], rows: list[tuple]
	) -> None:
		table_df = DataFrame(rows, columns=columns)
		os.makedirs(Path(self.temp_dir) / "test_ds", exist_ok=True)
		table_df.to_csv(Path(self.temp_dir) / "test_ds" / f"{table_name}.csv", index=False)
		self.db_api.ingest_dataset(self.config.DATA_SOURCES[0], str(Path(self.temp_dir) / "test_ds"))

	def test_apply_unions_dataset_tables_by_regex_and_adds_source_column(self):
		self._create_custom_dataset_table(
			"data_2021",
			columns=["id", "val"],
			rows=[(1, "a"), (2, "b")],
		)
		self._create_custom_dataset_table(
			"data_2022",
			columns=["id", "val"],
			rows=[(3, "c")],
		)

		result = self.table_union.apply(
			{
				"table_ids": [r"re:^test_ds\\.data_\\d{4}$"],
				"result_table_id": "unioned",
				"provenance_column_name": "year",
				"provenance_regex": r".*_(\\d{4})$",
			}
		)

		self.assertIn("year", result.columns)
		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'SELECT * FROM "unioned" ORDER BY "id"',
		)
		self.assertEqual(len(persisted), 3)
		self.assertEqual(set(persisted["year"].tolist()), {"2021", "2022"})

	def test_apply_unions_workspace_tables_by_regex(self):
		# Create two workspace tables
		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE OR REPLACE TABLE "tmp_2021" AS SELECT 1 AS "a";',
		)
		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE OR REPLACE TABLE "tmp_2022" AS SELECT 2 AS "a";',
		)

		self.table_union.apply(
			{
				"table_ids": [r"re:^tmp_\\d{4}$"],
				"result_table_id": "unioned",
				"provenance_column_name": "year",
				"provenance_regex": r".*_(\\d{4})$",
			}
		)

		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'\n'.join([
				'SELECT "a", "year"',
				'FROM "unioned"',
				'ORDER BY "a"',
			]),
		)
		self.assertEqual(persisted.iloc[0]["a"], 1)
		self.assertEqual(persisted.iloc[1]["a"], 2)
		self.assertEqual(set(persisted["year"].tolist()), {"2021", "2022"})

	def test_apply_unions_tables_with_different_schemas_using_nulls(self):
		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE OR REPLACE TABLE "t1" AS SELECT 1 AS "a", 10 AS "b";',
		)
		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE OR REPLACE TABLE "t2" AS SELECT 2 AS "a", 20 AS "c";',
		)

		self.table_union.apply(
			{
				"table_ids": ["t1", "t2"],
				"result_table_id": "unioned",
				"provenance_column_name": "src",
				"provenance_regex": r"^(.*)$",
			}
		)

		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'\n'.join([
				'SELECT "a", "b", "c", "src"',
				'FROM "unioned"',
				'ORDER BY "a"',
			]),
		)
		self.assertEqual(list(persisted.columns), ["a", "b", "c", "src"])
		self.assertTrue(pd.isna(persisted.iloc[0]["c"]))
		self.assertTrue(pd.isna(persisted.iloc[1]["b"]))

	def test_apply_can_extract_source_label_via_regex(self):
		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE OR REPLACE TABLE "tmp_2021" AS SELECT 1 AS "a";',
		)
		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'CREATE OR REPLACE TABLE "tmp_2022" AS SELECT 2 AS "a";',
		)

		self.table_union.apply(
			{
				"table_ids": [r"re:^tmp_\\d{4}$"],
				"result_table_id": "unioned",
				"provenance_regex": r".*_(\\d{4})$",
				"provenance_column_name": "year",
			}
		)

		persisted = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			'SELECT "year" FROM "unioned" ORDER BY "year"',
		)
		self.assertEqual(persisted.iloc[0]["year"], "2021")
		self.assertEqual(persisted.iloc[1]["year"], "2022")

	def test_apply_invalid_inputs_raise(self):
		with self.assertRaises(ValueError):
			self.table_union.apply({"table_ids": "t1", "result_table_id": "u"})

		with self.assertRaises(ValueError):
			self.table_union.apply({"table_ids": [], "result_table_id": "u"})

		with self.assertRaises(ValueError):
			self.table_union.apply({"table_ids": ["t1"], "result_table_id": None})

		with self.assertRaises(ValueError):
			self.table_union.apply(
				{
					"table_ids": ["t1"],
					"result_table_id": "u",
					"provenance_column_name": "year",
				}
			)

		with self.assertRaises(ValueError):
			self.table_union.apply(
				{
					"table_ids": ["t1"],
					"result_table_id": "u",
					"provenance_regex": r"(.*)",
				}
			)

		with pytest.raises(ValueError):
			self.table_union.apply(
				{
					"table_ids": ["re:("],
					"result_table_id": "u",
					"provenance_column_name": "x",
					"provenance_regex": r"(.*)",
				}
			)
