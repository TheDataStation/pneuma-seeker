from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import duckdb


class IndexingMetadataStore:
	"""Persists indexing run metadata in DuckDB."""

	def __init__(self, db_path: str | None = None):
		if db_path:
			self.db_path = Path(db_path)
		else:
			self.db_path = Path(__file__).resolve().parent / "indexing.db"

		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._con = duckdb.connect(self.db_path.as_posix(), read_only=False)
		self._define_tables()

	def _define_tables(self):
		self._con.execute(
			"""
			CREATE TABLE IF NOT EXISTS indexing_runs (
				run_id               VARCHAR PRIMARY KEY,
				dataset_name         VARCHAR NOT NULL,
				source_type          VARCHAR NOT NULL,
				source_config_json   VARCHAR NOT NULL,
				snapshot_id          VARCHAR NOT NULL,
				status               VARCHAR NOT NULL,
				error_message        VARCHAR,
				indexed_stream_count INTEGER,
				indexed_table_count  INTEGER,
				created_at           TIMESTAMP DEFAULT now(),
				updated_at           TIMESTAMP DEFAULT now(),
				completed_at         TIMESTAMP
			)
			"""
		)

	def record_run_started(
		self,
		dataset_name: str,
		source_type: str,
		source_config: dict,
		snapshot_id: str,
	) -> str:
		run_id = str(uuid4())
		self._con.execute(
			"""
			INSERT INTO indexing_runs (
				run_id,
				dataset_name,
				source_type,
				source_config_json,
				snapshot_id,
				status
			) VALUES (?, ?, ?, ?, ?, 'RUNNING')
			""",
			[
				run_id,
				dataset_name,
				source_type,
				json.dumps(source_config, sort_keys=True),
				snapshot_id,
			],
		)
		return run_id

	def mark_run_succeeded(
		self,
		run_id: str,
		indexed_stream_count: int,
		indexed_table_count: int,
	) -> None:
		self._con.execute(
			"""
			UPDATE indexing_runs
			SET status = 'SUCCEEDED',
				indexed_stream_count = ?,
				indexed_table_count = ?,
				updated_at = now(),
				completed_at = now()
			WHERE run_id = ?
			""",
			[indexed_stream_count, indexed_table_count, run_id],
		)

	def mark_run_failed(self, run_id: str, error_message: str) -> None:
		self._con.execute(
			"""
			UPDATE indexing_runs
			SET status = 'FAILED',
				error_message = ?,
				updated_at = now(),
				completed_at = now()
			WHERE run_id = ?
			""",
			[error_message, run_id],
		)

	def get_run(self, run_id: str) -> dict | None:
		row = self._con.execute(
			"""
			SELECT
				run_id,
				dataset_name,
				source_type,
				source_config_json,
				snapshot_id,
				status,
				error_message,
				indexed_stream_count,
				indexed_table_count,
				created_at,
				updated_at,
				completed_at
			FROM indexing_runs
			WHERE run_id = ?
			""",
			[run_id],
		).fetchone()

		if row is None:
			return None

		return {
			"run_id": row[0],
			"dataset_name": row[1],
			"source_type": row[2],
			"source_config_json": row[3],
			"snapshot_id": row[4],
			"status": row[5],
			"error_message": row[6],
			"indexed_stream_count": row[7],
			"indexed_table_count": row[8],
			"created_at": row[9],
			"updated_at": row[10],
			"completed_at": row[11],
		}

	def get_latest_run(self, dataset_name: str) -> dict | None:
		row = self._con.execute(
			"""
			SELECT
				run_id,
				dataset_name,
				source_type,
				source_config_json,
				snapshot_id,
				status,
				error_message,
				indexed_stream_count,
				indexed_table_count,
				created_at,
				updated_at,
				completed_at
			FROM indexing_runs
			WHERE dataset_name = ?
			ORDER BY created_at DESC
			LIMIT 1
			""",
			[dataset_name],
		).fetchone()

		if row is None:
			return None

		return {
			"run_id": row[0],
			"dataset_name": row[1],
			"source_type": row[2],
			"source_config_json": row[3],
			"snapshot_id": row[4],
			"status": row[5],
			"error_message": row[6],
			"indexed_stream_count": row[7],
			"indexed_table_count": row[8],
			"created_at": row[9],
			"updated_at": row[10],
			"completed_at": row[11],
		}

	def close(self):
		self._con.close()
