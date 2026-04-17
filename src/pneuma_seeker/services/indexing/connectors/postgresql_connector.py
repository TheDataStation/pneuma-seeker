from __future__ import annotations

from typing import Any, Iterator

import duckdb

from pneuma_seeker.services.indexing.connectors.base import SourceConnector
from pneuma_seeker.shared.str_processor import clean_column_table_name


class PostgreSQLConnector(SourceConnector):
	"""Connector for PostgreSQL sources via DuckDB postgres extension."""

	def __init__(self, config: dict[str, Any]):
		super().__init__(config)
		self._connection_string = self._build_connection_string(config)
		self.schema: str | None = config.get("schema")
		self.row_limit: int | None = config.get("row_limit")

	@property
	def source_type(self) -> str:
		return "postgres"

	@property
	def connection_string(self) -> str:
		return self._connection_string

	def _build_connection_string(self, config: dict[str, Any]) -> str:
		connection_string = config.get("connection_string")
		if connection_string:
			return str(connection_string)

		required = ["host", "port", "user", "password", "dbname"]
		missing = [key for key in required if key not in config]
		if missing:
			raise ValueError(
				"PostgreSQL connector config missing required keys: "
				+ ", ".join(sorted(missing))
			)

		return (
			f"host={config['host']} "
			f"port={config['port']} "
			f"user={config['user']} "
			f"password={config['password']} "
			f"dbname={config['dbname']}"
		)

	def _open_conn(self) -> duckdb.DuckDBPyConnection:
		con = duckdb.connect(":memory:")
		con.execute("INSTALL postgres;")
		con.execute("LOAD postgres;")
		return con

	def _escape_sql_literal(self, value: str) -> str:
		return value.replace("'", "''")

	def _quote_ident(self, name: str) -> str:
		return '"' + name.replace('"', '""') + '"'

	def _parse_stream(self, stream: str) -> tuple[str, str]:
		if "." in stream:
			schema_name, table_name = stream.split(".", 1)
			return schema_name, table_name

		default_schema = self.schema or "public"
		return default_schema, stream

	def check_connection(self) -> bool:
		con = self._open_conn()
		try:
			conn = self._escape_sql_literal(self.connection_string)
			con.execute(f"ATTACH '{conn}' AS source_db (TYPE postgres, READ_ONLY)")
			con.execute("DETACH source_db")
			return True
		except Exception:
			return False
		finally:
			con.close()

	def discover(self) -> list[dict[str, Any]]:
		con = self._open_conn()
		conn = self._escape_sql_literal(self.connection_string)
		con.execute(f"ATTACH '{conn}' AS source_db (TYPE postgres, READ_ONLY)")
		try:
			if self.schema:
				rows = con.execute(
					"""
					SELECT table_schema, table_name
					FROM source_db.information_schema.tables
					WHERE table_type = 'BASE TABLE' AND table_schema = ?
					ORDER BY table_schema, table_name
					""",
					[self.schema],
				).fetchall()
			else:
				rows = con.execute(
					"""
					SELECT table_schema, table_name
					FROM source_db.information_schema.tables
					WHERE table_type = 'BASE TABLE'
					  AND table_schema NOT IN ('pg_catalog', 'information_schema')
					ORDER BY table_schema, table_name
					"""
				).fetchall()

			discovered = []
			for table_schema, table_name in rows:
				stream_name = clean_column_table_name(f"{table_schema}_{table_name}")
				discovered.append(
					{
						"stream": f"{table_schema}.{table_name}",
						"table_name": stream_name,
						"schema": table_schema,
						"source_table_name": table_name,
					}
				)
			return discovered
		finally:
			con.execute("DETACH source_db")
			con.close()

	def read(self, stream: str) -> Iterator[dict[str, Any]]:
		schema_name, table_name = self._parse_stream(stream)

		con = self._open_conn()
		conn = self._escape_sql_literal(self.connection_string)
		con.execute(f"ATTACH '{conn}' AS source_db (TYPE postgres, READ_ONLY)")
		try:
			sql = (
				"SELECT * FROM source_db."
				f"{self._quote_ident(schema_name)}.{self._quote_ident(table_name)}"
			)
			if self.row_limit is not None and int(self.row_limit) > 0:
				sql += f" LIMIT {int(self.row_limit)}"

			table = con.execute(sql).fetchdf()
			for row in table.to_dict(orient="records"):
				yield row
		finally:
			con.execute("DETACH source_db")
			con.close()
