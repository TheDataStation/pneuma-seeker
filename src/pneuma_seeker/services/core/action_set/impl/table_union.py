
import re
from typing import Any

from pandas import DataFrame

from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Applicable
from pneuma_seeker.shared.schemas.core.action import ActionNames


class TableUnion(Action, Applicable):
	def get_name(self) -> str:
		return ActionNames.TABLE_UNION.value

	def get_description(self) -> str:
		return f"""**{ActionNames.TABLE_UNION.value}**
    - Unions multiple tables (internal, external, or intermediate) into a single workspace table.
    - Each entry in `table_ids` may be either:
        - an explicit table reference (e.g., `my_intermediate_table` or `y."x"`), OR
        - a regex selector prefixed with `re:` (e.g., `re:^data_\\d{{4}}$` or `re:^y\\.data_\\d{{4}}$`).
    - The output includes a provenance column derived from each source table ID.
    - Args: {{
		"table_ids": ["<table ref or re:<pattern>>", ...],
		"result_table_id": "<intermediate/target table name for the union result>",
		"provenance_column_name": "<output column name for provenance>",
		"provenance_regex": "<regex used to extract provenance from each table id/name>"
	}}
    - Notes:
        - Regex patterns are matched against both the full display name (e.g., `y.data_2021`) and the bare table name (e.g., `data_2021`).
        - If tables have different schemas, missing columns are filled with NULL.
        - The output table (`result_table_id`) is always created/overwritten in the workspace."""

	def get_input_schema(self) -> dict[str, str]:
		return {}

	def get_notes(self) -> str:
		return ""

	def apply(self, input: dict[str, Any]) -> DataFrame:
		table_ids = input.get("table_ids")
		result_table_id = input.get("result_table_id")
		provenance_column_name = input.get("provenance_column_name")
		provenance_regex = input.get("provenance_regex")

		if not isinstance(table_ids, list) or not all(isinstance(x, str) for x in table_ids):
			raise ValueError("Input 'table_ids' must be a list of strings.")
		if len(table_ids) == 0:
			raise ValueError("table_ids must contain at least one entry.")
		if not isinstance(result_table_id, str):
			raise ValueError("Input 'result_table_id' must be a string.")
		if not isinstance(provenance_column_name, str) or not provenance_column_name.strip():
			raise ValueError("Input 'provenance_column_name' must be a non-empty string.")
		if not isinstance(provenance_regex, str) or not provenance_regex.strip():
			raise ValueError("Input 'provenance_regex' must be a non-empty string.")

		provenance_column_name = provenance_column_name.strip()
		provenance_regex = provenance_regex.strip()

		resolved = self.__resolve_table_ids_and_patterns(table_ids)
		if not resolved:
			raise ValueError("No tables were resolved from table_ids.")

		try:
			prov_re = re.compile(self.__unescape_regex(provenance_regex))
		except re.error as e:
			raise ValueError(f"Invalid provenance_regex: {e}")

		# Collect per-table column lists
		tables: list[tuple[str, str, list[str]]] = []  # (display_name, table_ref, columns)
		for display_name, table_ref in resolved:
			cols = self.__get_table_columns(table_ref)
			tables.append((display_name, table_ref, cols))

		# Build superset of columns in deterministic order (first-seen)
		all_columns: list[str] = []
		seen: set[str] = set()
		for _, _, cols in tables:
			for c in cols:
				if c not in seen:
					seen.add(c)
					all_columns.append(c)

		if provenance_column_name in seen:
			raise ValueError(
				f"provenance_column_name {provenance_column_name!r} conflicts with an existing column name."
			)

		select_statements: list[str] = []
		for display_name, table_ref, cols in tables:
			cols_set = set(cols)
			projected_cols: list[str] = []
			for c in all_columns:
				if c in cols_set:
					projected_cols.append(f"t.{self.__quote_ident(c)} AS {self.__quote_ident(c)}")
				else:
					projected_cols.append(f"NULL AS {self.__quote_ident(c)}")

			# Always include a provenance column extracted from the table id/name.
			label_val = self.__extract_provenance_value(prov_re, display_name)
			projected_cols.append(
				f"{self.__quote_literal(label_val)} AS {self.__quote_ident(provenance_column_name)}"
			)

			select_statements.append(
				f"SELECT {', '.join(projected_cols)} FROM {table_ref} AS t"
			)

		union_sql = "\nUNION ALL\n".join(select_statements)

		self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			f"CREATE OR REPLACE TABLE {self.__quote_ident(result_table_id)} AS\n{union_sql};",
		)

		return self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			f"SELECT * FROM {self.__quote_ident(result_table_id)} LIMIT 5;",
		)

	def __quote_ident(self, identifier: str) -> str:
		escaped = identifier.replace('"', '""')
		return f'"{escaped}"'

	def __quote_literal(self, value: str) -> str:
		# DuckDB SQL string literal escaping: single quotes are doubled.
		return "'" + value.replace("'", "''") + "'"

	_IDENT_RE = r"[A-Za-z_][A-Za-z0-9_]*"
	_QUOTED_IDENT_RE = r'"(?:[^"]|"")*"'
	_TABLE_REF_RE = re.compile(
		rf"^(?:{_IDENT_RE}|{_QUOTED_IDENT_RE})(?:\.(?:{_IDENT_RE}|{_QUOTED_IDENT_RE}))?$"
	)

	def __validate_table_ref(self, table_ref: str) -> str:
		if not isinstance(table_ref, str):
			raise ValueError("Table reference must be a string.")
		stripped = table_ref.strip()
		if not self._TABLE_REF_RE.fullmatch(stripped):
			raise ValueError(
				"Invalid table reference format. Use a bare table name or dataset-qualified form like dataset.table or dataset.\"table\"."
			)
		return stripped

	def __link_datasets_for_table_ref(self, table_ref: str) -> None:
		if "." not in table_ref:
			return
		dataset_part = table_ref.split(".", 1)[0].strip()
		if dataset_part.startswith('"') and dataset_part.endswith('"'):
			dataset_part = dataset_part[1:-1].replace('""', '"')
		if self.config.DATA_SOURCES and dataset_part in self.config.DATA_SOURCES:
			self.db_api.link_dataset_tables(self.user_id, self.chat_id, dataset_part)

	def __list_available_tables(self) -> list[tuple[str, str]]:
		"""Return (display_name, sql_ref) for all visible base tables.

		display_name is:
		- workspace: <table>
		- attached datasets: <catalog>.<table>
		"""
		rows = self.db_api.execute_query(
			self.user_id,
			self.chat_id,
			"""
			SELECT table_catalog, table_schema, table_name
			FROM information_schema.tables
			WHERE table_type = 'BASE TABLE'
			""",
		)

		results: list[tuple[str, str]] = []
		for _, r in rows.iterrows():
			catalog = str(r["table_catalog"])
			schema = str(r["table_schema"])
			name = str(r["table_name"])

			# Skip system schemas
			if schema in {"information_schema", "pg_catalog"}:
				continue

			if catalog == "main" and schema == "main":
				results.append((name, self.__quote_ident(name)))
			else:
				# DuckDB allows catalog.table shorthand for attached DBs
				results.append((f"{catalog}.{name}", f"{self.__quote_ident(catalog)}.{self.__quote_ident(name)}"))
		return results

	def __resolve_table_ids_and_patterns(self, table_ids: list[str]) -> list[tuple[str, str]]:
		"""Resolve a mixed list of explicit table refs and regex patterns.

		Regex patterns are specified as strings starting with `re:` or `regex:`.
		They are matched against both:
		- workspace table names (e.g. my_table)
		- dataset-qualified display names (e.g. dataset.my_table)
		"""
		# Attach any datasets explicitly referenced.
		for item in table_ids:
			stripped = item.strip()
			if stripped.startswith("re:") or stripped.startswith("regex:"):
				continue
			try:
				table_ref = self.__validate_table_ref(stripped)
			except ValueError:
				# Not a safe table ref; leave it to regex resolution.
				continue
			self.__link_datasets_for_table_ref(table_ref)

		# Best-effort attach known datasets so patterns can match them.
		if self.config.DATA_SOURCES:
			for ds in self.config.DATA_SOURCES:
				try:
					self.db_api.link_dataset_tables(self.user_id, self.chat_id, ds)
				except FileNotFoundError:
					continue
				except Exception:
					continue

		available = self.__list_available_tables()
		available_by_display: dict[str, str] = {d: r for d, r in available}

		resolved: list[tuple[str, str]] = []
		seen: set[str] = set()

		def add(display: str, ref: str) -> None:
			if display in seen:
				return
			seen.add(display)
			resolved.append((display, ref))

		for item in table_ids:
			stripped = item.strip()
			is_regex = stripped.startswith("re:") or stripped.startswith("regex:")
			if is_regex:
				pattern = self.__unescape_regex(stripped.split(":", 1)[1])
				try:
					compiled = re.compile(pattern)
				except re.error as e:
					raise ValueError(f"Invalid regex pattern {pattern!r}: {e}")

				matches = [
					(display, ref)
					for display, ref in available
					if compiled.search(display) or compiled.search(display.split(".")[-1])
				]
				for display, ref in sorted(matches, key=lambda x: x[0]):
					add(display, ref)
				continue

			# Try explicit ref
			try:
				table_ref = self.__validate_table_ref(stripped)
				self.__link_datasets_for_table_ref(table_ref)
				display = self.__display_name_from_ref(table_ref)
				add(display, available_by_display.get(display, table_ref))
				continue
			except ValueError:
				# Treat as regex without prefix (fallback)
				try:
					compiled = re.compile(self.__unescape_regex(stripped))
				except re.error as e:
					raise ValueError(
						f"Entry {item!r} is neither a valid table ref nor a valid regex: {e}"
					)
				matches = [
					(display, ref)
					for display, ref in available
					if compiled.search(display) or compiled.search(display.split(".")[-1])
				]
				for display, ref in sorted(matches, key=lambda x: x[0]):
					add(display, ref)

		return resolved

	def __unescape_regex(self, pattern: str) -> str:
		"""Best-effort unescape for regex patterns coming from JSON.

		In JSON, backslashes are typically escaped (e.g. "\\d{4}").
		By the time Python receives the parsed JSON value, it is usually "\\d{4}".
		However, callers sometimes pass JSON-escaped regex strings directly.
		This method normalizes common double-backslash sequences.
		"""
		return pattern.replace("\\\\", "\\")

	def __extract_provenance_value(self, prov_re: re.Pattern[str], display_name: str) -> str:
		"""Extract provenance label from a table display name.

		We try the full display name first (e.g. "test_ds.data_2021"), then the bare
		table name (e.g. "data_2021"). If the regex has capture groups, group(1) is used;
		otherwise group(0). Raises if no match.
		"""
		candidates = [display_name, display_name.split(".")[-1]]
		for cand in candidates:
			m = prov_re.search(cand)
			if m:
				return m.group(1) if m.groups() else m.group(0)
		raise ValueError(
			f"provenance_regex did not match table id/name {display_name!r}."
		)

	def __display_name_from_ref(self, table_ref: str) -> str:
		if "." not in table_ref:
			return table_ref.strip().strip('"').replace('""', '"')
		left, right = table_ref.split(".", 1)
		left = left.strip().strip('"').replace('""', '"')
		right = right.strip().strip('"').replace('""', '"')
		return f"{left}.{right}"

	def __get_table_columns(self, table_ref: str) -> list[str]:
		df = self.db_api.execute_query(
			self.user_id, self.chat_id, f"SELECT * FROM {table_ref} LIMIT 0;"
		)
		return list(df.columns)
