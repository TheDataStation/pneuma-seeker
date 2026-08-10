# src/pneuma_seeker/services/db/datasets/manager.py
import csv
import os
from logging import Logger
from pathlib import Path

import duckdb
from pandas import isna, read_csv
from tqdm import tqdm

from pneuma_seeker.models import PermissionKey
from pneuma_seeker.shared.str_processor import clean_column_table_name


class DatasetManager:
    """Manages dataset databases and dataset linking into workspaces."""

    def __init__(self, dataset_db_path: Path, logger: Logger) -> None:
        self.dataset_db_path = Path(dataset_db_path)
        self.logger = logger
        self._pg_registry: dict[str, str] = {}

        os.makedirs(self.dataset_db_path, exist_ok=True)

    def get_dataset_connection(self, dataset_name: str, read_only: bool = True):
        """Returns a DuckDB connection to the dataset DB file (uses .db extension)."""
        os.makedirs(self.dataset_db_path / dataset_name, exist_ok=True)
        dataset_db_file = self.dataset_db_path / dataset_name / f"{dataset_name}.db"
        if read_only and not dataset_db_file.exists():
            temp_con = duckdb.connect(
                database=dataset_db_file.as_posix(), read_only=False
            )
            temp_con.close()
        con = duckdb.connect(database=dataset_db_file.as_posix(), read_only=read_only)
        return con

    def ingest_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata_path: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """
        Stores CSV files inside the dataset's own DuckDB file.
        - table name = cleaned(Path(csv_file).stem)
        - cleans column names
        - only reads file once for ingestion (fast path)
        """
        if (
            os.path.exists(self.dataset_db_path / dataset_name / f"{dataset_name}.db")
            and not overwrite
        ):
            self.logger.warning(
                "Dataset DB already exists for '%s' at %s. Skipping ingestion. Set overwrite=True to force re-ingestion.",
                dataset_name,
                self.dataset_db_path / dataset_name / f"{dataset_name}.db",
            )
            return

        if metadata_path is not None and os.path.exists(metadata_path):
            try:
                metadata = read_csv(metadata_path)
                if (
                    "table_name" in metadata.columns
                    and "description" in metadata.columns
                ):
                    metadata["table_name"] = metadata["table_name"].apply(
                        clean_column_table_name
                    )
                    dest_metadata_path = (
                        self.dataset_db_path / dataset_name / "metadata.csv"
                    )
                    os.makedirs(dest_metadata_path.parent, exist_ok=True)
                    metadata.to_csv(dest_metadata_path, index=False)
            except Exception as e:
                self.logger.warning(
                    "Failed to process metadata CSV at %s: %s", metadata_path, e
                )

        dataset_con = self.get_dataset_connection(dataset_name, read_only=False)
        try:
            dataset_con.begin()
            for table_file_name in tqdm(sorted(os.listdir(dataset_path))):
                if not table_file_name.lower().endswith(".csv"):
                    continue

                file_path = (Path(dataset_path) / table_file_name).as_posix()
                table_stem = Path(table_file_name).stem
                cleaned_table_name = clean_column_table_name(table_stem)

                original_cols = None
                try:
                    with open(file_path, "r") as f:
                        header_line = f.readline().strip()
                        original_cols = list(csv.reader([header_line]))[0]
                except Exception:
                    original_cols = None

                if original_cols:
                    cleaned_cols = self.__dedupe_columns(
                        [clean_column_table_name(c) for c in original_cols]
                    )
                    select_clause = ", ".join(
                        f'"{orig}" AS "{cleaned}"'
                        for orig, cleaned in zip(original_cols, cleaned_cols)
                    )

                    dataset_con.execute(f"""
                        CREATE OR REPLACE TABLE "{cleaned_table_name}" AS
                        SELECT {select_clause}
                        FROM read_csv_auto(
                            '{file_path}',
                            HEADER=TRUE,
                            IGNORE_ERRORS=TRUE,
                            STRICT_MODE=FALSE,
                            NULL_PADDING=TRUE,
                            SAMPLE_SIZE=100_000,
                            PARALLEL=FALSE
                        );
                        """)
                else:
                    dataset_con.execute(f"""
                        CREATE OR REPLACE TABLE "{cleaned_table_name}" AS
                        SELECT * FROM read_csv_auto(
                            '{file_path}',
                            HEADER=TRUE,
                            IGNORE_ERRORS=TRUE,
                            STRICT_MODE=FALSE,
                            NULL_PADDING=TRUE,
                            SAMPLE_SIZE=100_000,
                            PARALLEL=FALSE
                        );
                        """)
            dataset_con.commit()
        except Exception as exception:
            dataset_con.rollback()
            self.logger.error(
                "[DatasetManager] Failed to ingest dataset '%s': %s",
                dataset_name,
                exception,
            )
            raise
        finally:
            dataset_con.close()

    def list_tables(self, dataset_name: str) -> list[str]:
        """Lists table names directly from the dataset's own DB file (no chat/workspace needed)."""
        con = self.get_dataset_connection(dataset_name, read_only=True)
        try:
            return con.execute("SHOW TABLES").fetchdf()["name"].tolist()
        finally:
            con.close()

    def query_table(
        self,
        dataset_name: str,
        table_name: str,
        limit: int,
        offset: int,
        order_by: str | None,
        order_dir: str,
        search: str | None,
    ):
        """
        Queries a table directly from the dataset's own DB file, bypassing the
        per-chat workspace connection used by `link_dataset_tables` — this lets
        dataset browsing work without an active chat session.

        Returns (rows_df, total_count, columns).
        """
        con = self.get_dataset_connection(dataset_name, read_only=True)
        try:
            columns = list(
                con.execute(f'SELECT * FROM "{table_name}" LIMIT 0').fetchdf().columns
            )

            where_sql = ""
            params: list[str] = []
            if search and columns:
                conditions = " OR ".join(
                    f'CAST("{col}" AS VARCHAR) ILIKE ?' for col in columns
                )
                where_sql = f" WHERE {conditions}"
                params = [f"%{search}%"] * len(columns)

            total_count = con.execute(
                f'SELECT COUNT(*) FROM "{table_name}"{where_sql}', params
            ).fetchone()[  # type: ignore
                0
            ]

            order_sql = ""
            if order_by and order_by in columns:
                order_sql = f' ORDER BY "{order_by}" {order_dir.upper()}'

            rows_df = con.execute(
                f'SELECT * FROM "{table_name}"{where_sql}{order_sql} LIMIT {limit} OFFSET {offset}',
                params,
            ).fetchdf()

            return rows_df, int(total_count), columns
        finally:
            con.close()

    def query_sql(
        self,
        dataset_name: str,
        sql: str,
        params: tuple = (),
        limit: int = 500,
    ):
        """
        Runs a single read-only SELECT/WITH...SELECT against exactly one
        dataset's own DB file — the general-query counterpart to query_table
        (which only supports a fixed paginate/sort/search template). This is
        deliberately scoped to one dataset: get_dataset_connection opens only
        that dataset's .db file (no ATTACH of any other dataset), so even
        arbitrary SQL text can't reach data outside it. The connection is
        opened read_only=True, so DuckDB itself rejects any write regardless
        of query text — the statement-shape check below (single statement,
        must start with SELECT/WITH) is a second, independent layer on top of
        that, not the only guard.

        Returns (rows_df, total_count, columns). Raises ValueError for a
        malformed/non-read statement, propagates DuckDB errors otherwise.
        """
        cleaned = self.__require_single_read_statement(sql)
        con = self.get_dataset_connection(dataset_name, read_only=True)
        try:
            total_count = con.execute(
                f"SELECT COUNT(*) FROM ({cleaned}) AS _agentic_catalog_count", params
            ).fetchone()[  # type: ignore
                0
            ]
            rows_df = con.execute(
                f"SELECT * FROM ({cleaned}) AS _agentic_catalog_rows LIMIT {limit}",
                params,
            ).fetchdf()
            columns = list(rows_df.columns)
            return rows_df, int(total_count), columns
        finally:
            con.close()

    @staticmethod
    def __require_single_read_statement(sql: str) -> str:
        stripped = sql.strip()
        if not stripped:
            raise ValueError("sql must be a non-empty string.")
        if stripped.endswith(";"):
            stripped = stripped[:-1].strip()
        if ";" in stripped:
            raise ValueError("sql must be a single SQL statement (no embedded ';').")
        if not stripped.upper().startswith(("SELECT", "WITH")):
            raise ValueError("sql must be a SELECT/WITH...SELECT statement.")
        return stripped

    def get_table_description(self, dataset_name: str, table_name: str) -> str:
        """Returns the description of a table in the dataset."""
        os.makedirs(self.dataset_db_path / dataset_name, exist_ok=True)
        metadata_path = self.dataset_db_path / dataset_name / "metadata.csv"
        if os.path.exists(metadata_path) is False:
            return ""
        try:
            metadata = read_csv(metadata_path)
        except Exception:
            return ""

        if (
            "table_name" not in metadata.columns
            or "description" not in metadata.columns
        ):
            return ""

        table_meta = metadata[metadata["table_name"] == table_name]
        if table_meta.empty:
            return ""
        description = table_meta.iloc[0]["description"]
        if description is None or isna(description):
            return ""
        return str(description)

    def register_postgres_dataset(
        self, dataset_name: str, connection_string: str
    ) -> None:
        """Registers a PostgreSQL-backed dataset by storing its connection string."""
        self._pg_registry[dataset_name] = connection_string

    def is_dataset_accessible(
        self, dataset_name: str, is_admin: bool, group_permissions: dict[str, str]
    ) -> bool:
        """
        Single source of truth for whether a dataset is accessible to a user with
        the given admin status and group permissions. Every code path that reads
        or exposes dataset data — not just the dataset-listing endpoint — must
        call this before touching `dataset_name`.
        """
        dataset_db_file = self.dataset_db_path / dataset_name / f"{dataset_name}.db"
        if not dataset_db_file.exists():
            return False
        if is_admin:
            return True
        permission_key = f"{PermissionKey.DATASET_ACCESS_PREFIX.value}:{dataset_name}"
        return permission_key in group_permissions

    def get_accessible_local_datasets(
        self, is_admin: bool, group_permissions: dict[str, str]
    ) -> list[str]:
        """
        Returns a list of local datasets accessible to the user based on their admin status and group membership.
        """
        return [
            dataset_name
            for dataset_name in os.listdir(self.dataset_db_path)
            if self.is_dataset_accessible(dataset_name, is_admin, group_permissions)
        ]

    def link_dataset_tables(
        self,
        user_id: str,
        chat_id: str,
        dataset_name: str,
        get_ws_connection,
    ) -> None:
        """
        Attach a dataset into the workspace connection under a safe alias.

        - If dataset_name was registered via register_postgres_dataset, attaches
          using DuckDB's postgres extension (READ_ONLY).
        - Otherwise, attaches a local .db file.
        """
        ws_db_con = get_ws_connection(user_id, chat_id)

        alias = clean_column_table_name(dataset_name)
        attached = ws_db_con.execute("PRAGMA database_list").fetchdf()
        if "name" in attached.columns and alias in attached["name"].tolist():
            return

        if dataset_name in self._pg_registry:
            conn_str = self._pg_registry[dataset_name]
            ws_db_con.execute(
                f"ATTACH '{conn_str}' AS \"{alias}\" (TYPE postgres, READ_ONLY)"
            )
        else:
            dataset_db_file = self.dataset_db_path / dataset_name / f"{dataset_name}.db"
            if not dataset_db_file.exists():
                raise FileNotFoundError(
                    f"Dataset DB not found: {dataset_db_file.as_posix()}"
                )
            ws_db_con.execute(
                f"ATTACH DATABASE '{dataset_db_file.as_posix()}' AS \"{alias}\" (READ_ONLY)"
            )

    def __dedupe_columns(self, cols):
        """Deduplicates column names by appending _1, _2, etc. to duplicates."""
        seen = {}
        result = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                result.append(c)
            else:
                seen[c] += 1
                result.append(f"{c}_{seen[c]}")
        return result
