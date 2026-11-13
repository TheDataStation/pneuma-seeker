from logging import Logger
from pathlib import Path

import duckdb

from pneuma_seeker.models import PermissionKey
from pneuma_seeker.services.db.datasets import uploads
from pneuma_seeker.services.db.datasets.connectors import (
    CONNECTOR_TYPES,
    DatasetConnector,
    DatasetUnavailable,
    build,
    quote_ident,
)
from pneuma_seeker.services.db.datasets.registry import (
    LOCAL_TYPE,
    DatasetRegistry,
    DatasetSource,
    check_name,
)
from pneuma_seeker.shared.str_processor import clean_column_table_name


class DatasetManager:
    """
    Manages datasets: each is registered with a connector (uploaded Parquet
    tables, a folder of files, a database file or a Postgres database) that
    attaches it into DuckDB as <dataset>."<table>" views.
    """

    def __init__(self, dataset_db_path: Path, logger: Logger, registry: DatasetRegistry) -> None:
        self.logger = logger
        self.registry = registry
        self.dataset_db_path = Path(dataset_db_path)
        # (connection id, alias) -> fingerprint of the data the attached views show
        self._attached: dict[tuple[int, str], str] = {}

    @property
    def dataset_db_path(self) -> Path:
        """Gets the root folder of local (uploaded) datasets."""
        return self._root

    @dataset_db_path.setter
    def dataset_db_path(self, value: str | Path) -> None:
        """Sets the root folder of local datasets."""
        self._root = Path(value)
        self._root.mkdir(parents=True, exist_ok=True)
        self.registry.root = self._root

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def list_datasets(self) -> list[DatasetSource]:
        """Lists every registered dataset."""
        return self.registry.list_sources()

    def get_source(self, dataset_name: str) -> DatasetSource | None:
        """Returns a dataset's registration, or None."""
        return self.registry.get(dataset_name)

    def connector(self, dataset_name: str) -> DatasetConnector:
        """Returns the connector of a dataset. Raises DatasetUnavailable if it isn't registered."""
        source = self.registry.get(dataset_name)
        if source is None:
            raise DatasetUnavailable(f"No dataset named {dataset_name!r}.")
        return self._build(source)

    def is_local(self, source: DatasetSource) -> bool:
        """Checks whether a dataset is stored under the datasets root (so uploads may change it)."""
        if source.connector_type != LOCAL_TYPE:
            return False
        path = Path(source.config.get("path", ""))
        resolved = (path if path.is_absolute() else self._root / path).resolve()
        return resolved.is_relative_to(self._root.resolve()) and resolved != self._root.resolve()

    def test_source(self, connector_type: str, config: dict) -> tuple[bool, str | None, int]:
        """Checks whether a source can be read: (reachable, error, table count). Raises ValueError for a bad config."""
        return build(connector_type, config).health_check()

    def add_source(self, name: str, connector_type: str, config: dict, created_by: str | None) -> DatasetSource:
        """Registers a dataset over an existing source after checking it can be read. Raises ValueError."""
        check_name(name)
        if connector_type not in CONNECTOR_TYPES:
            raise ValueError(f"Unknown connector type: {connector_type!r}")
        connector = build(connector_type, config)
        ok, error, _ = connector.health_check()
        if not ok:
            raise ValueError(f"Couldn't read the source: {error}")
        return self.registry.add(DatasetSource(name, connector_type, connector.stored_config(), created_by))

    def update_source(self, name: str, config: dict) -> DatasetSource:
        """Replaces a dataset's connector config, keeping stored secrets left blank. Raises ValueError."""
        source = self.registry.get(name)
        if source is None:
            raise ValueError(f"No dataset named {name!r}.")
        cls = CONNECTOR_TYPES[source.connector_type]
        merged = {
            **config,
            **{f: source.config.get(f, "") for f in cls.SECRET_FIELDS if not config.get(f)},
        }
        connector = build(source.connector_type, merged)
        ok, error, _ = connector.health_check()
        if not ok:
            raise ValueError(f"Couldn't read the source: {error}")
        return self.registry.update_config(name, connector.stored_config())

    def remove_dataset(self, dataset_name: str) -> None:
        """Unregisters a dataset, deleting its files if it's local."""
        source = self.registry.get(dataset_name)
        if source is None:
            raise ValueError(f"No dataset named {dataset_name!r}.")
        if self.is_local(source):
            uploads.delete_dataset_files(self._root, dataset_name)
        self.registry.remove(dataset_name)

    # ------------------------------------------------------------------
    # Local data
    # ------------------------------------------------------------------
    def add_file(
        self,
        dataset_name: str,
        file: Path,
        file_name: str,
        *,
        create: bool = False,
        replace: bool = False,
        created_by: str | None = None,
    ) -> uploads.UploadResult:
        """Adds an uploaded file's tables to a local dataset, registering it if created. Raises UploadRefused."""
        source = self.registry.get(dataset_name)
        if source is not None and not self.is_local(source):
            raise uploads.UploadRefused("invalid", f"{dataset_name!r} is a connected source; files can't be added to it.")
        if source is None and not create:
            raise uploads.UploadRefused("not_found", f"No dataset named {dataset_name!r}.")
        result = uploads.add_file(self._root, dataset_name, file, file_name, create=create, replace=replace)
        if result.created_dataset:
            self.registry.register_local(dataset_name, created_by)
        return result

    def delete_table(self, dataset_name: str, table_name: str) -> None:
        """Deletes a table of a local dataset. Raises UploadRefused."""
        source = self.registry.get(dataset_name)
        if source is None or not self.is_local(source):
            raise uploads.UploadRefused("invalid", f"{dataset_name!r} isn't a local dataset.")
        uploads.delete_table(self._root, dataset_name, table_name)

    def ingest_dataset(self, dataset_name: str, dataset_path: str, overwrite: bool = True) -> None:
        """Stores every CSV file in `dataset_path` as a table of a local dataset (replacing it if `overwrite`)."""
        if (self._root / dataset_name).is_dir():
            if not overwrite:
                self.logger.warning(f"[DatasetManager] Dataset '{dataset_name}' exists; skipping ingestion.")
                return
            uploads.delete_dataset_files(self._root, dataset_name)
        for file in sorted(Path(dataset_path).iterdir()):
            if file.is_file() and file.suffix.lower() == ".csv":
                uploads.add_file(self._root, dataset_name, file, file.name, create=True, replace=True)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get_dataset_connection(
        self, dataset_name: str, tables: set[str] | None = None
    ) -> duckdb.DuckDBPyConnection:
        """
        Returns a new connection with only this dataset attached (as the current
        database), or only its `tables`, and file access limited to its own files.
        Raises DatasetUnavailable.
        """
        connector = self.connector(dataset_name)
        alias = clean_column_table_name(dataset_name)
        con = duckdb.connect()
        try:
            connector.attach(con, alias, tables)
            con.execute(f"USE {quote_ident(alias)}")
            connector.lockdown(con)
        except Exception:
            con.close()
            raise
        return con

    def list_tables(self, dataset_name: str) -> list[str]:
        """Lists a dataset's table names ([] if it's missing or unreadable)."""
        try:
            return self.connector(dataset_name).table_names()
        except DatasetUnavailable as exc:
            self.logger.warning(f"[DatasetManager] {exc}")
            return []

    def dataset_fingerprint(self, dataset_name: str) -> str:
        """Returns a fingerprint that changes when the dataset's data may have."""
        return self.connector(dataset_name).fingerprint()

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
        Queries one table of a dataset, without a chat session, for browsing.
        Returns (rows_df, total_count, columns).
        """
        con = self.get_dataset_connection(dataset_name, tables={table_name})
        try:
            table = quote_ident(table_name)
            columns = list(con.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf().columns)
            where_sql = ""
            params: list[str] = []
            if search and columns:
                where_sql = " WHERE " + " OR ".join(f"CAST({quote_ident(c)} AS VARCHAR) ILIKE ?" for c in columns)
                params = [f"%{search}%"] * len(columns)
            (total_count,) = con.execute(f"SELECT COUNT(*) FROM {table}{where_sql}", params).fetchone()  # type: ignore[misc]
            order_sql = ""
            if order_by and order_by in columns:
                order_sql = f" ORDER BY {quote_ident(order_by)} {'DESC' if order_dir.lower() == 'desc' else 'ASC'}"
            rows_df = con.execute(
                f"SELECT * FROM {table}{where_sql}{order_sql} LIMIT {int(limit)} OFFSET {int(offset)}", params
            ).fetchdf()
            return rows_df, int(total_count), columns
        finally:
            con.close()

    def query_sql(self, dataset_name: str, sql: str, params: tuple = (), limit: int = 500):
        """
        Runs a single SELECT/WITH statement against one dataset, without a chat
        session. Only that dataset is attached and its connection can't read other
        files, so the statement can't reach data outside it. Returns (rows_df,
        total_count, columns). Raises ValueError for a malformed or non-read statement.
        """
        cleaned = self.__require_single_read_statement(sql)
        con = self.get_dataset_connection(dataset_name)
        try:
            (total_count,) = con.execute(f"SELECT COUNT(*) FROM ({cleaned}) AS _count", params).fetchone()  # type: ignore[misc]
            rows_df = con.execute(f"SELECT * FROM ({cleaned}) AS _rows LIMIT {int(limit)}", params).fetchdf()
            return rows_df, int(total_count), list(rows_df.columns)
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

    # ------------------------------------------------------------------
    # Access and workspaces
    # ------------------------------------------------------------------
    def is_dataset_accessible(self, dataset_name: str, is_admin: bool, group_permissions: dict[str, str]) -> bool:
        """
        Checks whether a dataset exists and is accessible to a user with the given
        admin status and group permissions. Every code path that reads or exposes
        dataset data must call this first.
        """
        if self.registry.get(dataset_name) is None:
            return False
        if is_admin:
            return True
        return f"{PermissionKey.DATASET_ACCESS_PREFIX.value}:{dataset_name}" in group_permissions

    def get_accessible_datasets(self, is_admin: bool, group_permissions: dict[str, str]) -> list[str]:
        """Returns the datasets accessible to a user with the given admin status and group permissions."""
        prefix = PermissionKey.DATASET_ACCESS_PREFIX.value
        return [
            s.name
            for s in self.registry.list_sources()
            if is_admin or f"{prefix}:{s.name}" in group_permissions
        ]

    def link_dataset_tables(self, user_id: str, chat_id: str, dataset_name: str, get_ws_connection) -> None:
        """
        Attaches a dataset into the chat's workspace connection under its alias,
        re-attaching it if its data changed since. Raises DatasetUnavailable.
        """
        ws_db_con = get_ws_connection(user_id, chat_id)
        connector = self.connector(dataset_name)
        alias = clean_column_table_name(dataset_name)
        key = (id(ws_db_con), alias)
        fingerprint = connector.fingerprint()
        attached = {row[0] for row in ws_db_con.execute("SELECT database_name FROM duckdb_databases()").fetchall()}
        if alias in attached:
            if self._attached.get(key) == fingerprint:
                return
            connector.detach(ws_db_con, alias)
        connector.attach(ws_db_con, alias)
        self._attached[key] = fingerprint

    def _build(self, source: DatasetSource) -> DatasetConnector:
        config = dict(source.config)
        if source.connector_type == LOCAL_TYPE and not Path(config.get("path", "")).is_absolute():
            config["path"] = (self._root / config.get("path", source.name)).as_posix()
        return build(source.connector_type, config)

