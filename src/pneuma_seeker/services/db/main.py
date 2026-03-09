# services/db/main.py
import csv
import os
from logging import Logger
from pathlib import Path
from uuid import UUID, uuid4

import duckdb
from pandas import DataFrame, isna, read_csv
from tqdm import tqdm

from pneuma_seeker.provenance.graph import ProvenanceGraph, ProvenanceNode
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    Text,
)
from pneuma_seeker.shared.schemas.db.document_type import DocumentType
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.str_processor import clean_column_table_name


class PneumaDB:
    """
    PneumaDB manages:
      - dataset databases (one .db per dataset)
      - per-user workspace databases (one .db per user/chat)
      - external tables (uploaded by users)
      - intermediate & target tables
      - metadata tracking

    Notes / design choices:
    - Each dataset database file uses the `.db` extension.
    - Each workspace uses its own DB file (ws_{user}_{chat}.db).
    """

    def __init__(
        self,
        config: Config,
        logger: Logger,
        dataset_db_path: str | None = None,
        workspace_db_path: str | None = None,
    ):
        self.config = config
        self.logger = logger

        if dataset_db_path:
            self.dataset_db_path = Path(dataset_db_path)
        else:
            self.dataset_db_path = Path(__file__).resolve().parent / "datasets"

        if workspace_db_path:
            self.workspace_db_path = Path(workspace_db_path)
        else:
            self.workspace_db_path = Path(__file__).resolve().parent / "workspaces"

        os.makedirs(self.dataset_db_path, exist_ok=True)
        os.makedirs(self.workspace_db_path, exist_ok=True)

        self._conn_cache: dict[tuple[str, str], duckdb.DuckDBPyConnection] = {}
        self._pg_registry: dict[str, str] = {}

        self.target_tables_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "data_src",
            "target_tables",
        )

    # ------------------------------------------------------------------
    # Dataset Management (one .db per dataset)
    # ------------------------------------------------------------------
    def get_dataset_connection(self, dataset_name: str, read_only: bool = True):
        """Returns a DuckDB connection to the dataset DB file (uses .db extension)."""
        os.makedirs(self.dataset_db_path / dataset_name, exist_ok=True)
        dataset_db_file = self.dataset_db_path / dataset_name / f"{dataset_name}.db"
        # For read-only mode, ensure the file exists first
        if read_only and not dataset_db_file.exists():
            # Create it in read-write mode first
            temp_con = duckdb.connect(
                database=dataset_db_file.as_posix(), read_only=False
            )
            temp_con.close()
        con = duckdb.connect(database=dataset_db_file.as_posix(), read_only=read_only)
        return con

    def ingest_dataset(self, dataset_name: str, dataset_path: str):
        """
        Stores CSV files inside the dataset's own DuckDB file.
        - table name = cleaned(Path(csv_file).stem)
        - cleans column names
        - only reads file once for ingestion (fast path)
        """
        dataset_con = self.get_dataset_connection(dataset_name, read_only=False)
        try:
            dataset_con.begin()
            for table_file_name in tqdm(sorted(os.listdir(dataset_path))):
                if not table_file_name.lower().endswith(".csv"):
                    continue

                file_path = (Path(dataset_path) / table_file_name).as_posix()
                table_stem = Path(table_file_name).stem
                cleaned_table_name = clean_column_table_name(table_stem)

                # Read header only to get original column names (fast)
                original_cols = None
                try:
                    with open(file_path, "r") as f:
                        header_line = f.readline().strip()
                        # Simple CSV header parsing (handles quoted fields)
                        original_cols = list(csv.reader([header_line]))[0]
                except Exception as exception:
                    # Fallback: let DuckDB auto-detect and ingest (still fine)
                    original_cols = None

                if original_cols:
                    cleaned_cols = self.__dedupe_columns(
                        [clean_column_table_name(c) for c in original_cols]
                    )
                    select_clause = ", ".join(
                        f'"{orig}" AS "{cleaned}"'
                        for orig, cleaned in zip(original_cols, cleaned_cols)
                    )

                    dataset_con.execute(
                        f"""
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
                        """
                    )
                else:
                    # If we couldn't get header with pandas, let DuckDB create the table
                    dataset_con.execute(
                        f"""
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
                        """
                    )
            dataset_con.commit()
        except Exception as exception:
            dataset_con.rollback()
            self.logger.error(
                f"[PneumaDB] Failed to ingest dataset '{dataset_name}': {exception}"
            )
            raise exception
        finally:
            dataset_con.close()

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

    # ------------------------------------------------------------------
    # Workspace DB Management (one .db per user/chat)
    # ------------------------------------------------------------------
    def get_ws_db_connection(self, user_id: str, chat_id: str):
        """
        Returns a cached DuckDB connection for the workspace.
        - If connection is new, create it and initialize necessary tables.
        - Caller must not close the returned connection (use close_workspace_connection).
        """
        key = (user_id, chat_id)
        if key in self._conn_cache:
            return self._conn_cache[key]

        ws_db_file = self.__get_workspace_db_file_path(user_id, chat_id)
        os.makedirs(ws_db_file.parent, exist_ok=True)

        ws_db_con = duckdb.connect(database=ws_db_file.as_posix(), read_only=False)
        self.__define_all_persistence_tables(ws_db_con)

        self._conn_cache[key] = ws_db_con
        return ws_db_con

    def __define_all_persistence_tables(self, con: duckdb.DuckDBPyConnection):
        """Defines all necessary tables for state persistence in the workspace DB."""
        try:
            con.begin()
            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS chat_history (
                        chat_history_id     UUID PRIMARY KEY,
                        role                VARCHAR,
                        content             VARCHAR,
                        creation_timestamp  TIMESTAMP DEFAULT now()
                    );
                """
            )

            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS conductor_state (
                        state_id                        UUID PRIMARY KEY,
                        chat_history_id                 UUID,
                        are_target_tables_materialized  BOOLEAN,
                        python_script                   VARCHAR,
                        is_python_script_executed       BOOLEAN,
                        join_paths                      VARCHAR,
                        creation_timestamp              TIMESTAMP DEFAULT now(),
                        FOREIGN KEY (chat_history_id) REFERENCES chat_history(chat_history_id)
                    );
                """
            )

            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS provenance_nodes (
                        state_id            UUID,
                        node_id             UUID PRIMARY KEY,
                        source_retriever    VARCHAR,
                        python_code         VARCHAR,
                        description         VARCHAR,
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id)
                    );
                """
            )

            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id          VARCHAR PRIMARY KEY,
                        retriever_type  VARCHAR,
                        content         VARCHAR,
                        path            VARCHAR,
                        last_node_id    UUID
                    );
                """
            )
            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        doc_id          VARCHAR,
                        metadata_key    VARCHAR,
                        metadata_value  VARCHAR,
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                    );
                """
            )
            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS state_document_roles (
                        state_id    UUID,
                        doc_id      VARCHAR,
                        role        VARCHAR,
                        PRIMARY KEY (state_id, doc_id, role),
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                    );
                """
            )

            con.execute(
                """
                    CREATE TABLE IF NOT EXISTS provenance_edges (
                        state_id        UUID,
                        parent_node_id  UUID,
                        child_node_id   UUID,
                        PRIMARY KEY (state_id, parent_node_id, child_node_id),
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id),
                        FOREIGN KEY (parent_node_id) REFERENCES provenance_nodes(node_id),
                        FOREIGN KEY (child_node_id) REFERENCES provenance_nodes(node_id)
                    );
                """
            )

            con.commit()
        except Exception as e:
            con.rollback()
            self.__log(f"Failed to define persistence tables: {e}")
            raise

    def __get_workspace_db_file_path(self, user_id: str, chat_id: str) -> Path:
        """Returns the Path to the workspace DB file for the given user/chat."""
        user_dir = self.workspace_db_path / user_id
        chat_dir = user_dir / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        return chat_dir / "ws.db"

    def close_workspace_connection(self, user_id: str, chat_id: str):
        """Closes and removes a cached workspace connection if it exists."""
        key = (user_id, chat_id)
        con = self._conn_cache.pop(key, None)
        if con:
            try:
                con.close()
            except Exception:
                pass

    def close_all_connections(self):
        """Closes all cached workspace connections."""
        for con in list(self._conn_cache.values()):
            try:
                con.close()
            except Exception:
                pass
        self._conn_cache.clear()

    # ------------------------------------------------------------------
    # PostgreSQL Dataset Registry
    # ------------------------------------------------------------------
    def register_postgres_dataset(self, dataset_name: str, connection_string: str) -> None:
        """Registers a PostgreSQL-backed dataset by storing its libpq connection string.

        When link_dataset_tables is called for this dataset_name, DuckDB will ATTACH
        via the postgres extension (READ_ONLY) instead of looking for a local .db file.

        Note: Prefer using DuckDB secrets over embedding credentials directly in the
        connection string to avoid accidental credential exposure in error output.
        See: https://duckdb.org/docs/stable/core_extensions/postgres#configuring-via-secrets
        """
        self._pg_registry[dataset_name] = connection_string

    # ------------------------------------------------------------------
    # Dataset DB Linking into Workspace DB
    # ------------------------------------------------------------------
    def link_dataset_tables(self, user_id: str, chat_id: str, dataset_name: str):
        """
        Attach a dataset into the workspace connection under a safe alias.

        - If dataset_name was registered via register_postgres_dataset, attaches
          using DuckDB's postgres extension (READ_ONLY).
        - Otherwise, attaches a local .db file (original behaviour).

        The alias is clean_column_table_name(dataset_name). Idempotent.
        """
        self.__log(f"[PneumaDB] Linking dataset '{dataset_name}' into workspace.")
        ws_db_con = self.get_ws_db_connection(user_id, chat_id)

        alias = clean_column_table_name(dataset_name)

        # Check attached databases (PRAGMA database_list)
        attached = ws_db_con.execute("PRAGMA database_list").fetchdf()
        # `name` column contains aliases; defensive checks
        if "name" in attached.columns and alias in attached["name"].tolist():
            return  # already attached

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
            # Attach read-only
            ws_db_con.execute(
                f"ATTACH DATABASE '{dataset_db_file.as_posix()}' AS \"{alias}\" (READ_ONLY)"
            )

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def execute_query(
        self, user_id: str, chat_id: str, sql: str, sql_params: tuple = ()
    ) -> DataFrame:
        """
        Execute SQL in the context of the workspace DB connection.
        Note: workspace connection is cached so ATTACH persists between calls.
        """
        ws_db_con = self.get_ws_db_connection(user_id, chat_id)
        return ws_db_con.execute(sql, sql_params).fetchdf()

    # ------------------------------------------------------------------
    # Session Persistence
    # ------------------------------------------------------------------
    def register_temporary_df(
        self, user_id: str, chat_id: str, df: DataFrame, table_name: str
    ):
        """Registers a temporary DataFrame in the workspace DB connection.

        Note: DuckDB's `con.register(name, df)` creates a view-like relation that can
        shadow an existing persistent table with the same name. To prevent subtle
        correctness issues (e.g., accidentally replacing a materialized target table
        with a small preview DF), we refuse to register a DF under a name that
        already exists as a BASE TABLE in the workspace schema.
        """

        con = self.get_ws_db_connection(user_id, chat_id)

        existing = con.execute(
            """
            SELECT table_type
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            (table_name,),
        ).fetchall()

        if any(row[0] == "BASE TABLE" for row in existing):
            raise ValueError(
                f"Refusing to register temporary DataFrame as '{table_name}' because a persistent table with that name already exists. "
                "Use a different temporary name (e.g., '<name>__preview' or 'tmp_<name>')."
            )

        try:
            # If a temporary view with this name already exists (e.g., from a prior
            # register call), try to unregister it and re-register.
            if existing:
                try:
                    con.unregister(table_name)
                except Exception:
                    pass
            con.register(table_name, df)
        except Exception as e:
            self.__log(f"Failed to register temporary DataFrame: {e}")
            raise

    def persist_df(
        self,
        user_id: str,
        chat_id: str,
        df: DataFrame,
        table_name: str,
        overwrite_content: bool,
    ):
        """Persists a DataFrame as a table in the workspace DB (skip if exists)."""
        con = self.get_ws_db_connection(user_id, chat_id)
        try:
            con.begin()
            con.register("df", df)
            if overwrite_content:
                con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" AS SELECT * FROM df'
            )
            con.commit()
            con.checkpoint()
        except Exception as e:
            con.rollback()
            self.__log(f"Failed to persist DataFrame: {e}")
        finally:
            try:
                con.unregister("df")
            except Exception:
                pass

    def persist_session(
        self,
        user_id: str,
        chat_id: str,
        new_user_input: str,
        new_system_response: str,
        conductor_state: ConductorState,
        provenance_graph: ProvenanceGraph,
        retrieved_tables: list[AbstractDocument],
        enumerated_tables: list[AbstractDocument],
        web_search_result: AbstractDocument | None = None,
        web_crawl_result: AbstractDocument | None = None,
        join_paths: str | None = None,
    ):
        """
        Persists the chat session.
        """
        con = self.get_ws_db_connection(user_id, chat_id)
        new_user_message_id = uuid4()
        new_system_response_id = uuid4()
        try:
            con.begin()
            con.execute(
                """
                INSERT INTO chat_history (
                    chat_history_id,
                    role,
                    content
                ) VALUES (?, ?, ?);
                """,
                (new_user_message_id, Role.USER.value, new_user_input),
            )
            con.execute(
                """
                INSERT INTO chat_history (
                    chat_history_id,
                    role,
                    content
                ) VALUES (?, ?, ?);
                """,
                (new_system_response_id, Role.ASSISTANT.value, new_system_response),
            )
            con.commit()
            if not self.config.ENABLE_FINE_GRAINED_STATE_CHANGE_TRACKING:
                con.begin()
                con.execute("""DELETE FROM document_metadata;""")
                con.commit()
                con.begin()
                con.execute("""DELETE FROM state_document_roles;""")
                con.commit()
                con.begin()
                con.execute("""DELETE FROM provenance_edges;""")
                con.commit()
                con.begin()
                con.execute("""DELETE FROM documents;""")
                con.commit()
                con.begin()
                con.execute("""DELETE FROM provenance_nodes;""")
                con.commit()
                con.begin()
                con.execute("""DELETE FROM conductor_state;""")
                con.commit()

            con.begin()
            new_state_id = uuid4()
            con.execute(
                """
                INSERT INTO conductor_state (
                    state_id,
                    chat_history_id,
                    are_target_tables_materialized,
                    python_script,
                    is_python_script_executed,
                    join_paths
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    new_state_id,
                    new_system_response_id,
                    conductor_state.is_T_materialized,
                    conductor_state.S,
                    conductor_state.is_S_executed,
                    join_paths,
                ),
            )

            # Insert all nodes first to satisfy FK constraints.
            self.__log(
                f"Persisting provenance graph with {len(provenance_graph.nodes)} nodes..."
            )
            for provenance_node in provenance_graph.nodes.values():
                con.execute(
                    """
                    INSERT INTO provenance_nodes (
                        state_id,
                        node_id,
                        source_retriever,
                        python_code,
                        description
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        new_state_id,
                        provenance_node.id,
                        provenance_node.source_retriever.value,
                        provenance_node.python_code,
                        provenance_node.description,
                    ),
                )

            # Insert edges after all nodes are present.
            for provenance_node in provenance_graph.nodes.values():
                for child_node in provenance_node.children:
                    con.execute(
                        """
                        INSERT INTO provenance_edges (
                            state_id,
                            parent_node_id,
                            child_node_id
                        ) VALUES (?, ?, ?)
                        ON CONFLICT (state_id, parent_node_id, child_node_id) DO NOTHING;
                        """,
                        (
                            new_state_id,
                            provenance_node.id,
                            child_node.id,
                        ),
                    )

                for parent_node in provenance_node.parents:
                    con.execute(
                        """
                        INSERT INTO provenance_edges (
                            state_id,
                            parent_node_id,
                            child_node_id
                        ) VALUES (?, ?, ?)
                        ON CONFLICT (state_id, parent_node_id, child_node_id) DO NOTHING;
                        """,
                        (
                            new_state_id,
                            parent_node.id,
                            provenance_node.id,
                        ),
                    )

            self.__log(
                f"=> Persisting {len(conductor_state.T)} target tables, {len(retrieved_tables)} retrieved tables, and {len(enumerated_tables)} enumerated tables..."
            )
            for i in conductor_state.T.values():
                self.__insert_document(
                    con,
                    new_state_id,
                    i,
                    DocumentType.TARGET_TABLE.value,
                )
            for i in retrieved_tables:
                self.__insert_document(
                    con,
                    new_state_id,
                    i,
                    DocumentType.RETRIEVED_TABLE.value,
                )
            for e in enumerated_tables:
                self.__insert_document(
                    con,
                    new_state_id,
                    e,
                    DocumentType.ENUMERATED_TABLE.value,
                )
            if web_search_result:
                self.__insert_document(
                    con,
                    new_state_id,
                    web_search_result,
                    DocumentType.WEB_SEARCH_RESULT.value,
                )
            if web_crawl_result:
                self.__insert_document(
                    con,
                    new_state_id,
                    web_crawl_result,
                    DocumentType.WEB_CRAWL_RESULT.value,
                )

            con.commit()
            con.checkpoint()
        except Exception as e:
            con.rollback()
            self.__log(f"Failed to persist session: {e}")

    def __insert_document(
        self,
        con: duckdb.DuckDBPyConnection,
        state_id: UUID | None,
        document: AbstractDocument,
        role: str,
    ):
        """Inserts a document and its metadata into the workspace DB."""
        document_content = document.content
        last_node_id = self.__get_document_last_node_id(document)
        if isinstance(document_content, DataFrame):
            if "dataset_name" in document.metadata:
                dataset_name = document.metadata["dataset_name"]
                document_content = f"{dataset_name}.{document.doc_id}"
            else:
                document_content = document.doc_id

        # doc_id is a PRIMARY KEY, so in fine-grained tracking we must support reusing
        # the same doc_id across states. Upsert keeps the latest document representation.
        con.execute(
            """
            INSERT INTO documents (
                doc_id,
                retriever_type,
                content,
                path,
                last_node_id
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (doc_id) DO UPDATE SET
                retriever_type = excluded.retriever_type,
                content = excluded.content,
                path = excluded.path,
                last_node_id = excluded.last_node_id;
            """,
            (
                document.doc_id,
                document.retriever_type.value,
                document_content,
                document.path,
                last_node_id,
            ),
        )

        # Keep metadata consistent for the doc_id (avoid duplicates).
        con.execute(
            """
            DELETE FROM document_metadata
            WHERE doc_id = ?;
            """,
            (document.doc_id,),
        )
        for meta_key, meta_value in document.metadata.items():
            con.execute(
                """
                INSERT INTO document_metadata (
                    doc_id,
                    metadata_key,
                    metadata_value
                ) VALUES (?, ?, ?);
                """,
                (document.doc_id, meta_key, meta_value),
            )

        if state_id is not None:
            con.execute(
                """
                INSERT INTO state_document_roles (
                    state_id,
                    doc_id,
                    role
                ) VALUES (?, ?, ?)
                ON CONFLICT (state_id, doc_id, role) DO NOTHING
                """,
                (state_id, document.doc_id, role),
            )

    def __get_document_last_node_id(self, doc: AbstractDocument) -> str | None:
        """Helper to get the last_node_id for a document, handling potential NaN or missing values."""
        last_node_id = doc.last_node_id
        if last_node_id is not None:
            try:
                if isna(last_node_id):
                    last_node_id = None
            except Exception:
                pass
        if isinstance(last_node_id, str) and last_node_id.strip() in {
            "",
            "<NA>",
            "NA",
            "N/A",
            "nan",
            "NaN",
            "None",
            "null",
        }:
            last_node_id = None
        return last_node_id

    def load_session(
        self,
        user_id: str,
        chat_id: str,
    ) -> tuple[
        list[LLMMessage],
        ConductorState,
        ProvenanceGraph,
        list[AbstractDocument],
        list[AbstractDocument],
        AbstractDocument | None,
        AbstractDocument | None,
        str | None,
    ]:
        """
        Loads the latest chat session from the chat_session table.
        If no session is found, returns empty structures.
        """
        con = self.get_ws_db_connection(user_id, chat_id)

        row = con.execute(
            """
            SELECT state_id, are_target_tables_materialized, python_script, is_python_script_executed, join_paths
            FROM conductor_state
            ORDER BY creation_timestamp DESC
            LIMIT 1
            """
        ).fetchone()

        if not row:
            return (
                [],
                ConductorState(),
                ProvenanceGraph(self.logger),
                [],
                [],
                None,
                None,
                None,
            )

        state_id = row[0]  # Already a UUID object from DuckDB
        conductor_state = ConductorState()
        conductor_state.is_T_materialized = row[1]
        conductor_state.S = row[2]
        conductor_state.is_S_executed = row[3]
        join_paths = row[4]

        chat_history_rows = con.execute(
            """
            SELECT role, content
            FROM chat_history
            ORDER BY creation_timestamp ASC
            """
        ).fetchdf()
        chat_history: list[LLMMessage] = []
        for _, chat_row in chat_history_rows.iterrows():
            chat_history.append(
                LLMMessage(role=chat_row["role"], content=chat_row["content"])
            )

        provenance_graph = ProvenanceGraph(self.logger, create_default_root=False)
        node_rows = con.execute(
            """
            SELECT node_id, source_retriever, python_code, description
            FROM provenance_nodes
            WHERE state_id = ?
            """,
            (state_id,),
        ).fetchdf()
        nodes_dict: dict[str, ProvenanceNode] = {}
        for _, node_row in node_rows.iterrows():
            node = ProvenanceNode(
                source_retriever=RetrieverType(node_row["source_retriever"]),
                python_code=node_row["python_code"],
                description=node_row["description"],
            )
            node.id = str(node_row["node_id"])
            nodes_dict[node.id] = node
            provenance_graph.add_node(node)

        edge_rows = con.execute(
            """
            SELECT parent_node_id, child_node_id
            FROM provenance_edges
            WHERE state_id = ?
            """,
            (state_id,),
        ).fetchdf()
        for _, edge_row in edge_rows.iterrows():
            parent_id = str(edge_row["parent_node_id"])
            child_id = str(edge_row["child_node_id"])
            if parent_id in nodes_dict and child_id in nodes_dict:
                parent_node = nodes_dict[parent_id]
                child_node = nodes_dict[child_id]
                parent_node.add_child(child_node)

        conductor_state.T = {
            i.doc_id: i
            for i in self.__load_documents_by_role(
                user_id,
                chat_id,
                con,
                state_id,
                DocumentType.TARGET_TABLE.value,
            )
        }

        retrieved_tables = self.__load_documents_by_role(
            user_id,
            chat_id,
            con,
            state_id,
            DocumentType.RETRIEVED_TABLE.value,
        )
        enumerated_tables = self.__load_documents_by_role(
            user_id,
            chat_id,
            con,
            state_id,
            DocumentType.ENUMERATED_TABLE.value,
        )
        web_search_result = None
        web_crawl_result = None
        web_search_docs = self.__load_documents_by_role(
            user_id,
            chat_id,
            con,
            state_id,
            DocumentType.WEB_SEARCH_RESULT.value,
        )
        if web_search_docs:
            web_search_result = web_search_docs[0]
        web_crawl_docs = self.__load_documents_by_role(
            user_id,
            chat_id,
            con,
            state_id,
            DocumentType.WEB_CRAWL_RESULT.value,
        )
        if web_crawl_docs:
            web_crawl_result = web_crawl_docs[0]

        self.__log(
            f"=> Loaded session with {len(chat_history)} chat messages, {len(provenance_graph.nodes)} provenance nodes, {len(conductor_state.T)} target tables, {len(retrieved_tables)} retrieved tables, {len(enumerated_tables)} enumerated tables, {1 if web_search_result else 0} web search results, and {1 if web_crawl_result else 0} web crawl results."
        )

        return (
            chat_history,
            conductor_state,
            provenance_graph,
            retrieved_tables,
            enumerated_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
        )

    def __load_documents_by_role(
        self,
        user_id: str,
        chat_id: str,
        con: duckdb.DuckDBPyConnection,
        state_id: UUID,
        role: str,
    ) -> list[AbstractDocument]:
        """Loads documents for the given state_id and role."""
        doc_rows = con.execute(
            """
            SELECT d.doc_id, d.retriever_type, d.content, d.path, d.last_node_id
            FROM documents d
            JOIN state_document_roles sdr ON d.doc_id = sdr.doc_id
            WHERE sdr.state_id = ? AND sdr.role = ?
            """,
            (state_id, role),
        ).fetchdf()

        documents: list[AbstractDocument] = []
        for _, doc_row in doc_rows.iterrows():
            metadata_rows = con.execute(
                """
                SELECT metadata_key, metadata_value
                FROM document_metadata
                WHERE doc_id = ?
                """,
                (doc_row["doc_id"],),
            ).fetchdf()
            metadata: dict[str, str] = {}
            for _, meta_row in metadata_rows.iterrows():
                metadata[meta_row["metadata_key"]] = meta_row["metadata_value"]

            retriever_type_value = doc_row["retriever_type"]
            retriever_type = (
                retriever_type_value
                if isinstance(retriever_type_value, RetrieverType)
                else RetrieverType(retriever_type_value)
            )

            last_node_id_value = doc_row["last_node_id"]
            last_node_id = (
                None if last_node_id_value is None else str(last_node_id_value)
            )

            if (
                retriever_type == RetrieverType.PNEUMA_RETRIEVER
                or retriever_type == RetrieverType.MATERIALIZER
                or retriever_type == RetrieverType.CONDUCTOR
                or retriever_type == RetrieverType.ENUMERATOR
                or retriever_type == RetrieverType.USER
            ):
                dataset_name = self.config.DATA_SOURCES[0]
                if "dataset_name" in metadata:
                    dataset_name = metadata["dataset_name"]

                self.link_dataset_tables(
                    user_id,
                    chat_id,
                    dataset_name,
                )

                if (
                    retriever_type == RetrieverType.MATERIALIZER
                    or retriever_type == RetrieverType.CONDUCTOR
                ):
                    select_query = (
                        f"""SELECT * FROM "{doc_row['doc_id']}";"""
                    )
                else:
                    select_query = (
                        f"""SELECT * FROM "{dataset_name}"."{doc_row['doc_id']}";"""
                    )
                content = self.execute_query(
                    user_id,
                    chat_id,
                    select_query,
                )
                document = Table(
                    doc_id=doc_row["doc_id"],
                    retriever_type=retriever_type,
                    content=content,
                    metadata=metadata,
                    path=doc_row["path"],
                    last_node_id=last_node_id,
                )
            else:
                document = Text(
                    doc_id=doc_row["doc_id"],
                    retriever_type=retriever_type,
                    content=doc_row["content"],
                    metadata=metadata,
                    path=doc_row["path"],
                    last_node_id=last_node_id,
                )
            documents.append(document)

        return documents

    def __log(self, message: str):
        """Helper logging method."""
        self.logger.info(f"[PneumaDB] {message}")
