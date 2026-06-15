import datetime
import os
from logging import Logger
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import duckdb
from pandas import DataFrame, isna

from pneuma_seeker.provenance.graph import ProvenanceGraph, ProvenanceNode
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.db.datasets.manager import DatasetManager
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


class WorkspaceManager:
    """Manages per-user workspace DBs and session persistence."""

    def __init__(
        self,
        workspace_db_path: Path,
        config: Config,
        logger: Logger,
        dataset_manager: DatasetManager,
    ) -> None:
        self.workspace_db_path = Path(workspace_db_path)
        self.config = config
        self.logger = logger
        self.dataset_manager = dataset_manager

        os.makedirs(self.workspace_db_path, exist_ok=True)

        self._conn_cache: dict[tuple[str, str], duckdb.DuckDBPyConnection] = {}

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
            con.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        chat_history_id     UUID PRIMARY KEY,
                        role                VARCHAR,
                        content             VARCHAR,
                        creation_timestamp  TIMESTAMP WITH TIME ZONE DEFAULT now()
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS conductor_state (
                        state_id                        UUID PRIMARY KEY,
                        chat_history_id                 UUID,
                        are_target_tables_materialized  BOOLEAN,
                        python_script                   VARCHAR,
                        is_python_script_executed       BOOLEAN,
                        join_paths                      VARCHAR,
                        creation_timestamp              TIMESTAMP WITH TIME ZONE DEFAULT now(),
                        FOREIGN KEY (chat_history_id) REFERENCES chat_history(chat_history_id)
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS provenance_nodes (
                        state_id            UUID,
                        node_id             UUID,
                        source_retriever    VARCHAR,
                        python_code         VARCHAR,
                        description         VARCHAR,
                        PRIMARY KEY (state_id, node_id),
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id)
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        state_id        UUID,
                        doc_id          VARCHAR,
                        retriever_type  VARCHAR,
                        content         VARCHAR,
                        path            VARCHAR,
                        last_node_id    UUID,
                        PRIMARY KEY (state_id, doc_id),
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id)
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        state_id        UUID,
                        doc_id          VARCHAR,
                        metadata_key    VARCHAR,
                        metadata_value  VARCHAR,
                        FOREIGN KEY (state_id, doc_id) REFERENCES documents(state_id, doc_id)
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS state_document_roles (
                        state_id    UUID,
                        doc_id      VARCHAR,
                        role        VARCHAR,
                        PRIMARY KEY (state_id, doc_id, role),
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id),
                        FOREIGN KEY (state_id, doc_id) REFERENCES documents(state_id, doc_id)
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS provenance_edges (
                        state_id        UUID,
                        parent_node_id  UUID,
                        child_node_id   UUID,
                        PRIMARY KEY (state_id, parent_node_id, child_node_id),
                        FOREIGN KEY (state_id) REFERENCES conductor_state(state_id),
                        FOREIGN KEY (state_id, parent_node_id) REFERENCES provenance_nodes(state_id, node_id),
                        FOREIGN KEY (state_id, child_node_id) REFERENCES provenance_nodes(state_id, node_id)
                    );
                """)

            con.execute("""
                    CREATE TABLE IF NOT EXISTS session_metadata (
                        dataset_name  VARCHAR NOT NULL
                    );
                """)

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

    def execute_query(
        self, user_id: str, chat_id: str, sql: str, sql_params: tuple = ()
    ) -> DataFrame:
        """
        Executes SQL in the context of the workspace DB connection.
        Note: workspace connection is cached so ATTACH persists between calls.
        """
        ws_db_con = self.get_ws_db_connection(user_id, chat_id)
        return ws_db_con.execute(sql, sql_params).fetchdf()

    def register_temporary_df(
        self, user_id: str, chat_id: str, df: DataFrame, table_name: str
    ):
        """Registers a temporary DataFrame in the workspace DB connection."""
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
        dataset_name: str,
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
        """Persists the chat session."""
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
                INSERT INTO session_metadata (dataset_name)
                SELECT ? WHERE NOT EXISTS (SELECT 1 FROM session_metadata);
                """,
                (dataset_name,),
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

            self.__log(
                f"Persisting provenance graph with {len(provenance_graph.nodes)} nodes..."
            )

            if provenance_graph.nodes:
                node_data = [
                    (
                        new_state_id,
                        node.id,
                        node.source_retriever.value,
                        node.python_code,
                        node.description,
                    )
                    for node in provenance_graph.nodes.values()
                ]

                con.executemany(
                    """
                    INSERT INTO provenance_nodes (
                        state_id,
                        node_id,
                        source_retriever,
                        python_code,
                        description
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    node_data,
                )

            edge_set = set()
            for provenance_node in provenance_graph.nodes.values():
                for child_node in provenance_node.children:
                    edge_set.add((new_state_id, provenance_node.id, child_node.id))
                for parent_node in provenance_node.parents:
                    edge_set.add((new_state_id, parent_node.id, provenance_node.id))

            if edge_set:
                edge_data = list(edge_set)
                con.executemany(
                    """
                    INSERT INTO provenance_edges (
                        state_id,
                        parent_node_id,
                        child_node_id
                    ) VALUES (?, ?, ?);
                    """,
                    edge_data,
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
        except Exception as e:
            con.rollback()
            self.__log(f"Failed to persist session: {e}")

    def load_chat_history(self, user_id: str, chat_id: str) -> list[LLMMessage]:
        """Loads the persisted chat messages for a workspace in chronological order."""
        con = self.get_ws_db_connection(user_id, chat_id)
        rows = con.execute("""
            SELECT role, content
            FROM chat_history
            ORDER BY creation_timestamp ASC
            """).fetchdf()
        return [
            LLMMessage(role=row["role"], content=row["content"])
            for _, row in rows.iterrows()
        ]

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

        con.execute(
            """
            INSERT INTO documents (
                state_id,
                doc_id,
                retriever_type,
                content,
                path,
                last_node_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (state_id, doc_id) DO UPDATE SET
                retriever_type = excluded.retriever_type,
                content = excluded.content,
                path = excluded.path,
                last_node_id = excluded.last_node_id;
            """,
            (
                state_id,
                document.doc_id,
                document.retriever_type.value,
                document_content,
                document.path,
                last_node_id,
            ),
        )

        con.execute(
            """
            DELETE FROM document_metadata
            WHERE state_id = ? AND doc_id = ?;
            """,
            (state_id, document.doc_id),
        )
        for meta_key, meta_value in document.metadata.items():
            con.execute(
                """
                INSERT INTO document_metadata (
                    state_id,
                    doc_id,
                    metadata_key,
                    metadata_value
                ) VALUES (?, ?, ?, ?);
                """,
                (state_id, document.doc_id, meta_key, meta_value),
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
        """Helper to get the last_node_id for a document."""
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
        str | None,
    ]:
        """
        Loads the latest chat session from the chat_session table.
        If no session is found, returns empty structures.
        """
        con = self.get_ws_db_connection(user_id, chat_id)

        row = con.execute("""
            SELECT state_id, are_target_tables_materialized, python_script, is_python_script_executed, join_paths
            FROM conductor_state
            ORDER BY creation_timestamp DESC
            LIMIT 1
            """).fetchone()

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
                None,
            )

        state_id = row[0]
        conductor_state = ConductorState()
        conductor_state.is_T_materialized = row[1]
        conductor_state.S = row[2]
        conductor_state.is_S_executed = row[3]
        join_paths = row[4]

        chat_history = self.load_chat_history(user_id, chat_id)

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

        meta_row = con.execute(
            "SELECT dataset_name FROM session_metadata LIMIT 1"
        ).fetchone()
        dataset_name: str | None = meta_row[0] if meta_row else None

        retrieved_tables: list[AbstractDocument] = []
        enumerated_tables: list[AbstractDocument] = []
        web_search_result = None
        web_crawl_result = None
        if dataset_name is not None:
            conductor_state.T = {
                i.doc_id: i
                for i in self.__load_documents_by_role(
                    user_id,
                    chat_id,
                    dataset_name,
                    con,
                    state_id,
                    DocumentType.TARGET_TABLE.value,
                )
            }

            retrieved_tables = self.__load_documents_by_role(
                user_id,
                chat_id,
                dataset_name,
                con,
                state_id,
                DocumentType.RETRIEVED_TABLE.value,
            )
            enumerated_tables = self.__load_documents_by_role(
                user_id,
                chat_id,
                dataset_name,
                con,
                state_id,
                DocumentType.ENUMERATED_TABLE.value,
            )
            web_search_docs = self.__load_documents_by_role(
                user_id,
                chat_id,
                dataset_name,
                con,
                state_id,
                DocumentType.WEB_SEARCH_RESULT.value,
            )
            web_crawl_docs = self.__load_documents_by_role(
                user_id,
                chat_id,
                dataset_name,
                con,
                state_id,
                DocumentType.WEB_CRAWL_RESULT.value,
            )

            if web_search_docs:
                web_search_result = web_search_docs[0]
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
            dataset_name,
        )

    def get_user_chat_sessions(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """
        Scans the user directory for active chat databases, inspects their
        metadata from the chat_history table, and returns a paginated list
        ordered by the most recent activity.
        """
        user_dir = self.workspace_db_path / user_id
        if not user_dir.exists() or not user_dir.is_dir():
            return {"chats": [], "has_more": False, "next_offset": None}

        all_sessions = []

        # Iterate over all chat_id directories for the given user
        for chat_dir in user_dir.iterdir():
            if chat_dir.is_dir():
                db_file = chat_dir / "ws.db"
                if db_file.exists():
                    chat_id = chat_dir.name
                    try:
                        # Open connection transiently to read metadata
                        con = None
                        try:
                            con = duckdb.connect(database=db_file.as_posix())

                            # Fetch the first message content for the title and the max timestamp for activity
                            query = """
                                SELECT 
                                    (SELECT content FROM chat_history ORDER BY creation_timestamp ASC LIMIT 1) as first_msg,
                                    (SELECT MAX(creation_timestamp) FROM chat_history) as last_active
                                FROM chat_history 
                                LIMIT 1;
                            """
                            res = con.execute(query).fetchone()
                        finally:
                            if con:
                                con.close()

                        if res and res[0] is not None:
                            content = res[0]
                            # Mimic the frontend title generation logic
                            title = (
                                f"{content[:20]}..." if len(content) > 20 else content
                            )

                            # Standardize timestamp to ISO 8601 string format
                            last_active_dt = res[1]
                            last_active_str = (
                                last_active_dt.astimezone(
                                    datetime.timezone.utc
                                ).strftime("%Y-%m-%dT%H:%M:%S.%f")
                                + "Z"
                                if hasattr(last_active_dt, "isoformat")
                                else str(last_active_dt)
                            )

                            all_sessions.append(
                                {
                                    "id": chat_id,
                                    "title": title,
                                    "lastActive": last_active_str,
                                    "_sort_ts": last_active_dt,  # Keep datetime reference for sorting
                                }
                            )
                    except Exception as e:
                        self.__log(
                            f"Failed to extract session metadata from {db_file}: {e}"
                        )
                        continue

        # Sort all sessions descending by their last active timestamp
        all_sessions.sort(key=lambda x: x["_sort_ts"], reverse=True)

        # Clean up internal sorting helpers before slicing
        for session in all_sessions:
            session.pop("_sort_ts", None)

        # Apply pagination slicing
        sliced_sessions = all_sessions[offset : offset + limit]
        has_more = len(all_sessions) > (offset + limit)
        next_offset = offset + limit if has_more else None

        return {
            "chats": sliced_sessions,
            "has_more": has_more,
            "next_offset": next_offset,
        }

    def search_chat_sessions(
        self, user_id: str, query: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """
        Searches all chat sessions for the given user where any message content
        matches the query using ILIKE, sorted by most recent activity descending.
        """
        user_dir = self.workspace_db_path / user_id
        if not user_dir.exists() or not user_dir.is_dir():
            return {"chats": [], "has_more": False, "next_offset": None}

        matching_sessions = []

        for chat_dir in user_dir.iterdir():
            if not chat_dir.is_dir():
                continue
            db_file = chat_dir / "ws.db"
            if not db_file.exists():
                continue

            chat_id = chat_dir.name
            try:
                con = None
                try:
                    con = duckdb.connect(database=db_file.as_posix())
                    res = con.execute(
                        """
                        SELECT
                            (SELECT content FROM chat_history ORDER BY creation_timestamp ASC LIMIT 1) AS first_msg,
                            MAX(creation_timestamp) AS last_active
                        FROM chat_history
                        WHERE content ILIKE ?
                        HAVING COUNT(*) > 0
                        """,
                        (f"%{query}%",),
                    ).fetchone()
                finally:
                    if con:
                        con.close()

                if res and res[0] is not None and res[1] is not None:
                    content = res[0]
                    title = f"{content[:20]}..." if len(content) > 20 else content
                    last_active_dt = res[1]
                    last_active_str = (
                        last_active_dt.astimezone(datetime.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.%f"
                        )
                        + "Z"
                        if hasattr(last_active_dt, "isoformat")
                        else str(last_active_dt)
                    )
                    matching_sessions.append(
                        {
                            "id": chat_id,
                            "title": title,
                            "lastActive": last_active_str,
                            "_sort_ts": last_active_dt,
                        }
                    )
            except Exception as e:
                self.__log(f"Failed to search session {db_file}: {e}")
                continue

        matching_sessions.sort(key=lambda x: x["_sort_ts"], reverse=True)
        for session in matching_sessions:
            session.pop("_sort_ts", None)

        sliced = matching_sessions[offset : offset + limit]
        has_more = len(matching_sessions) > (offset + limit)
        next_offset = offset + limit if has_more else None

        return {
            "chats": sliced,
            "has_more": has_more,
            "next_offset": next_offset,
        }

    def delete_chat_session(self, user_id: str, chat_id: str) -> None:
        """
        Deletes a specific chat session by closing active connections and
        permanently removing the corresponding workspace directory.
        Raises FileNotFoundError if the chat session directory does not exist.
        """
        chat_dir = self.workspace_db_path / user_id / chat_id
        if not chat_dir.exists() or not chat_dir.is_dir():
            raise FileNotFoundError(
                f"Chat session with ID '{chat_id}' for user '{user_id}' not found."
            )

        self.close_workspace_connection(user_id, chat_id)

        try:
            from shutil import rmtree

            rmtree(chat_dir)
            self.__log(f"Successfully deleted chat session directory: {chat_dir}")
        except Exception as e:
            self.__log(f"Failed to delete chat session directory {chat_dir}: {e}")
            raise e

    def __load_documents_by_role(
        self,
        user_id: str,
        chat_id: str,
        dataset_name: str,
        con: duckdb.DuckDBPyConnection,
        state_id: UUID,
        role: str,
    ) -> list[AbstractDocument]:
        """Loads documents for the given state_id and role."""
        doc_rows = con.execute(
            """
            SELECT d.doc_id, d.retriever_type, d.content, d.path, d.last_node_id
            FROM documents d
            JOIN state_document_roles sdr
                ON d.state_id = sdr.state_id AND d.doc_id = sdr.doc_id
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
                WHERE state_id = ? AND doc_id = ?
                """,
                (state_id, doc_row["doc_id"]),
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
                self.dataset_manager.link_dataset_tables(
                    user_id,
                    chat_id,
                    dataset_name,
                    self.get_ws_db_connection,
                )

                if (
                    retriever_type == RetrieverType.MATERIALIZER
                    or retriever_type == RetrieverType.CONDUCTOR
                ):
                    select_query = f"""SELECT * FROM \"{doc_row['doc_id']}\";"""
                else:
                    select_query = (
                        f"""SELECT * FROM \"{dataset_name}\".\"{doc_row['doc_id']}\";"""
                    )
                try:
                    content = self.execute_query(
                        user_id,
                        chat_id,
                        select_query,
                    )
                except Exception as e:
                    continue
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
        self.logger.info(f"[WorkspaceManager] {message}")
