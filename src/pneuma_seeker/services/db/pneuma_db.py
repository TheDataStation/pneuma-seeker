import os
from logging import Logger
from pathlib import Path
from typing import Any

from pandas import DataFrame

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.db.datasets.manager import DatasetManager
from pneuma_seeker.services.db.memory.manager import MemoryManager
from pneuma_seeker.services.db.workspaces.manager import WorkspaceManager
from pneuma_seeker.services.db.workspaces.session_index import SessionIndex
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class PneumaDB:
    """
    Facade for dataset, workspace, and memory database operations.

    - DatasetManager: dataset .db files and dataset metadata
    - WorkspaceManager: per-user workspace DBs, session persistence, queries
    - MemoryManager: tribal knowledge, user preferences, agent-learning cache,
      and table/column metadata (all Postgres-backed, same instance as UserDB)
    """

    def __init__(
        self,
        config: Config,
        logger: Logger,
        dataset_db_path: str | None = None,
        workspace_db_path: str | None = None,
    ) -> None:
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

        self.dataset_manager = DatasetManager(self.dataset_db_path, self.logger)

        # Try to connect to the Postgres session index.  If Postgres is unavailable
        # (e.g. local dev without a running DB), we log a warning and fall back to
        # the O(n) filesystem scan that was in place before this index was introduced.
        try:
            session_index: SessionIndex | None = SessionIndex(self.config, self.logger)
        except Exception as e:
            self.logger.warning(
                f"[PneumaDB] Postgres session index unavailable, "
                f"falling back to filesystem session scan: {e}"
            )
            session_index = None

        self.workspace_manager = WorkspaceManager(
            self.workspace_db_path,
            self.config,
            self.logger,
            self.dataset_manager,
            session_index=session_index,
        )

        # Same defensive fallback as SessionIndex above: the memory layer is
        # additive functionality, so its absence must never take down the rest
        # of the system (e.g. dataset ingestion, chat) when Postgres is down.
        try:
            self.memory_manager: MemoryManager | None = MemoryManager(
                self.config, self.logger
            )
        except Exception as e:
            self.logger.warning(
                f"[PneumaDB] Postgres memory store unavailable, "
                f"memory layer disabled: {e}"
            )
            self.memory_manager = None

        self._conn_cache = self.workspace_manager._conn_cache
        self._pg_registry = self.dataset_manager._pg_registry

        self.target_tables_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "data_src",
            "target_tables",
        )

    @property
    def dataset_db_path(self) -> Path:
        """Gets the path to the dataset database directory."""
        return self._dataset_db_path

    @dataset_db_path.setter
    def dataset_db_path(self, value: str | Path) -> None:
        """Sets the path to the dataset database directory and updates the dataset manager."""
        self._dataset_db_path = Path(value)
        if hasattr(self, "dataset_manager"):
            self.dataset_manager.dataset_db_path = self._dataset_db_path

    @property
    def workspace_db_path(self) -> Path:
        """Gets the path to the workspace database directory."""
        return self._workspace_db_path

    @workspace_db_path.setter
    def workspace_db_path(self, value: str | Path) -> None:
        """Sets the path to the workspace database directory and updates the workspace manager."""
        self._workspace_db_path = Path(value)
        if hasattr(self, "workspace_manager"):
            self.workspace_manager.workspace_db_path = self._workspace_db_path

    # ------------------------------------------------------------------
    # Dataset Management
    # ------------------------------------------------------------------
    def get_dataset_connection(self, dataset_name: str, read_only: bool = True):
        """Gets a connection to the specified dataset."""
        return self.dataset_manager.get_dataset_connection(dataset_name, read_only)

    def ingest_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata_path: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """Ingests a dataset from the specified path, with optional metadata."""
        self.dataset_manager.ingest_dataset(
            dataset_name, dataset_path, metadata_path, overwrite
        )

    def list_dataset_tables(self, dataset_name: str) -> list[str]:
        """Lists table names in a dataset directly, without needing a chat session."""
        return self.dataset_manager.list_tables(dataset_name)

    def query_dataset_table(
        self,
        dataset_name: str,
        table_name: str,
        limit: int,
        offset: int,
        order_by: str | None,
        order_dir: str,
        search: str | None,
    ):
        """Queries a dataset table directly, without needing a chat session."""
        return self.dataset_manager.query_table(
            dataset_name, table_name, limit, offset, order_by, order_dir, search
        )

    def get_table_description(self, dataset_name: str, table_name: str) -> str:
        """
        Gets the description of a table in the specified dataset.

        Checks the memory layer's admin-editable table_metadata first (settable any
        time via CSV upload or the Memory page), falling back to the legacy
        ingest-time metadata.csv baked into the dataset directory if no memory
        entry exists yet.
        """
        if self.memory_manager is not None:
            entry = self.memory_manager.get_by_key(
                "table_metadata", dataset_name, table_name
            )
            if entry is not None:
                return entry.content
        return self.dataset_manager.get_table_description(dataset_name, table_name)

    def register_postgres_dataset(
        self, dataset_name: str, connection_string: str
    ) -> None:
        """Registers a PostgreSQL dataset with the specified name and connection string."""
        self.dataset_manager.register_postgres_dataset(dataset_name, connection_string)

    def link_dataset_tables(self, user_id: str, chat_id: str, dataset_name: str):
        """Links tables from the specified dataset into the user's workspace for the current chat session."""
        self.dataset_manager.link_dataset_tables(
            user_id, chat_id, dataset_name, self.get_ws_db_connection
        )

    def get_accessible_local_datasets(
        self, is_admin: bool, group_permissions: dict[str, str]
    ) -> list[str]:
        """Gets a list of local datasets accessible to the user based on their admin status and group permissions."""
        return self.dataset_manager.get_accessible_local_datasets(
            is_admin, group_permissions
        )

    def is_dataset_accessible(
        self, dataset_name: str, is_admin: bool, group_permissions: dict[str, str]
    ) -> bool:
        """Checks whether a single dataset is accessible to a user with the given admin status and group permissions."""
        return self.dataset_manager.is_dataset_accessible(
            dataset_name, is_admin, group_permissions
        )

    # ------------------------------------------------------------------
    # Workspace DB Management
    # ------------------------------------------------------------------
    def get_ws_db_connection(self, user_id: str, chat_id: str):
        """Gets a connection to the user's workspace database for the current chat session."""
        return self.workspace_manager.get_ws_db_connection(user_id, chat_id)

    def close_workspace_connection(self, user_id: str, chat_id: str):
        """Closes the connection to the user's workspace database for the current chat session."""
        self.workspace_manager.close_workspace_connection(user_id, chat_id)

    def close_all_connections(self):
        """Closes all database connections managed by the workspace manager."""
        self.workspace_manager.close_all_connections()

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def execute_query(
        self, user_id: str, chat_id: str, sql: str, sql_params: tuple = ()
    ) -> DataFrame:
        """Executes a SQL query against the user's WS database for the current chat session and returns the results as a DataFrame."""
        return self.workspace_manager.execute_query(user_id, chat_id, sql, sql_params)

    # ------------------------------------------------------------------
    # Session Persistence
    # ------------------------------------------------------------------
    def register_temporary_df(
        self, user_id: str, chat_id: str, df: DataFrame, table_name: str
    ):
        """Registers a temporary DataFrame in the user's workspace database for the current chat session."""
        self.workspace_manager.register_temporary_df(user_id, chat_id, df, table_name)

    def persist_df(
        self,
        user_id: str,
        chat_id: str,
        df: DataFrame,
        table_name: str,
        overwrite_content: bool,
    ):
        """Persists a DataFrame as a table in the user's workspace database for the current chat session, with an option to overwrite existing content."""
        self.workspace_manager.persist_df(
            user_id, chat_id, df, table_name, overwrite_content
        )

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
        is_plan_proposal: bool = False,
    ):
        """Persists the current chat session state, including user input, system response, conductor state, provenance graph, retrieved tables, enumerated tables, web search results, web crawl results, and join paths."""
        self.workspace_manager.persist_session(
            user_id,
            chat_id,
            dataset_name,
            new_user_input,
            new_system_response,
            conductor_state,
            provenance_graph,
            retrieved_tables,
            enumerated_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
            is_plan_proposal,
        )

    def load_chat_history(self, user_id: str, chat_id: str) -> list[LLMMessage]:
        """Loads the chat history for the specified user and chat session, returning a list of LLMMessage objects representing the conversation history."""
        return self.workspace_manager.load_chat_history(user_id, chat_id)

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
        """Loads the entire chat session state for the specified user and chat session, including chat history, conductor state, provenance graph, retrieved tables, enumerated tables, web search results, web crawl results, and join paths."""
        return self.workspace_manager.load_session(user_id, chat_id)

    def get_user_chat_sessions(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """Gets a list of chat sessions for the specified user."""
        return self.workspace_manager.get_user_chat_sessions(user_id, limit, offset)

    def search_chat_sessions(
        self, user_id: str, query: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """Searches chat sessions by message content for the specified user."""
        return self.workspace_manager.search_chat_sessions(
            user_id, query, limit, offset
        )

    def update_script_description(
        self, user_id: str, chat_id: str, description: str
    ) -> None:
        """Persists a lazily-generated script description to session_metadata."""
        self.workspace_manager.update_script_description(user_id, chat_id, description)

    def prepare_chat_deletion(self, user_id: str, chat_id: str) -> Path:
        """Closes the workspace connection and returns the directory path for background removal."""
        return self.workspace_manager.prepare_chat_deletion(user_id, chat_id)

    def delete_chat_session(self, user_id: str, chat_id: str) -> None:
        """Deletes the specified chat session for the user."""
        self.workspace_manager.delete_chat_session(user_id, chat_id)

    # ------------------------------------------------------------------
    # Memory Layer
    # ------------------------------------------------------------------
    def get_column_description(
        self, dataset_name: str, table_name: str, column_name: str
    ) -> str:
        """Gets the admin-set description of a column, or "" if none exists."""
        if self.memory_manager is None:
            return ""
        entry = self.memory_manager.get_by_key(
            "column_metadata", dataset_name, f"{table_name}.{column_name}"
        )
        return entry.content if entry is not None else ""

    def retrieve_memory_context(
        self,
        user_id: str,
        dataset_name: str,
        group_ids: list[str],
        query: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Returns up to k of the user's own preferences plus up to k tribal-knowledge
        entries visible to `group_ids` (the caller's group and its ancestors),
        ranked by BM25 relevance to `query`. Used to fold passive context into the
        Conductor's prompt every turn.
        """
        if self.memory_manager is None:
            return []
        preferences = self.memory_manager.search(
            "user_preference", query, user_id=user_id, k=k
        )
        tribal_knowledge = self.memory_manager.search(
            "tribal_knowledge",
            query,
            group_ids=group_ids,
            dataset_name=dataset_name,
            match_null_dataset=True,
            k=k,
        )
        return [
            {
                "memory_id": record.memory_id,
                "memory_type": record.memory_type,
                "content": record.content,
            }
            for record, _score in preferences + tribal_knowledge
        ]

    def record_memory_candidates(
        self,
        entries: list[dict[str, Any]],
        source_user_id: str,
        group_id: str | None,
        dataset_name: str | None,
        chat_id: str | None,
    ) -> None:
        """
        Persists LLM-extracted memory candidates. Each entry is
        {"scope": "local"|"global", "content": str} — "local" becomes a
        user_preference owned by source_user_id, "global" becomes tribal_knowledge
        owned by group_id (the contributing user's group).
        """
        if self.memory_manager is None:
            return
        for entry in entries:
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            if entry.get("scope") == "global" and group_id:
                self.memory_manager.create_entry(
                    "tribal_knowledge",
                    content,
                    "llm_extraction",
                    source_user_id,
                    group_id=group_id,
                    dataset_name=dataset_name,
                    chat_id=chat_id,
                )
            else:
                self.memory_manager.create_entry(
                    "user_preference",
                    content,
                    "llm_extraction",
                    source_user_id,
                    user_id=source_user_id,
                    dataset_name=dataset_name,
                    chat_id=chat_id,
                )

    def get_or_create_agent_learning(
        self,
        dataset_name: str,
        question: str,
        table_ids: list[str],
        resolve_fn,
    ) -> str:
        """
        Returns a cached answer for `question` on `dataset_name` if one already
        exists (exact key match, then BM25-similarity match above
        MEMORY_AGENT_LEARNING_SIMILARITY_THRESHOLD); otherwise calls `resolve_fn()`
        to compute the answer, caches it, and returns it. Shared by every
        CONTEXT_EXTRACTION call site (Conductor, Materializer, DS-Skeptic).
        """
        if self.memory_manager is None:
            return resolve_fn()
        exact = self.memory_manager.get_by_key("agent_learning", dataset_name, question)
        if exact is not None:
            return exact.content

        ranked = self.memory_manager.search(
            "agent_learning", question, dataset_name=dataset_name, k=1
        )
        if (
            ranked
            and ranked[0][1] >= self.config.MEMORY_AGENT_LEARNING_SIMILARITY_THRESHOLD
        ):
            return ranked[0][0].content

        answer = resolve_fn()
        self.memory_manager.create_entry(
            "agent_learning",
            answer,
            "agent_learning",
            None,
            dataset_name=dataset_name,
            key_text=question,
            extra={"table_ids": table_ids},
        )
        return answer
