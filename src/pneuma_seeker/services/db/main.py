import os
from logging import Logger
from pathlib import Path

from pandas import DataFrame

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.db.datasets.manager import DatasetManager
from pneuma_seeker.services.db.workspaces.manager import WorkspaceManager
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class PneumaDB:
    """
    Facade for dataset and workspace database operations.

    - DatasetManager: dataset .db files and dataset metadata
    - WorkspaceManager: per-user workspace DBs, session persistence, queries
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
        self.workspace_manager = WorkspaceManager(
            self.workspace_db_path, self.config, self.logger, self.dataset_manager
        )

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
        return self._dataset_db_path

    @dataset_db_path.setter
    def dataset_db_path(self, value: str | Path) -> None:
        self._dataset_db_path = Path(value)
        if hasattr(self, "dataset_manager"):
            self.dataset_manager.dataset_db_path = self._dataset_db_path

    @property
    def workspace_db_path(self) -> Path:
        return self._workspace_db_path

    @workspace_db_path.setter
    def workspace_db_path(self, value: str | Path) -> None:
        self._workspace_db_path = Path(value)
        if hasattr(self, "workspace_manager"):
            self.workspace_manager.workspace_db_path = self._workspace_db_path

    # ------------------------------------------------------------------
    # Dataset Management
    # ------------------------------------------------------------------
    def get_dataset_connection(self, dataset_name: str, read_only: bool = True):
        return self.dataset_manager.get_dataset_connection(dataset_name, read_only)

    def ingest_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata_path: str | None = None,
        overwrite: bool = True,
    ) -> None:
        self.dataset_manager.ingest_dataset(
            dataset_name, dataset_path, metadata_path, overwrite
        )

    def get_table_description(self, dataset_name: str, table_name: str) -> str:
        return self.dataset_manager.get_table_description(dataset_name, table_name)

    def register_postgres_dataset(
        self, dataset_name: str, connection_string: str
    ) -> None:
        self.dataset_manager.register_postgres_dataset(dataset_name, connection_string)

    def link_dataset_tables(self, user_id: str, chat_id: str, dataset_name: str):
        self.dataset_manager.link_dataset_tables(
            user_id, chat_id, dataset_name, self.get_ws_db_connection
        )

    # ------------------------------------------------------------------
    # Workspace DB Management
    # ------------------------------------------------------------------
    def get_ws_db_connection(self, user_id: str, chat_id: str):
        return self.workspace_manager.get_ws_db_connection(user_id, chat_id)

    def close_workspace_connection(self, user_id: str, chat_id: str):
        self.workspace_manager.close_workspace_connection(user_id, chat_id)

    def close_all_connections(self):
        self.workspace_manager.close_all_connections()

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def execute_query(
        self, user_id: str, chat_id: str, sql: str, sql_params: tuple = ()
    ) -> DataFrame:
        return self.workspace_manager.execute_query(user_id, chat_id, sql, sql_params)

    # ------------------------------------------------------------------
    # Session Persistence
    # ------------------------------------------------------------------
    def register_temporary_df(
        self, user_id: str, chat_id: str, df: DataFrame, table_name: str
    ):
        self.workspace_manager.register_temporary_df(user_id, chat_id, df, table_name)

    def persist_df(
        self,
        user_id: str,
        chat_id: str,
        df: DataFrame,
        table_name: str,
        overwrite_content: bool,
    ):
        self.workspace_manager.persist_df(
            user_id, chat_id, df, table_name, overwrite_content
        )

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
        self.workspace_manager.persist_session(
            user_id,
            chat_id,
            new_user_input,
            new_system_response,
            conductor_state,
            provenance_graph,
            retrieved_tables,
            enumerated_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
        )

    def load_chat_history(self, user_id: str, chat_id: str) -> list[LLMMessage]:
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
    ]:
        return self.workspace_manager.load_session(user_id, chat_id)
