# services/core/api/db.py
from logging import Logger
from typing import Any

from pandas import DataFrame

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class DBAPI:
    """
    DBAPI communicates with DB Service for datasets and workspaces management.
    """

    def __init__(
        self,
        config: Config,
        logger: Logger,
        dataset_db_path: str | None = None,
        workspace_db_path: str | None = None,
        pneuma_db: PneumaDB | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        if pneuma_db:
            self.pneuma_db = pneuma_db
        else:
            self.pneuma_db = PneumaDB(
                self.config, self.logger, dataset_db_path, workspace_db_path
            )

    # ------------------------------------------------------------------
    # Dataset Management (one .db per dataset)
    # ------------------------------------------------------------------
    def ingest_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata_path: str | None = None,
        overwrite: bool = True,
    ):
        """
        Stores CSV files inside the dataset's own DuckDB file.
        - table name = cleaned(Path(csv_file).stem)
        - cleans column names
        - only reads file once for ingestion (fast path)
        """
        self.pneuma_db.ingest_dataset(
            dataset_name, dataset_path, metadata_path, overwrite
        )

    def get_table_description(self, dataset_name: str, table_name: str) -> str:
        """Returns the description of a table in the dataset."""
        return self.pneuma_db.get_table_description(dataset_name, table_name)

    # ------------------------------------------------------------------
    # Dataset DB Linking into Workspace DB
    # ------------------------------------------------------------------
    def link_dataset_tables(self, user_id: str, chat_id: str, dataset_name: str):
        """
        Attach a dataset DB into the workspace connection under a safe alias.
        The alias is the cleaned dataset_name.
        This is idempotent (no error if already attached).
        """
        self.pneuma_db.link_dataset_tables(user_id, chat_id, dataset_name)

    def register_postgres_dataset(self, dataset_name: str, connection_string: str):
        """Registers a PostgreSQL dataset connection string for later linking."""
        self.pneuma_db.register_postgres_dataset(dataset_name, connection_string)

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
        return self.pneuma_db.execute_query(user_id, chat_id, sql, sql_params)

    # ------------------------------------------------------------------
    # Chat Session Persistence
    # ------------------------------------------------------------------
    def register_temporary_df(
        self, user_id: str, chat_id: str, df: DataFrame, table_name: str
    ):
        """Registers a temporary DataFrame in the workspace DB connection."""
        self.pneuma_db.register_temporary_df(user_id, chat_id, df, table_name)

    def persist_df(
        self,
        user_id: str,
        chat_id: str,
        df: DataFrame,
        table_name: str,
        overwrite_content: bool,
    ):
        """
        Persists a DataFrame as a table in the workspace DB connection.
        If the table already exists, it will be replaced.
        """
        self.pneuma_db.persist_df(user_id, chat_id, df, table_name, overwrite_content)

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
        """
        Persists the chat session.
        """
        self.pneuma_db.persist_session(
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
        )

    def load_chat_history(self, user_id: str, chat_id: str) -> list[LLMMessage]:
        """Loads persisted chat messages for the workspace."""
        return self.pneuma_db.load_chat_history(user_id, chat_id)

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
        return self.pneuma_db.load_session(user_id, chat_id)
    
    def get_user_chat_sessions(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """Gets a list of chat sessions for the specified user, returning a list of tuples containing chat IDs and their corresponding creation timestamps."""
        return self.pneuma_db.get_user_chat_sessions(user_id, limit, offset)
