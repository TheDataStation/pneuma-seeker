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
    # Dataset Management
    # ------------------------------------------------------------------
    def ingest_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata_path: str | None = None,
        overwrite: bool = True,
    ):
        """
        Ingests the CSV files at `dataset_path` as a local dataset (one Parquet
        table per file, with cleaned table and column names).
        """
        self.pneuma_db.ingest_dataset(
            dataset_name, dataset_path, metadata_path, overwrite
        )

    def dataset_fingerprint(self, dataset_name: str) -> str:
        """Returns a fingerprint that changes when the dataset's data may have."""
        return self.pneuma_db.dataset_fingerprint(dataset_name)

    def list_dataset_tables(self, dataset_name: str) -> list[str]:
        """Lists the table names in a dataset."""
        return self.pneuma_db.list_dataset_tables(dataset_name)

    def get_dataset_description(self, dataset_name: str) -> str:
        """Returns the description of a dataset ("" if nobody wrote one)."""
        return self.pneuma_db.get_dataset_description(dataset_name)

    def get_table_description(self, dataset_name: str, table_name: str) -> str:
        """Returns the description of a table in the dataset."""
        return self.pneuma_db.get_table_description(dataset_name, table_name)

    # ------------------------------------------------------------------
    # Memory Layer
    # ------------------------------------------------------------------
    def get_column_description(
        self, dataset_name: str, table_name: str, column_name: str
    ) -> str:
        """Returns the admin-set description of a column, or "" if none exists."""
        return self.pneuma_db.get_column_description(
            dataset_name, table_name, column_name
        )

    def rank_tribal_knowledge(
        self, dataset_name: str, group_ids: list[str], query: str
    ) -> list[dict[str, Any]]:
        """Returns the visible tribal knowledge, best BM25 match to `query` first."""
        return self.pneuma_db.rank_tribal_knowledge(dataset_name, group_ids, query)

    # ------------------------------------------------------------------
    # Dataset DB Linking into Workspace DB
    # ------------------------------------------------------------------
    def link_dataset_tables(self, user_id: str, chat_id: str, dataset_name: str):
        """
        Attaches a dataset into the workspace connection under its cleaned name,
        re-attaching it only if its data changed since.
        """
        self.pneuma_db.link_dataset_tables(user_id, chat_id, dataset_name)

    def detach_datasets_except(
        self, user_id: str, chat_id: str, allowed: set[str]
    ) -> list[str]:
        """Detaches every attached dataset not in `allowed`; returns the detached aliases."""
        return self.pneuma_db.detach_datasets_except(user_id, chat_id, allowed)

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
    def drop_relation(self, user_id: str, chat_id: str, name: str) -> None:
        """Drops the workspace table or view `name`, if it exists (whichever it is)."""
        kinds = self.execute_query(
            user_id,
            chat_id,
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_name = ? AND table_catalog = current_database()",
            (name,),
        )["table_type"].tolist()
        quoted = '"' + name.replace('"', '""') + '"'
        for kind in kinds:
            self.execute_query(
                user_id, chat_id, f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} IF EXISTS {quoted}"
            )

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
        is_plan_proposal: bool = False,
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
            is_plan_proposal,
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

    def update_script_description(
        self, user_id: str, chat_id: str, description: str
    ) -> None:
        """Persists a lazily-generated script description to session_metadata."""
        self.pneuma_db.update_script_description(user_id, chat_id, description)

    def get_user_chat_sessions(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """Gets a list of chat sessions for the specified user, returning a list of tuples containing chat IDs and their corresponding creation timestamps."""
        return self.pneuma_db.get_user_chat_sessions(user_id, limit, offset)
