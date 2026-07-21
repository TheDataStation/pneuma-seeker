from logging import Logger
from typing import Any

from pandas import DataFrame

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.provenance.provenance_helper import (
    append_comment_to_existing_code,
    generate_pandas_read_csv_code,
    generate_pandas_read_multi_doc_code,
    generate_read_external_tables_code,
    generate_semantic_col_generator_code,
    generate_semantic_join_generator_code,
    generate_table_select_code,
    generate_view_textual_document_code,
)
from pneuma_seeker.services.core.action_set.impl.join_path_extraction import (
    JoinPathExtraction,
)
from pneuma_seeker.services.core.action_set.registry import ActionRegistry
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.main import Retriever
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class ActionSet:
    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        prov_graph: ProvenanceGraph,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.prov_graph = prov_graph
        self.db_api = db_api
        self.language_model_api = language_model_api

        self.registry = ActionRegistry(
            user_id, chat_id, config, logger, db_api, language_model_api
        )
        self.ir_system = Retriever(
            user_id, chat_id, config, logger, db_api, language_model_api
        )
        # Keep a typed reference for direct internal use (not LLM-accessible)
        self.join_path_extraction: JoinPathExtraction = self.registry.get(  # type: ignore[assignment]
            ActionNames.JOIN_PATH_EXTRACTION
        )

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def is_valid_conductor_action(self, action_name: str) -> bool:
        return self.registry.is_valid_action(action_name, AgentType.CONDUCTOR)

    def is_valid_materializer_action(self, action_name: str) -> bool:
        return self.registry.is_valid_action(action_name, AgentType.MATERIALIZER)

    # ------------------------------------------------------------------
    # Description helpers (used by prompt factories)
    # ------------------------------------------------------------------

    def get_action_description(
        self, action_name: ActionNames, agent: AgentType | None = None
    ) -> str:
        return self.registry.get_description(action_name, agent)

    # ------------------------------------------------------------------
    # Data retrieval wrappers
    # ------------------------------------------------------------------

    def retrieve_documents(
        self,
        prompt: str,
        retriever_type: RetrieverType,
        dataset_name: str = "",
        k: int = 10,
        sample_only: bool = False,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        return self.ir_system.retrieve_documents(
            retriever_type, prompt, dataset_name, k, sample_only, sample_size
        )

    def retrieve_multi_topic_documents(
        self,
        prompts: list[str],
        retriever_type: RetrieverType,
        dataset_name: str = "",
        k: int = 10,
        sample_only: bool = False,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        multi_topic_docs: dict[str, list[AbstractDocument]] = (
            self.ir_system.retrieve_multi_topic_documents(
                retriever_type, prompts, dataset_name, k, sample_only, sample_size
            )
        )
        aggregated: list[AbstractDocument] = []
        for topic, docs in multi_topic_docs.items():
            for doc in docs:
                doc.metadata["topic"] = topic
            aggregated.extend(docs)
        return aggregated

    def discover_join_paths(self, tables: list[AbstractDocument]) -> str:
        tables_df: dict[str, DataFrame] = {doc.doc_id: doc.content for doc in tables}
        return self.join_path_extraction.discover_join_paths(tables_df)

    # ------------------------------------------------------------------
    # Operator wrappers (thin delegates to action instances)
    # ------------------------------------------------------------------

    def join_equality(
        self,
        left_table_id: str,
        right_table_id: str,
        left_table_column_keys: list[str],
        right_table_column_keys: list[str],
        result_table_id: str,
        dataset_name: str = "",
    ) -> DataFrame:
        return self.registry.get(ActionNames.EQUALITY_JOIN).apply(  # type: ignore[union-attr]
            {
                "left_table_id": left_table_id,
                "right_table_id": right_table_id,
                "left_table_column_keys": left_table_column_keys,
                "right_table_column_keys": right_table_column_keys,
                "result_table_id": result_table_id,
                "dataset_name": dataset_name,
            }
        )

    def union_tables(
        self,
        table_ids: list[str],
        result_table_id: str,
        provenance_column_name: str,
        provenance_regex: str,
        dataset_name: str = "",
    ) -> DataFrame:
        return self.registry.get(ActionNames.TABLE_UNION).apply(  # type: ignore[union-attr]
            {
                "table_ids": table_ids,
                "result_table_id": result_table_id,
                "provenance_column_name": provenance_column_name,
                "provenance_regex": provenance_regex,
                "dataset_name": dataset_name,
            }
        )

    def project_table(
        self,
        src_table_id: str,
        target_table_id: str,
        column_mapping: dict[str, str],
        dataset_name: str = "",
    ) -> DataFrame:
        return self.registry.get(ActionNames.TABLE_PROJECTION).apply(  # type: ignore[union-attr]
            {
                "src_table_id": src_table_id,
                "target_table_id": target_table_id,
                "column_mapping": column_mapping,
                "dataset_name": dataset_name,
            }
        )

    def execute_code(self, code: str, result_table_id: str) -> DataFrame:
        return self.registry.get(ActionNames.PYTHON_EXECUTOR).execute(  # type: ignore[union-attr]
            {"code": code, "result_table_id": result_table_id}
        )

    def execute_query(self, query: str, result_table_id: str) -> DataFrame:
        return self.registry.get(ActionNames.QUERY_EXECUTOR).execute(  # type: ignore[union-attr]
            {"query": query, "result_table_id": result_table_id}
        )

    def user_facing_communication(
        self,
        planning_messages: list[LLMMessage],
        user_message: str,
        interaction_history: list[LLMMessage],
        forced: bool = False,
        plan_mode: bool = False,
    ) -> str:
        return self.registry.get(ActionNames.USER_FACING_COMMUNICATION).execute(  # type: ignore[union-attr]
            {
                "planning_messages": planning_messages,
                "user_message": user_message,
                "interaction_history": interaction_history,
                "forced": forced,
                "plan_mode": plan_mode,
            }
        )

    def join_semantic(
        self,
        left_table_id: str,
        right_table_id: str,
        relevant_left_cols: list[str],
        relevant_right_cols: list[str],
        joined_table_id: str,
        alpha: float = 0.5,
        top_k: int = 3,
        delimiter: str = " [SEP] ",
        embed_batch_size: int = 30,
        mode: str | None = None,
        use_llm: bool = False,
        dataset_name: str = "",
    ) -> DataFrame:
        args: dict[str, Any] = {
            "left_table_id": left_table_id,
            "right_table_id": right_table_id,
            "relevant_left_cols": relevant_left_cols,
            "relevant_right_cols": relevant_right_cols,
            "joined_table_id": joined_table_id,
            "alpha": alpha,
            "top_k": top_k,
            "delimiter": delimiter,
            "embed_batch_size": embed_batch_size,
            "use_llm": use_llm,
            "dataset_name": dataset_name,
        }
        if mode is not None:
            args["mode"] = mode
        return self.registry.get(ActionNames.SEMANTIC_JOIN).apply(args)  # type: ignore[union-attr]

    def generate_semantic_column(
        self,
        src_table_id: str,
        src_table_columns: list[str],
        new_column_name: str,
        instruction: str,
    ) -> DataFrame:
        return self.registry.get(ActionNames.SEMANTIC_COLUMN_GENERATION).apply(  # type: ignore[union-attr]
            {
                "src_table_id": src_table_id,
                "src_table_columns": src_table_columns,
                "new_column_name": new_column_name,
                "instruction": instruction,
            }
        )

    # ------------------------------------------------------------------
    # Provenance code-generation helpers (unchanged)
    # ------------------------------------------------------------------

    def generate_read_external_tables_code(
        self, table_number: int, doc: AbstractDocument
    ):
        return generate_read_external_tables_code(table_number, doc)

    def generate_pandas_read_csv_code(self, doc: AbstractDocument):
        return generate_pandas_read_csv_code(doc)

    def generate_view_textual_document_code(self, doc: AbstractDocument):
        return generate_view_textual_document_code(doc)

    def generate_pandas_read_multi_doc_code(self, docs: list[AbstractDocument]):
        return generate_pandas_read_multi_doc_code(docs)

    def generate_table_select_code(
        self, target_var_name: str, source_id: str, relevant_cols: list[str]
    ):
        return generate_table_select_code(target_var_name, source_id, relevant_cols)

    def generate_semantic_col_generator_code(
        self,
        src_table_columns: list[str],
        doc: AbstractDocument,
        new_col_name: str,
        new_col_values: list[Any],
    ):
        return generate_semantic_col_generator_code(
            src_table_columns, doc, new_col_name, new_col_values
        )

    def generate_semantic_join_generator_code(
        self,
        doc_1: AbstractDocument,
        doc_2: AbstractDocument,
        relevant_left_cols: list[str],
        relevant_right_cols: list[str],
        top_k: int,
    ):
        return generate_semantic_join_generator_code(
            doc_1, doc_2, relevant_left_cols, relevant_right_cols, top_k
        )

    def run_context_extraction(
        self,
        uncertainties: list[dict],
        available_tables: list[AbstractDocument],
        result_table_name: str,
    ) -> tuple[str, list[str]]:
        return self.registry.get(ActionNames.CONTEXT_EXTRACTION).apply(  # type: ignore[union-attr]
            {
                "uncertainties": uncertainties,
                "available_tables": available_tables,
                "result_table_name": result_table_name,
                "execute_code_fn": self.execute_code,
            }
        )

    def resolve_entities(
        self,
        source_table_id: str,
        target_column: str,
        output_mapping_table_id: str,
        canonical_entities: list[str] | None = None,
    ) -> DataFrame:
        args: dict[str, Any] = {
            "source_table_id": source_table_id,
            "target_column": target_column,
            "output_mapping_table_id": output_mapping_table_id,
        }
        if canonical_entities is not None:
            args["canonical_entities"] = canonical_entities
        return self.registry.get(ActionNames.ENTITY_RESOLUTION).apply(args)  # type: ignore[union-attr]

    def append_comment_to_existing_code(self, code: str, comment: str):
        return append_comment_to_existing_code(code, comment)
