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
from pneuma_seeker.services.core.action_set.impl.context_extraction import ContextExtraction
from pneuma_seeker.services.core.action_set.impl.query_executor import QueryExecutor
from pneuma_seeker.services.core.action_set.impl.python_executor import PythonExecutor
from pneuma_seeker.services.core.action_set.impl.equality_join import EqualityJoin
from pneuma_seeker.services.core.action_set.impl.semantic_column_generation import (
    SemanticColumnGeneration,
)
from pneuma_seeker.services.core.action_set.impl.semantic_join import (
    SemanticJoin,
    SyntacticSimMetric,
)
from pneuma_seeker.services.core.action_set.impl.situational_analysis import SituationalAnalysis
from pneuma_seeker.services.core.action_set.impl.state_manipulation import StateManipulation
from pneuma_seeker.services.core.action_set.impl.table_projection import (
    TableProjection,
)
from pneuma_seeker.services.core.action_set.impl.table_union import TableUnion
from pneuma_seeker.services.core.action_set.impl.join_path_extraction import (
    JoinPathExtraction,
)
from pneuma_seeker.services.core.action_set.impl.table_enumeration import (
    TableEnumeration,
)
from pneuma_seeker.services.core.action_set.impl.table_retrieve import TableRetrieve
from pneuma_seeker.services.core.action_set.impl.user_facing_communication import UserFacingCommunication
from pneuma_seeker.services.core.action_set.impl.web_crawl import WebCrawl
from pneuma_seeker.services.core.action_set.impl.web_search import WebSearch
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.main import Retriever
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType


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

        self.ir_system = Retriever(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.table_retrieve = TableRetrieve(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.table_enumeration = TableEnumeration(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.web_search = WebSearch(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.web_crawl = WebCrawl(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.join_path_extraction = JoinPathExtraction(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.context_extraction = ContextExtraction(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.situational_analysis = SituationalAnalysis(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.python_executor = PythonExecutor(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.query_executor = QueryExecutor(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.semantic_join = SemanticJoin(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.semantic_column_generation = SemanticColumnGeneration(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.table_projection = TableProjection(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.equality_join = EqualityJoin(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.table_union = TableUnion(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.state_manipulation = StateManipulation(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )
        self.user_facing_communication = UserFacingCommunication(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.language_model_api,
        )

        self.valid_conductor_actions = [
            ActionNames.USER_FACING_COMMUNICATION.value,
            ActionNames.SITUATIONAL_ANALYSIS.value,
            ActionNames.TABLE_RETRIEVE.value,
            ActionNames.TABLE_ENUMERATION.value,
            ActionNames.STATE_MANIPULATION.value,
            ActionNames.MATERIALIZER.value,
            ActionNames.PYTHON_EXECUTOR.value,
        ]
        self.valid_materializer_actions = [
            ActionNames.SITUATIONAL_ANALYSIS.value,
            ActionNames.TABLE_RETRIEVE.value,
            ActionNames.TABLE_ENUMERATION.value,
            ActionNames.QUERY_EXECUTOR.value,
            ActionNames.PYTHON_EXECUTOR.value,
            ActionNames.TABLE_PROJECTION.value,
            ActionNames.SEMANTIC_JOIN.value,
            ActionNames.SEMANTIC_COLUMN_GENERATION.value,
            ActionNames.TABLE_UNION.value,
            ActionNames.EQUALITY_JOIN.value,
        ]
        if self.config.ENABLE_WEB_SEARCH:
            self.valid_conductor_actions.append(ActionNames.WEB_SEARCH.value)
            self.valid_materializer_actions.append(ActionNames.WEB_SEARCH.value)
        if self.config.ENABLE_WEB_CRAWL:
            self.valid_conductor_actions.append(ActionNames.WEB_CRAWL.value)
            self.valid_materializer_actions.append(ActionNames.WEB_CRAWL.value)
        if self.config.ENABLE_CONTEXT_EXTRACTION:
            self.valid_conductor_actions.append(ActionNames.CONTEXT_EXTRACTION.value)
            self.valid_materializer_actions.append(ActionNames.CONTEXT_EXTRACTION.value)

    def is_valid_conductor_action(self, action_name: str) -> bool:
        return action_name in self.valid_conductor_actions

    def is_valid_materializer_action(self, action_name: str) -> bool:
        return action_name in self.valid_materializer_actions
    
    def get_action_description(self, action_name: ActionNames):
        match action_name:
            case ActionNames.TABLE_RETRIEVE:
                return self.table_retrieve.get_description()
            case ActionNames.JOIN_PATH_EXTRACTION:
                return self.join_path_extraction.get_description()
            case ActionNames.CONTEXT_EXTRACTION:
                return self.context_extraction.get_description()
            case ActionNames.TABLE_ENUMERATION:
                return self.table_enumeration.get_description()
            case ActionNames.WEB_SEARCH:
                return self.web_search.get_description()
            case ActionNames.WEB_CRAWL:
                return self.web_crawl.get_description()
            case ActionNames.QUERY_EXECUTOR:
                return self.query_executor.get_description()
            case ActionNames.PYTHON_EXECUTOR:
                return self.python_executor.get_description()
            case ActionNames.SEMANTIC_JOIN:
                return self.semantic_join.get_description()
            case ActionNames.SEMANTIC_COLUMN_GENERATION:
                return self.semantic_column_generation.get_description()
            case ActionNames.TABLE_PROJECTION:
                return self.table_projection.get_description()
            case ActionNames.EQUALITY_JOIN:
                return self.equality_join.get_description()
            case ActionNames.TABLE_UNION:
                return self.table_union.get_description()
            case ActionNames.SITUATIONAL_ANALYSIS:
                return self.situational_analysis.get_description()
            case ActionNames.STATE_MANIPULATION:
                return self.state_manipulation.get_description()
            case ActionNames.MATERIALIZER:
                return self.__get_materializer_action_description()
            case ActionNames.USER_FACING_COMMUNICATION:
                return self.user_facing_communication.get_description()
            case _:
                raise ValueError(f"Unknown action: {action_name}")
    
    def __get_materializer_action_description(self):
        return f"""**{ActionNames.MATERIALIZER.value}**:
  Populate tables in T with rows based on data integration and processing.
  - **Args**: {{"note": "<additional note or empty string>"}}"""

    def retrieve_documents(
        self,
        prompt: str,
        retriever_type: RetrieverType,
        k=10,
        sample_only: bool = False,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        return self.ir_system.retrieve_documents(
            retriever_type, prompt, k, sample_only, sample_size
        )

    def retrieve_multi_topic_documents(
        self,
        prompts: list[str],
        retriever_type: RetrieverType,
        k=10,
        sample_only: bool = False,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        multi_topic_docs: dict[str, list[AbstractDocument]] = (
            self.ir_system.retrieve_multi_topic_documents(
                retriever_type, prompts, k, sample_only, sample_size
            )
        )

        aggregated_docs: list[AbstractDocument] = []
        for topic, docs in multi_topic_docs.items():
            for doc in docs:
                doc.metadata["topic"] = topic
            aggregated_docs.extend(docs)
        return aggregated_docs

    def discover_join_paths(self, tables: list[AbstractDocument]) -> str:
        tables_df: dict[str, DataFrame] = {doc.doc_id: doc.content for doc in tables}
        return self.join_path_extraction.discover_join_paths(tables_df)
    
    def join_equality(
        self,
        left_table_id: str,
        right_table_id: str,
        left_table_column_keys: list[str],
        right_table_column_keys: list[str],
        result_table_id: str,
    ) -> DataFrame:
        return self.equality_join.apply(
            {
                "left_table_id": left_table_id,
                "right_table_id": right_table_id,
                "left_table_column_keys": left_table_column_keys,
                "right_table_column_keys": right_table_column_keys,
                "result_table_id": result_table_id,
            }
        )

    def union_tables(
        self,
        table_ids: list[str],
        result_table_id: str,
        provenance_column_name: str,
        provenance_regex: str,
    ) -> DataFrame:
        return self.table_union.apply(
            {
                "table_ids": table_ids,
                "result_table_id": result_table_id,
                "provenance_column_name": provenance_column_name,
                "provenance_regex": provenance_regex,
            }
        )

    def project_table(
        self,
        src_table_id: str,
        target_table_id: str,
        column_mapping: dict[str, str],
    ) -> DataFrame:
        return self.table_projection.apply(
            {
                "src_table_id": src_table_id,
                "target_table_id": target_table_id,
                "column_mapping": column_mapping,
            }
        )

    def execute_code(self, code: str, result_table_id: str) -> DataFrame:
        return self.python_executor.execute({"code": code, "result_table_id": result_table_id})

    def execute_query(self, query: str, result_table_id: str) -> DataFrame:
        return self.query_executor.execute(
            {"query": query, "result_table_id": result_table_id}
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
        syntactic_sim_metric: SyntacticSimMetric = SyntacticSimMetric.EDIT_DIST,
        use_llm: bool = False,
    ) -> DataFrame:
        return self.semantic_join.apply(
            {
                "left_table_id": left_table_id,
                "right_table_id": right_table_id,
                "relevant_left_cols": relevant_left_cols,
                "relevant_right_cols": relevant_right_cols,
                "joined_table_id": joined_table_id,
                "alpha": alpha,
                "top_k": top_k,
                "delimiter": delimiter,
                "embed_batch_size": embed_batch_size,
                "syntactic_sim_metric": syntactic_sim_metric,
                "use_llm": use_llm,
            }
        )

    def generate_semantic_column(
        self,
        src_table_id: str,
        src_table_columns: list[str],
        new_column_name: str,
        instruction: str,
    ) -> DataFrame:
        """
        Generates a new column based on the instruction and source table columns,
        using the language model to perform the transformation. Returns the updated
        table with the new column (sample rows only).
        """
        return self.semantic_column_generation.apply(
            {
                "src_table_id": src_table_id,
                "src_table_columns": src_table_columns,
                "new_column_name": new_column_name,
                "instruction": instruction,
            }
        )

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

    def append_comment_to_existing_code(self, code: str, comment: str):
        return append_comment_to_existing_code(code, comment)
