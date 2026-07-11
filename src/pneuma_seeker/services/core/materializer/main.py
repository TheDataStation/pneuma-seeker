# src/pneuma_seeker/core/materializer/main.py
from logging import Logger
from time import time
from typing import Any, Callable

from pandas import DataFrame
from tiktoken import encoding_for_model

from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.shared.schemas.core.action import ActionExecutionStatus, ActionNames
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    convert_retrieval_results_to_str,
)
from pneuma_seeker.services.core.materializer.prompt_factory import (
    MaterializerPromptFactory,
)
from pneuma_seeker.services.core.materializer.state import MaterializerState
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.provenance.graph import ProvenanceGraph, ProvenanceNode
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_code, parse_json
from pneuma_seeker.shared.str_processor import (
    dataframe_to_preview_str,
    extract_table_ids_from_code,
    extract_tables_from_sql_regex,
)


class Materializer:
    """
    Materializer class that orchestrates the materialization
    process using LLMs and various operations.
    """

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        prov_graph: ProvenanceGraph,
        action_set: ActionSet,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
        log_callback: Callable[[str], None] = lambda _: None,
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.prov_graph = prov_graph
        self.action_set = action_set
        self.db_api = db_api
        self.language_model_api = language_model_api
        self.log_callback = log_callback

        self.prompt_factory = MaterializerPromptFactory(self.config, self.action_set)
        self.state = MaterializerState(self.user_id, self.chat_id, self.db_api)
        self.actions: list[str] = []
        self.llm_messages: list[LLMMessage] = []
        self.last_intermediate_table_ids: set[str] = set()
        self._saved_intermediate_tables: list[AbstractDocument] = []

        self._action_handlers: dict[str, Any] = {
            ActionNames.SITUATIONAL_ANALYSIS.value: self._handle_situational_analysis,
            ActionNames.TABLE_RETRIEVE.value: self._handle_table_retrieve,
            ActionNames.WEB_SEARCH.value: self._handle_web_search,
            ActionNames.WEB_CRAWL.value: self._handle_web_crawl,
            ActionNames.TABLE_ENUMERATION.value: self._handle_table_enumeration,
            ActionNames.TABLE_PROJECTION.value: self._handle_table_projection,
            ActionNames.EQUALITY_JOIN.value: self._handle_equality_join,
            ActionNames.TABLE_UNION.value: self._handle_table_union,
            ActionNames.SEMANTIC_COLUMN_GENERATION.value: self._handle_semantic_column_generation,
            ActionNames.SEMANTIC_JOIN.value: self._handle_semantic_join,
            ActionNames.PYTHON_EXECUTOR.value: self._handle_python_executor,
            ActionNames.QUERY_EXECUTOR.value: self._handle_query_executor,
            ActionNames.CONTEXT_EXTRACTION.value: self._handle_context_extraction,
            ActionNames.ENTITY_RESOLUTION.value: self._handle_entity_resolution,
        }

    def materialize_T(
        self,
        T: dict[str, DataFrame],
        column_descriptions: dict[str, dict[str, str]],
        S: str,
        client_note="",
        update_mode: bool = False,
        external_tables: list[AbstractDocument] = [],
        prefetched_tables: list[AbstractDocument] = [],
        prefetched_web_search_result: AbstractDocument | None = None,
        prefetched_web_crawl_result: AbstractDocument | None = None,
        precomputed_join_paths: str | None = None,
    ) -> tuple[
        list[AbstractDocument],
        AbstractDocument | None,
        AbstractDocument | None,
        str | None,
        dict[str, DataFrame],
    ]:
        materializer_start_time = time()
        message = f"Materializing {len(T)} target tables (update_mode={update_mode})..."
        self._log(message)
        self.log_callback(message)

        self._reset_materializer(update_mode=update_mode)

        self.state.T = T
        self.state.column_descriptions = column_descriptions
        self.state.S = S

        self.db_api.link_dataset_tables(
            self.user_id, self.chat_id, self.config.DATA_SOURCES[0]
        )

        if len(prefetched_tables) > 0:
            self.state.retrieved_tables = prefetched_tables
        if len(external_tables) > 0:
            self.state.external_tables = external_tables
        if prefetched_web_search_result is not None:
            self.state.web_search_result = prefetched_web_search_result
        if prefetched_web_crawl_result is not None:
            self.state.web_crawl_result = prefetched_web_crawl_result
        if precomputed_join_paths is not None:
            self.state.join_paths = precomputed_join_paths

        self.llm_messages = [
            LLMMessage(
                role=Role.SYSTEM.value,
                content=self.prompt_factory.get_planning_prompt(
                    self.state.T,
                    self.state.column_descriptions,
                    self.state.S,
                    is_update_mode=update_mode,
                    prior_intermediate_table_ids=[
                        i.doc_id for i in self.state.intermediate_tables
                    ],
                ),
            )
        ]

        current_step = 0
        llm = self.language_model_api.llm
        prev_accumulated_in_tokens = llm.total_input_tokens
        prev_accumulated_out_tokens = llm.total_output_tokens
        prev_accumulated_llm_time = llm.total_llm_time
        baseline_in_tokens = prev_accumulated_in_tokens
        baseline_out_tokens = prev_accumulated_out_tokens
        baseline_llm_time = prev_accumulated_llm_time

        while (
            not self._check_completion(self.state.T)
            and current_step < self.config.MAX_MATERIALIZER_STEPS
        ):
            step_start_time = time()

            current_step += 1
            message = f"[Step {current_step} / up to {self.config.MAX_MATERIALIZER_STEPS}]: Planning materialization actions..."
            self._log(message)
            self.log_callback(message)

            context_msg = LLMMessage(
                role=Role.USER.value,
                content=self.prompt_factory.get_context_prompt(
                    self.state.retrieved_tables,
                    list(self.state.intermediate_tables),
                    self.actions[-5:],
                    current_step,
                    client_note,
                    self.state.external_tables,
                    self.state.web_search_result,
                    self.state.web_crawl_result,
                    self.state.join_paths,
                ),
            )

            try:
                llm_response = "".join(
                    self.language_model_api.chat(
                        self.llm_messages + [context_msg],
                        LLMOption(json_mode=True, max_new_tokens=2500),
                    )
                )
            except Exception as exc:
                error_msg = (
                    f"An unexpected error occurred while generating the plan: {exc}."
                )
                self._log(error_msg)
                self.log_callback(error_msg)
                raise exc

            self._log(f"Plan: {llm_response}")
            self.llm_messages.append(
                LLMMessage(
                    role=Role.USER.value,
                    content=self.prompt_factory.get_skeleton_context_prompt(
                        current_step
                    ),
                )
            )
            self.llm_messages.append(
                LLMMessage(role=Role.ASSISTANT.value, content=llm_response)
            )

            try:
                plan: list[dict[str, Any]] = self._validate_plan(
                    parse_json(llm_response).get("plan", [])
                )
                self.actions.append(str(plan))
            except Exception as exc:
                message = f"Parsing error: {exc}. Please fix the issue and try again."
                self._log(message)
                self.llm_messages.append(
                    LLMMessage(role=Role.USER.value, content=message)
                )
                self.log_callback("Fixing error in produced plan...")
                continue

            for action_plan in plan:
                self._log(f"Executing action: {action_plan}")
                action_name: str = action_plan.get("action", "")
                action_args: dict[str, Any] = action_plan.get("args", {})
                status = self.__execute_action(action_name, action_args)
                if status == ActionExecutionStatus.ERROR:
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=(
                                f"Remaining actions in this step were skipped because "
                                f"'{action_name}' failed. Address the issue above before "
                                "continuing."
                            ),
                        )
                    )
                    break

            step_end_time = time()
            llm = self.language_model_api.llm

            step_input_tokens = llm.total_input_tokens - prev_accumulated_in_tokens
            step_output_tokens = llm.total_output_tokens - prev_accumulated_out_tokens
            step_llm_time = llm.total_llm_time - prev_accumulated_llm_time

            prev_accumulated_in_tokens = llm.total_input_tokens
            prev_accumulated_out_tokens = llm.total_output_tokens
            prev_accumulated_llm_time = llm.total_llm_time

            self._log_step_profiling(
                step_input_tokens,
                step_output_tokens,
                step_end_time - step_start_time,
                step_llm_time,
            )

        materializer_end_time = time()
        llm = self.language_model_api.llm
        self._log_overall_profiling(
            materializer_start_time,
            materializer_end_time,
            llm.total_input_tokens - baseline_in_tokens,
            llm.total_output_tokens - baseline_out_tokens,
            llm.total_llm_time - baseline_llm_time,
        )

        message = f"Materialization completed successfuly!"
        self._log(message)
        self.log_callback(message)

        self.last_intermediate_table_ids = set(
            i.doc_id for i in self.state.intermediate_tables
        )
        self._saved_intermediate_tables = list(self.state.intermediate_tables)
        final_result: dict[str, DataFrame] = {}
        for intermediate_table_doc in self.state.intermediate_tables:
            if intermediate_table_doc.doc_id in self.state.T.keys():
                final_result[intermediate_table_doc.doc_id] = (
                    intermediate_table_doc.content
                )
        return (
            self.state.retrieved_tables,
            self.state.web_search_result,
            self.state.web_crawl_result,
            self.state.join_paths,
            final_result,
        )

    def _validate_plan(self, plan: Any) -> list[dict[str, Any]]:
        if not isinstance(plan, list):
            raise ValueError("The 'plan' field must be a list.")
        if len(plan) == 0:
            raise ValueError("The 'plan' list cannot be empty.")
        if not all(isinstance(item, dict) for item in plan):
            raise ValueError("All items in the 'plan' list must be JSON objects.")
        for action_plan in plan:
            if "action" not in action_plan:
                raise ValueError("Each action in the plan must have an 'action' field.")
            if not self.action_set.is_valid_materializer_action(action_plan["action"]):
                raise ValueError(f"Invalid action '{action_plan['action']}' specified.")
        return plan

    def __execute_action(
        self,
        action_name: str,
        action_args: dict[str, Any],
    ) -> ActionExecutionStatus:
        self._log(f"==> Executing action {action_name}...")

        handler = self._action_handlers.get(action_name)
        if handler is None:
            error_msg = f"{action_name} is not a valid action."
            self._log(f"==> {error_msg}")
            outcome = error_msg
            status = ActionExecutionStatus.ERROR
        else:
            outcome, status = handler(action_args)

        self.llm_messages.append(LLMMessage(role=Role.USER.value, content=outcome))
        return status

    def _handle_situational_analysis(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        message: str = action_args.get("message", "")
        return (
            f"You did a situational analysis: {message}",
            ActionExecutionStatus.SUCCESS,
        )

    def _handle_table_retrieve(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        prompts = action_args.get("prompts", [])
        if not isinstance(prompts, list) or not all(
            isinstance(p, str) for p in prompts
        ):
            error_msg = "The 'prompts' argument must be a list of strings."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if len(prompts) == 0:
            error_msg = "The 'prompts' list cannot be empty."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        self.state.retrieved_tables = self.action_set.retrieve_multi_topic_documents(
            prompts, RetrieverType.PNEUMA_RETRIEVER, 10, True, 5
        )
        if len(self.state.retrieved_tables) == 0:
            error_msg = "No tables were retrieved."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        self.state.join_paths = self.action_set.discover_join_paths(
            self.state.retrieved_tables
        )
        success_msg = 'Successfully retrieved tables. Notice that the "retrieved internal tables" have been filled.'
        self._log(f"==> {success_msg}")
        self._log(
            f"Retrieved tables:\n {[i.doc_id for i in self.state.retrieved_tables]}"
        )
        for doc in self.state.retrieved_tables:
            if doc.path is not None:
                node_id = self._create_or_get_read_node(
                    doc,
                    RetrieverType.PNEUMA_RETRIEVER,
                    self.action_set.generate_pandas_read_csv_code(doc),
                    "Retrieves an internal table from Pneuma-Retriever.",
                )
                if node_id is not None:
                    doc.last_node_id = node_id
        self._log_table_repr_tokens(self.state.retrieved_tables)
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_web_search(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        if not self.config.ENABLE_WEB_SEARCH:
            error_msg = (
                f"{ActionNames.WEB_SEARCH.value} is not enabled in the configuration."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        prompt = action_args.get("prompt", "")
        web_search_results = self.action_set.retrieve_documents(
            prompt, RetrieverType.WEB_SEARCH
        )
        if len(web_search_results) == 0:
            error_msg = f"No relevant information was found from {ActionNames.WEB_SEARCH.value}."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        self.state.web_search_result = web_search_results[0]
        success_msg = f'Successfully retrieved information from {ActionNames.WEB_SEARCH.value}. Notice that the "{ActionNames.WEB_SEARCH.value} result" have been filled. Join paths have also been updated accordingly."'
        self._log(f"==> {success_msg}")
        new_node = ProvenanceNode(
            source_retriever=RetrieverType.WEB_SEARCH,
            python_code=self.action_set.generate_view_textual_document_code(
                self.state.web_search_result
            ),
            description=f"Searches the web using this query:\n{prompt}",
        )
        self.prov_graph.add_node(new_node, True)
        self.state.web_search_result.last_node_id = new_node.id
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_web_crawl(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        if not self.config.ENABLE_WEB_CRAWL:
            error_msg = (
                f"{ActionNames.WEB_CRAWL.value} is not enabled in the configuration."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        prompt = action_args.get("url", "")
        web_crawl_results = self.action_set.retrieve_documents(
            prompt, RetrieverType.WEB_CRAWL
        )
        if len(web_crawl_results) == 0:
            error_msg = (
                f"No relevant information was found from {ActionNames.WEB_CRAWL.value}."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        self.state.web_crawl_result = web_crawl_results[0]
        success_msg = f'Successfully retrieved information from {ActionNames.WEB_CRAWL.value}. Notice that the "{ActionNames.WEB_CRAWL.value} result" have been filled.'
        self._log(f"==> {success_msg}")
        new_node = ProvenanceNode(
            source_retriever=RetrieverType.WEB_CRAWL,
            python_code=self.action_set.generate_view_textual_document_code(
                self.state.web_crawl_result
            ),
            description=f"Crawls this web page:\n{prompt}",
        )
        self.prov_graph.add_node(new_node, True)
        self.state.web_crawl_result.last_node_id = new_node.id
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_table_enumeration(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        patterns = action_args.get("patterns", [])
        if not isinstance(patterns, list) or not all(
            isinstance(p, str) for p in patterns
        ):
            error_msg = "The 'patterns' argument must be a list of strings."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if len(patterns) == 0:
            error_msg = "The 'patterns' list cannot be empty."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        extra_tables: list[AbstractDocument] = (
            self.action_set.retrieve_multi_topic_documents(
                patterns, RetrieverType.ENUMERATOR, 20, True, 5
            )
        )
        pattern_desc = str(patterns)
        if len(extra_tables) == 0:
            error_msg = "There are no tables that match the pattern."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        success_msg = f'Successfully retrieved all tables that match the pattern(s) {pattern_desc}. You can use them to materialize T, even if you have not called {ActionNames.TABLE_RETRIEVE.value} before, as these tables have been included to "retrieved internal tables".'
        self._log(f"==> {success_msg}")
        read_code = self.action_set.generate_pandas_read_multi_doc_code(extra_tables)
        new_node = ProvenanceNode(
            source_retriever=RetrieverType.ENUMERATOR,
            python_code=read_code,
            description=f"Enumerates all tables whose names match this regular expression (RegEx) pattern:\n{pattern_desc}.",
        )
        self.prov_graph.add_node(new_node, True)
        for extra_table in extra_tables:
            last_id = getattr(extra_table, "last_node_id", None)
            if last_id is not None:
                existing_node = self.prov_graph.get_node_by_id(last_id)
                if existing_node is not None:
                    continue
            extra_table.last_node_id = new_node.id
        self.state.retrieved_tables = list(
            set(self.state.retrieved_tables).union(set(extra_tables))
        )
        self._log(
            f"Retrieved tables:\n {[i.doc_id for i in self.state.retrieved_tables]}"
        )
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_table_projection(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        all_tables = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )
        all_table_doc_ids = [i.doc_id for i in all_tables]
        outcome_messages: list[str] = []

        for target_table_id, retrieved_table_info in action_args.items():
            if isinstance(retrieved_table_info, list):
                if len(retrieved_table_info) == 0:
                    msg = f"Skipping {target_table_id!r}: empty list provided as value."
                    self._log(f"==> {msg}")
                    outcome_messages.append(msg)
                    continue
                retrieved_table_info = retrieved_table_info[0]
            if not isinstance(retrieved_table_info, dict):
                msg = (
                    f"Invalid argument for target {target_table_id!r}: expected a dict."
                )
                self._log(f"==> {msg}")
                outcome_messages.append(msg)
                continue
            table_id_to_project = str(retrieved_table_info.get("id", "")).strip()
            if table_id_to_project.startswith("Table "):
                table_id_to_project = table_id_to_project[6:].strip()
            column_mapping = retrieved_table_info.get("columns", {})
            target_table_id = target_table_id.strip()
            if not isinstance(column_mapping, dict):
                error_msg = "Invalid 'columns' for table_projection: expected a dict mapping output column names to source column names."
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            if len(column_mapping) == 0:
                error_msg = (
                    "Invalid 'columns' for table_projection: empty mapping provided."
                )
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            if not all(
                isinstance(out_col, str) and isinstance(src_col, str)
                for out_col, src_col in column_mapping.items()
            ):
                error_msg = (
                    "Invalid 'columns' for table_projection: mapping must be str->str."
                )
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            output_columns = list(column_mapping.keys())
            source_columns = list(column_mapping.values())
            if table_id_to_project.split(".")[-1].strip('"') not in all_table_doc_ids:
                error_msg = "Invalid table ID to select. Ensure the table exists."
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            if target_table_id not in self.state.T:
                error_msg = f"Error: The ID {target_table_id} does not exist in T."
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            matches = [
                i
                for i in all_tables
                if i.doc_id == table_id_to_project.split(".")[-1].strip('"')
            ]
            if not matches:
                error_msg = f"Table {table_id_to_project.split('.')[-1].strip(chr(34))!r} not found in the available tables."
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            self._log(
                f"==> target_table_id: {target_table_id}; table_id_to_project: {table_id_to_project}"
            )
            try:
                projected_table_sample_rows = self.action_set.project_table(
                    table_id_to_project, target_table_id, column_mapping
                )
            except Exception as e:
                error_msg = f"Failed projecting columns {column_mapping!r} from table {table_id_to_project!r}: {e}"
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
            select_code = self.action_set.generate_table_select_code(
                target_table_id, table_id_to_project, source_columns
            )
            parent_node_id = self._create_or_get_read_node(
                matches[0],
                matches[0].retriever_type,
                self.action_set.generate_pandas_read_csv_code(matches[0]),
                "",
            )
            child_node_desc = (
                f"Projects a table\n\n"
                f"- ID: `{table_id_to_project}`\n"
                f"- columns: {', '.join(f'`{col}`' for col in output_columns)}\n\n"
                f"into a target table:\n\n`{target_table_id}`"
            )
            child_node_desc += (
                " (partially)."
                if set(output_columns) != set(self.state.T[target_table_id].columns)
                else "."
            )
            child_node = ProvenanceNode(
                source_retriever=RetrieverType.MATERIALIZER,
                python_code=select_code,
                description=child_node_desc,
            )
            self.prov_graph.add_node(child_node, True)
            if parent_node_id is not None:
                parent_node = self.prov_graph.get_node_by_id(parent_node_id)
                if parent_node is not None:
                    self.prov_graph.connect(parent_node, child_node)
            self.state.add_intermediate_table(
                Table(
                    doc_id=target_table_id,
                    retriever_type=RetrieverType.MATERIALIZER,
                    content=projected_table_sample_rows,
                    metadata={},
                    last_node_id=child_node.id,
                )
            )
            success_msg = "Successfully projected retrieved tables in the mapping as target tables. Notice the state's intermediate tables have changed, but please CHECK if the schemas in the selected tables match, either fully or partially, with the ones in target tables."
            self._log(f"==> {success_msg}")
            outcome_messages.append(success_msg)

        if not outcome_messages:
            return "No table projections were processed.", ActionExecutionStatus.ERROR
        return "\n".join(outcome_messages), ActionExecutionStatus.SUCCESS

    def _handle_equality_join(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        left_table_id = action_args.get("left_table_id")
        right_table_id = action_args.get("right_table_id")
        left_keys = action_args.get("left_table_column_keys")
        right_keys = action_args.get("right_table_column_keys")
        result_table_id = action_args.get("result_table_id")
        if not isinstance(left_table_id, str) or not isinstance(right_table_id, str):
            error_msg = "Invalid equality_join args: left_table_id and right_table_id must be strings."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(left_keys, list) or not all(
            isinstance(x, str) for x in left_keys
        ):
            error_msg = "Invalid equality_join args: left_table_column_keys must be a list[str]."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(right_keys, list) or not all(
            isinstance(x, str) for x in right_keys
        ):
            error_msg = "Invalid equality_join args: right_table_column_keys must be a list[str]."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(result_table_id, str):
            error_msg = "Invalid equality_join args: result_table_id must be a string."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        try:
            joined_preview = self.action_set.join_equality(
                left_table_id, right_table_id, left_keys, right_keys, result_table_id
            )
        except Exception as e:
            error_msg = f"Failed equality_join: {e}"
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        child_node = ProvenanceNode(
            source_retriever=RetrieverType.MATERIALIZER,
            python_code="",
            description=(
                f"Equality-joins tables\n\n"
                f"- left: `{left_table_id}` on {left_keys}\n"
                f"- right: `{right_table_id}` on {right_keys}\n"
                f"- result: `{result_table_id}`"
            ),
        )
        self.prov_graph.add_node(child_node, True)
        self.state.add_intermediate_table(
            Table(
                doc_id=result_table_id,
                retriever_type=RetrieverType.MATERIALIZER,
                content=joined_preview,
                metadata={},
                last_node_id=child_node.id,
            )
        )
        success_msg = (
            "Successfully performed equality join and materialized the result table."
        )
        self._log(f"==> {success_msg}")
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_table_union(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        table_ids = action_args.get("table_ids")
        result_table_id = action_args.get("result_table_id")
        provenance_regex = action_args.get("provenance_regex")
        provenance_column_name = action_args.get("provenance_column_name")
        if not isinstance(table_ids, list) or not all(
            isinstance(x, str) for x in table_ids
        ):
            error_msg = "Invalid table_union args: table_ids must be a list[str]."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(result_table_id, str):
            error_msg = "Invalid table_union args: result_table_id must be a string."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(provenance_column_name, str) or not provenance_column_name:
            error_msg = "Invalid table_union args: provenance_column_name must be a non-empty string."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(provenance_regex, str) or not provenance_regex:
            error_msg = (
                "Invalid table_union args: provenance_regex must be a non-empty string."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        try:
            union_preview = self.action_set.union_tables(
                table_ids=table_ids,
                result_table_id=result_table_id,
                provenance_column_name=str(provenance_column_name),
                provenance_regex=str(provenance_regex),
            )
        except Exception as e:
            error_msg = f"Failed table_union: {e}"
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        child_node = ProvenanceNode(
            source_retriever=RetrieverType.MATERIALIZER,
            python_code="",
            description=(
                f"Unions tables into `{result_table_id}`\n\n"
                f"- sources: {table_ids}\n"
                f"- provenance: `{provenance_column_name}` via regex"
            ),
        )
        self.prov_graph.add_node(child_node, True)
        self.state.add_intermediate_table(
            Table(
                doc_id=result_table_id,
                retriever_type=RetrieverType.MATERIALIZER,
                content=union_preview,
                metadata={},
                last_node_id=child_node.id,
            )
        )
        success_msg = "Successfully unioned tables and materialized the result table."
        self._log(f"==> {success_msg}")
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_semantic_column_generation(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        table_id: str | None = action_args.get("table_id")
        new_column_name: str | None = action_args.get("new_column_name")
        src_table_columns: list[str] | None = action_args.get("relevant_columns")
        instruction: str | None = action_args.get("instruction")
        if table_id is None or table_id not in [
            i.doc_id for i in self.state.intermediate_tables
        ]:
            error_msg = "table_id is not valid (not part of intermediate tables)."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if new_column_name is None:
            error_msg = "new_column_name is not provided."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if src_table_columns is None:
            error_msg = "relevant_columns is not provided."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        conditioned_table_doc = [
            i for i in self.state.intermediate_tables if i.doc_id == table_id
        ][0]
        if conditioned_table_doc.retriever_type != RetrieverType.MATERIALIZER:
            error_msg = "Semantic column generation is only supported for intermediate tables generated within the materialization process."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        conditioned_table_sample_rows: DataFrame = conditioned_table_doc.content
        if not set(src_table_columns) <= set(
            list(conditioned_table_sample_rows.columns)
        ):
            error_msg = (
                f"relevant_columns must be a subset of the columns of table {table_id}."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if instruction is None:
            error_msg = "instruction is not provided."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        augmented_table_sample_rows = self.action_set.generate_semantic_column(
            conditioned_table_doc.doc_id,
            src_table_columns,
            new_column_name,
            instruction,
        )
        conditioned_table_doc.content = augmented_table_sample_rows
        success_msg = f"Successfully added a new column named {new_column_name} to table with ID {table_id}."
        self._log(f"==> {success_msg}")
        sem_col_code = self.action_set.generate_semantic_col_generator_code(
            src_table_columns,
            conditioned_table_doc,
            new_column_name,
            list(augmented_table_sample_rows[new_column_name]),
        )
        parent_node_id = self._create_or_get_read_node(
            conditioned_table_doc,
            conditioned_table_doc.retriever_type,
            self.action_set.generate_pandas_read_csv_code(conditioned_table_doc),
            "",
        )
        new_node = ProvenanceNode(
            source_retriever=RetrieverType.MATERIALIZER,
            python_code=sem_col_code,
            description=f"Uses an LLM to generate column named `{new_column_name}` in the table `{table_id}`, conditioned on the following columns: {', '.join(f'`{col}`' for col in src_table_columns)}.",
        )
        self.prov_graph.add_node(new_node, True)
        if parent_node_id is not None:
            parent_node = self.prov_graph.get_node_by_id(parent_node_id)
            if parent_node is not None:
                self.prov_graph.connect(parent_node, new_node)
        conditioned_table_doc.last_node_id = new_node.id
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_semantic_join(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        all_tables = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )
        left_table_id: str | None = action_args.get("left_table_id")
        right_table_id: str | None = action_args.get("right_table_id")
        relevant_left_cols: list[str] | None = action_args.get("relevant_left_cols")
        relevant_right_cols: list[str] | None = action_args.get("relevant_right_cols")
        joined_table_id: str | None = action_args.get("joined_table_id")
        all_table_ids = [i.doc_id for i in all_tables]
        if left_table_id is None or left_table_id not in all_table_ids:
            error_msg = "left_table_id is not valid (not part of retrieved tables or the state's intermediate tables)."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if right_table_id is None or right_table_id not in all_table_ids:
            error_msg = "right_table_id is not valid (not part of retrieved tables or the state's intermediate tables)."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        left_table: DataFrame | None = None
        left_table_doc: AbstractDocument | None = None
        right_table: DataFrame | None = None
        right_table_doc: AbstractDocument | None = None
        for doc in all_tables:
            if doc.doc_id == left_table_id:
                left_table = doc.content
                left_table_doc = doc
            if doc.doc_id == right_table_id:
                right_table = doc.content
                right_table_doc = doc
        for check_val, error_msg in [
            (
                isinstance(left_table, DataFrame),
                f"left_table with ID {left_table_id} is not a DataFrame.",
            ),
            (
                isinstance(right_table, DataFrame),
                f"right_table with ID {right_table_id} is not a DataFrame.",
            ),
            (
                isinstance(left_table_doc, AbstractDocument),
                f"ID {left_table_id} does not correspond to a document.",
            ),
            (
                isinstance(right_table_doc, AbstractDocument),
                f"ID {right_table_id} does not correspond to a document.",
            ),
        ]:
            if not check_val:
                self._log(f"==> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
        if relevant_left_cols is None:
            error_msg = "relevant_left_cols is not provided."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if relevant_right_cols is None:
            error_msg = "relevant_right_cols is not provided."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not set(relevant_left_cols) <= set(list(left_table.columns)):  # type: ignore[union-attr]
            error_msg = "relevant_left_cols is not a subset of left_table's columns."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not set(relevant_right_cols) <= set(list(right_table.columns)):  # type: ignore[union-attr]
            error_msg = "relevant_right_cols is not a subset of right_table's columns."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not joined_table_id:
            error_msg = "joined_table_id is not provided."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        joined_table_sample_rows = self.action_set.join_semantic(
            left_table_id,
            right_table_id,
            relevant_left_cols,
            relevant_right_cols,
            joined_table_id,
            top_k=self.config.SEMANTIC_JOIN_TOP_K,
        )
        join_code = self.action_set.generate_semantic_join_generator_code(
            left_table_doc,  # type: ignore
            right_table_doc,  # type: ignore
            relevant_left_cols,
            relevant_right_cols,
            self.config.SEMANTIC_JOIN_TOP_K,
        )
        parent_node_1_id = self._create_or_get_read_node(
            left_table_doc,  # type: ignore
            left_table_doc.retriever_type,  # type: ignore
            self.action_set.generate_pandas_read_csv_code(left_table_doc),  # type: ignore
            "",
        )
        parent_node_2_id = self._create_or_get_read_node(
            right_table_doc,  # type: ignore
            right_table_doc.retriever_type,  # type: ignore
            self.action_set.generate_pandas_read_csv_code(right_table_doc),  # type: ignore
            "",
        )
        new_node = ProvenanceNode(
            source_retriever=RetrieverType.MATERIALIZER,
            python_code=join_code,
            description=f"Semantically joins `{left_table_id}` and `{right_table_id}`. For each row in the left table, keeps the `top-{self.config.SEMANTIC_JOIN_TOP_K}` matches from the right table (i.e., the most similar rows based on syntactic and semantic similarity). Similarity is computed based on these columns:\n- Left table: {', '.join(f'`{col}`' for col in relevant_left_cols)}\n- Right table: {', '.join(f'`{col}`' for col in relevant_right_cols)}",
        )
        self.prov_graph.add_node(new_node, True)
        for pid in [parent_node_1_id, parent_node_2_id]:
            node = self.prov_graph.get_node_by_id(pid) if pid else None
            if node is not None:
                self.prov_graph.connect(node, new_node)
        self.state.add_intermediate_table(
            Table(
                doc_id=joined_table_id,
                retriever_type=RetrieverType.MATERIALIZER,
                content=joined_table_sample_rows,
                metadata={},
                last_node_id=new_node.id,
            )
        )
        success_msg = "Successfully joined the left and right tables semantically. Notice the state's intermediate tables have changed."
        self._log(f"==> {success_msg}")
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_python_executor(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        all_tables = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )
        id_docs: dict[str, AbstractDocument] = {t.doc_id: t for t in all_tables}
        assign_to: str | None = action_args.get("assign_to")
        if assign_to is None or assign_to.strip() == "":
            error_msg = "'assign_to' argument is missing or empty."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        try:
            python_code: str = parse_code(action_args.get("code", ""))
            exec_res = self.action_set.execute_code(python_code, assign_to)
            used_table_ids = extract_table_ids_from_code(python_code)
            parent_nodes: list[ProvenanceNode] = []
            for used_table_id in used_table_ids:
                try:
                    used_table_id = used_table_id.split(".")[-1].strip('"')
                    used_table_doc = id_docs[used_table_id]
                    parent_node = self.prov_graph.get_node_by_id(
                        used_table_doc.last_node_id or ""
                    )
                    if parent_node is not None:
                        parent_nodes.append(parent_node)
                except Exception as e:
                    self._log(
                        f"Warning: Failed to extract retriever type or provenance node for used table ID {used_table_id}: {e}"
                    )
                    continue
            code_nl_summary = f"Executes Python code to produce a new table named {assign_to} by performing operations on the following tables: {', '.join(f'`{tid}`' for tid in used_table_ids)}. The code uses these tables as inputs and produces a new table as output."
            new_node = ProvenanceNode(
                source_retriever=RetrieverType.MATERIALIZER,
                python_code=self.action_set.append_comment_to_existing_code(
                    python_code, f"Result: {assign_to}"
                ),
                description=code_nl_summary,
            )
            self.prov_graph.add_node(new_node, True)
            for parent_node in parent_nodes:
                if parent_node is None:
                    continue
                try:
                    self.prov_graph.connect(parent_node, new_node)
                except Exception:
                    fallback = self.prov_graph.get_node_by_id(parent_node.id)
                    if fallback is not None:
                        try:
                            self.prov_graph.connect(fallback, new_node)
                        except Exception:
                            continue
            self.state.add_intermediate_table(
                Table(
                    doc_id=assign_to,
                    retriever_type=RetrieverType.MATERIALIZER,
                    content=exec_res,
                    metadata={},
                    last_node_id=new_node.id,
                )
            )
            success_msg = f"Successfully executed the Python code, resulting in a table named {assign_to}"
            self._log(f"==> {success_msg}")
            return success_msg, ActionExecutionStatus.SUCCESS
        except Exception as exception:
            error_msg = f"Exception occured during Python code execution: {exception}."
            self._log(error_msg)
            return error_msg, ActionExecutionStatus.ERROR

    def _handle_query_executor(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        all_tables = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )
        id_docs: dict[str, AbstractDocument] = {t.doc_id: t for t in all_tables}
        assign_to: str | None = action_args.get("assign_to")
        if assign_to is None or assign_to.strip() == "":
            error_msg = "'assign_to' argument is missing or empty."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        query = action_args.get("query")
        if not isinstance(query, str) or query.strip() == "":
            error_msg = "'query' argument is missing, empty, or not a string."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        try:
            exec_res = self.action_set.execute_query(query, assign_to)
            used_table_ids = extract_tables_from_sql_regex(query)
            parent_nodes: list[ProvenanceNode] = []
            for used_table_id in used_table_ids:
                try:
                    used_table_key = used_table_id.split(".")[-1].strip('"')
                    used_table_doc = id_docs.get(used_table_key)
                    if used_table_doc is None:
                        continue
                    parent_node = self.prov_graph.get_node_by_id(
                        used_table_doc.last_node_id or ""
                    )
                    if parent_node is not None:
                        parent_nodes.append(parent_node)
                except Exception as e:
                    self._log(
                        f"Warning: Failed to resolve provenance for used table ID {used_table_id}: {e}"
                    )
                    continue
            query_nl_summary = (
                f"Executes a SQL query to produce a new table named {assign_to} "
                f"from the following tables: {', '.join(f'`{tid}`' for tid in used_table_ids)}."
            )
            new_node = ProvenanceNode(
                source_retriever=RetrieverType.MATERIALIZER,
                python_code=self.action_set.append_comment_to_existing_code(
                    query, f"Result: {assign_to}"
                ),
                description=query_nl_summary,
            )
            self.prov_graph.add_node(new_node, True)
            for parent_node in parent_nodes:
                if parent_node is None:
                    continue
                try:
                    self.prov_graph.connect(parent_node, new_node)
                except Exception:
                    fallback = self.prov_graph.get_node_by_id(parent_node.id)
                    if fallback is not None:
                        try:
                            self.prov_graph.connect(fallback, new_node)
                        except Exception:
                            continue
            self.state.add_intermediate_table(
                Table(
                    doc_id=assign_to,
                    retriever_type=RetrieverType.MATERIALIZER,
                    content=exec_res,
                    metadata={},
                    last_node_id=new_node.id,
                )
            )
            success_msg = f"Successfully executed the SQL query, resulting in a table named {assign_to}"
            self._log(f"==> {success_msg}")
            return success_msg, ActionExecutionStatus.SUCCESS
        except Exception as exception:
            error_msg = f"Exception occured during SQL query execution: {exception}."
            self._log(error_msg)
            return error_msg, ActionExecutionStatus.ERROR

    def _handle_context_extraction(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        if not self.config.ENABLE_CONTEXT_EXTRACTION:
            error_msg = f"{ActionNames.CONTEXT_EXTRACTION.value} is not enabled in the configuration."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        # Backwards compat: old-style {"code": "..."} arg
        if "code" in action_args and "uncertainties" not in action_args:
            self._log("==> Falling back to legacy code-based context extraction.")
            try:
                python_code: str = parse_code(action_args.get("code", ""))
                exec_res = self.action_set.execute_code(
                    python_code, "materializer_assumption_check"
                )
                success_msg = (
                    f"Context extraction result: {dataframe_to_preview_str(exec_res)}"
                )
                self._log(f"==> {success_msg}")
                for cleanup_stmt in (
                    "DROP TABLE IF EXISTS materializer_assumption_check;",
                    "DROP VIEW IF EXISTS materializer_assumption_check;",
                ):
                    try:
                        self.db_api.execute_query(
                            self.user_id, self.chat_id, cleanup_stmt
                        )
                    except Exception as cleanup_exc:
                        self._log(f"Cleanup warning ({cleanup_stmt}): {cleanup_exc}")
                return success_msg, ActionExecutionStatus.SUCCESS
            except Exception as exception:
                error_msg = f"Error during context extraction: {exception}"
                self._log(error_msg)
                return error_msg, ActionExecutionStatus.ERROR

        uncertainties = action_args.get("uncertainties")
        if not isinstance(uncertainties, list) or len(uncertainties) == 0:
            error_msg = "=> `uncertainties` must be a non-empty list of {table_ids, question} objects"
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        available_tables = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )
        try:
            summary, log_msgs = self.action_set.run_context_extraction(
                uncertainties, available_tables, "materializer_assumption_check"
            )
            for msg in log_msgs:
                self.log_callback(msg)
            self._log(f"==> Context Extraction summary: {summary}")
            for cleanup_stmt in (
                "DROP VIEW IF EXISTS materializer_assumption_check;",
                "DROP TABLE IF EXISTS materializer_assumption_check;",
            ):
                try:
                    self.db_api.execute_query(self.user_id, self.chat_id, cleanup_stmt)
                except Exception as cleanup_exc:
                    self._log(f"Cleanup warning ({cleanup_stmt}): {cleanup_exc}")
            return (
                f"Context extraction result:\n{summary}",
                ActionExecutionStatus.SUCCESS,
            )
        except Exception as exception:
            error_msg = f"Error during context extraction: {exception}"
            self._log(error_msg)
            return error_msg, ActionExecutionStatus.ERROR

    def _handle_entity_resolution(
        self, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        source_table_id = action_args.get("source_table_id")
        target_column = action_args.get("target_column")
        output_mapping_table_id = action_args.get("output_mapping_table_id")
        canonical_entities = action_args.get("canonical_entities")

        if not isinstance(source_table_id, str) or not source_table_id.strip():
            error_msg = (
                "entity_resolution requires a non-empty string 'source_table_id'."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(target_column, str) or not target_column.strip():
            error_msg = "entity_resolution requires a non-empty string 'target_column'."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if (
            not isinstance(output_mapping_table_id, str)
            or not output_mapping_table_id.strip()
        ):
            error_msg = "entity_resolution requires a non-empty string 'output_mapping_table_id'."
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if canonical_entities is not None and (
            not isinstance(canonical_entities, list)
            or not all(isinstance(e, str) for e in canonical_entities)
        ):
            error_msg = (
                "entity_resolution 'canonical_entities' must be a list of strings."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        all_available = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )
        available_ids = sorted({t.doc_id for t in all_available})
        # Also accept schema-qualified forms like proc_spend."fy2025_gems_card_data"
        bare_id = source_table_id.split(".")[-1].strip('"')
        if source_table_id not in available_ids and bare_id not in available_ids:
            error_msg = (
                f"entity_resolution: source table '{source_table_id}' is not available. "
                f"Available tables: {available_ids}. "
                "For retrieved tables use the dataset-qualified form (e.g. 'dataset.\"table_id\"'). "
                "For tables not yet in scope, use query_executor to create an intermediate first."
            )
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        try:
            mapping_sample = self.action_set.resolve_entities(
                source_table_id=source_table_id,
                target_column=target_column,
                output_mapping_table_id=output_mapping_table_id,
                canonical_entities=canonical_entities,
            )
            new_node = ProvenanceNode(
                source_retriever=RetrieverType.MATERIALIZER,
                python_code="",
                description=(
                    f"Entity resolution on column `{target_column}` of table `{source_table_id}` "
                    f"→ mapping table `{output_mapping_table_id}`"
                ),
            )
            self.prov_graph.add_node(new_node, True)
            self.state.add_intermediate_table(
                Table(
                    doc_id=output_mapping_table_id,
                    retriever_type=RetrieverType.MATERIALIZER,
                    content=mapping_sample,
                    metadata={},
                    last_node_id=new_node.id,
                )
            )
            success_msg = (
                f"Successfully resolved entities in column '{target_column}' of table "
                f"'{source_table_id}'. Mapping table '{output_mapping_table_id}' is now available "
                "in intermediate tables. Join it back to your source table on `original_value` to "
                "get the canonical form."
            )
            self._log(f"==> {success_msg}")
            return success_msg, ActionExecutionStatus.SUCCESS
        except Exception as exc:
            error_msg = f"entity_resolution failed: {exc}"
            self._log(f"==> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

    def _create_or_get_read_node(
        self,
        doc: AbstractDocument,
        source: RetrieverType,
        python_code: str,
        description: str,
    ) -> str | None:
        try:
            last_id = getattr(doc, "last_node_id", None)
            if last_id is not None:
                existing = self.prov_graph.get_node_by_id(last_id)
                if existing is not None:
                    return last_id

            read_node = ProvenanceNode(
                source_retriever=source,
                python_code=python_code,
                description=description,
            )
            self.prov_graph.add_node(read_node, True)
            return read_node.id
        except Exception:
            # On any error, do not crash materializer; return None so caller can handle
            return None

    def _check_completion(self, T: dict[str, DataFrame]) -> bool:
        """Check if all target tables (T) have been materialized correctly."""
        self._log("Checking completion...")
        all_T_ids = set(T.keys())
        id_dfs: dict[str, DataFrame] = {}

        materialized_table_ids: set[str] = set()
        for doc in self.state.intermediate_tables:
            if isinstance(doc.content, DataFrame):
                id_dfs[doc.doc_id] = doc.content
                materialized_table_ids.add(doc.doc_id)
            else:
                self._log(
                    f"=> Warning: doc {doc.doc_id} has invalid content type {type(doc.content)}"
                )

        self._log(f"=> all_T_ids: {all_T_ids}")
        self._log(f"=> materialized_table_ids: {materialized_table_ids}")
        ids_complete = all_T_ids <= materialized_table_ids
        is_complete = ids_complete

        warning_messages: list[str] = []
        if not ids_complete:
            warning_msg = f"You have not materialized these tables: {all_T_ids - materialized_table_ids}"
            self._log(f"=> {warning_msg}")
            warning_messages.append(warning_msg)

        column_issues: list[str] = []
        if ids_complete:
            for target_table_id in all_T_ids:
                df = id_dfs.get(target_table_id)
                if df is None or not isinstance(df, DataFrame):
                    issue_msg = f"- For table `{target_table_id}`: no valid table was materialized."
                    self._log(issue_msg)
                    column_issues.append(issue_msg)
                    is_complete = False
                    continue

                self._log(f"=> Checking the target schema {target_table_id}.")
                target_cols = set(T[target_table_id].columns)
                materialized_cols = set(df.columns)

                self._log(f"==> target_cols {target_cols}")
                self._log(f"==> materialized_cols {materialized_cols}")

                missing_cols = target_cols - materialized_cols
                extra_cols = materialized_cols - target_cols

                # Treat the target schema as a *required subset* of the materialized schema.
                # Extra columns are allowed and should not trigger a "repair" projection that
                # could drop useful data (e.g., wide sample columns like s001..s153).
                if missing_cols:
                    is_complete = False
                    issue_msg = f"- For table `{target_table_id}`: missing columns {sorted(missing_cols)}."
                    column_issues.append(issue_msg)

                if extra_cols:
                    self._log(
                        f"==> Note: `{target_table_id}` has extra columns (allowed): {sorted(extra_cols)}"
                    )

        if ids_complete and not is_complete:
            warning_msg = f"There are issues with the columns of materialized tables: {'\n'.join(column_issues)}"
            self._log(f"=> {warning_msg}")
            warning_messages.append(warning_msg)

        if warning_messages:
            self.llm_messages.append(
                LLMMessage(
                    role=Role.USER.value,
                    content="\n".join(warning_messages),
                )
            )
        self._log(
            f"Completion check: {is_complete} ({len(materialized_table_ids)}/{len(all_T_ids)} tables materialized)"
        )
        return is_complete

    def _reset_materializer(self, update_mode: bool = False):
        """Reset the state and clear intermediate files.

        In update_mode the intermediate tables from the prior run are kept in the
        database (STATE_MANIPULATION no longer wipes T tables) and pre-loaded into
        the fresh state so the LLM can build on them incrementally.
        """
        self._log(f"Resetting materializer (update_mode={update_mode})...")
        saved = list(self._saved_intermediate_tables)

        self.state.reset()
        if not update_mode:
            self.prov_graph.reset_materialization_nodes()
        self.actions = []
        self.llm_messages = []
        self.join_paths = None

        if not update_mode:
            for doc_id in self.last_intermediate_table_ids:
                try:
                    self.db_api.execute_query(
                        self.user_id,
                        self.chat_id,
                        f'DROP TABLE IF EXISTS "{doc_id}";',
                    )
                except Exception as e:
                    self._log(
                        f"Warning: failed to delete intermediate table {doc_id} from the database: {e}"
                    )
                    continue
            self.last_intermediate_table_ids = set()
        else:
            # Pre-populate state with intermediates from the prior run.
            # Because STATE_MANIPULATION is now a pure in-memory operation, these tables
            # are still present in the database and can be used directly.
            for doc in saved:
                self.state.add_intermediate_table(doc)

        self._log("Materializer reset complete.")

    def _log_step_profiling(
        self, input_tokens: int, output_tokens: int, total_time: float, llm_time: float
    ) -> None:
        self._log(
            f"[STEP PROFILING] Time taken: {total_time:.2f}s (CPU: {total_time - llm_time:.2f}s, LLM: {llm_time:.2f}s)"
        )
        self._log(
            f"[STEP PROFILING] Input tokens: {input_tokens}, Output tokens: {output_tokens}"
        )

    def _log_overall_profiling(
        self,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        llm_time: float,
    ) -> None:
        total_time = end_time - start_time
        self._log(
            f"[OVERALL PROFILING] Total Time taken: {total_time:.2f}s (CPU: {total_time - llm_time:.2f}s, LLM: {llm_time:.2f}s)"
        )
        self._log(
            f"[OVERALL PROFILING] Total Input tokens: {input_tokens}, Total Output tokens: {output_tokens}"
        )

    def _log_table_repr_tokens(self, tables: list[AbstractDocument]) -> None:
        text = convert_retrieval_results_to_str(tables)
        try:
            enc = encoding_for_model("o4-mini")
            n = len(enc.encode(text))
            self._log(
                f"[PROFILING] Retrieved tables repr: ~{n} tokens (tiktoken approx)"
            )
        except Exception:
            n = len(text.split())
            self._log(
                f"[PROFILING] Retrieved tables repr: ~{n} tokens (whitespace approx)"
            )

    def _log(self, text: str):
        formatted_log(self.logger, "Materializer", text)
