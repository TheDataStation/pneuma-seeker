# src/pneuma_seeker/core/materializer/main.py
from collections import deque
from json import dumps
from logging import Logger
from typing import Any

from pandas import DataFrame

from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.impl.semantic_join import (
    SyntacticSimMetric,
)
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
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
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.prov_graph = prov_graph
        self.action_set = action_set
        self.db_api = db_api
        self.language_model_api = language_model_api

        self.__log(
            f"Initializing Materializer for user_id: {self.user_id}, chat_id: {self.chat_id}"
        )

        self.prompt_factory = MaterializerPromptFactory(self.config, self.action_set)
        self.state = MaterializerState(self.user_id, self.chat_id, self.db_api)
        self.actions: list[str] = []
        self.llm_messages: list[LLMMessage] = []
        self.last_intermediate_table_ids: set[str] = set()

        self.premade_plans = deque()

    def materialize_T(
        self,
        T: dict[str, DataFrame],
        column_descriptions: dict[str, dict[str, str]],
        S: str,
        client_note="",
        external_tables: list[AbstractDocument] = [],
        prefetched_tables: list[AbstractDocument] = [],
        prefetched_web_search_result: AbstractDocument | None = None,
        prefetched_web_crawl_result: AbstractDocument | None = None,
        precomputed_join_paths: str | None = None,
    ) -> tuple[
        list[AbstractDocument],  # Retrieved tables
        AbstractDocument | None,  # Web search result
        AbstractDocument | None,  # Web crawl result
        str | None,  # Join paths
        dict[str, DataFrame],  # Materialized T
    ]:
        """Materialize target tables T based on the provided script S and external tables."""
        self.__log(f"Materializing {len(T)} target tables...")
        self.__reset_materializer()

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
                ),
            )
        ]

        current_step = 0
        last_env_state_idx: int | None = None
        while (
            not self.__check_completion(self.state.T)
            and current_step < self.config.MAX_MATERIALIZER_STEPS
        ):
            self.__log(
                f"=> [Step {current_step}/{self.config.MAX_MATERIALIZER_STEPS}] Planning materialization actions..."
            )
            current_step += 1

            self.llm_messages.append(
                LLMMessage(
                    role=Role.USER.value,
                    content=self.prompt_factory.get_context_prompt(
                        self.state.retrieved_tables,
                        list(self.state.intermediate_tables),
                        self.actions[-5:],  # only include last 5 actions for brevity
                        current_step,
                        client_note,
                        self.state.external_tables,
                        self.state.web_search_result,
                        self.state.web_crawl_result,
                        self.state.join_paths,
                    ),
                )
            )
            last_env_state_idx = len(self.llm_messages) - 1

            if self.premade_plans:
                llm_response = self.premade_plans.popleft()
            else:
                llm_response = "".join(
                    self.language_model_api.chat(
                        self.llm_messages,
                        LLMOption(json_mode=True, max_new_tokens=2500),
                    )
                )
            self.llm_messages.append(
                LLMMessage(
                    role=Role.ASSISTANT.value,
                    content=llm_response,
                )
            )
            if last_env_state_idx is not None:
                self.llm_messages[last_env_state_idx]["content"] = (
                    self.prompt_factory.get_skeleton_context_prompt(
                        current_step,
                    )
                )

            try:
                self.__log("==> Parsing the response...")
                plan: list[dict[str, Any]] = parse_json(llm_response).get("plan", [])

                if not isinstance(plan, list):
                    raise ValueError("The 'plan' field must be a list.")
                if len(plan) == 0:
                    raise ValueError("The 'plan' list cannot be empty.")
                if not all(isinstance(item, dict) for item in plan):
                    raise ValueError(
                        "All items in the 'plan' list must be JSON objects."
                    )

                for action_plan in plan:
                    if not isinstance(action_plan, dict):
                        raise ValueError(
                            "Each action in the plan must be a JSON object."
                        )
                    if "action" not in action_plan:
                        raise ValueError(
                            "Each action in the plan must have an 'action' field."
                        )
                    if not self.action_set.is_valid_materializer_action(
                        action_plan["action"]
                    ):
                        raise ValueError(
                            f"Invalid action '{action_plan['action']}' specified."
                        )
                self.actions.append(str(plan))
            except ValueError as exc:
                error_msg = f"Error parsing the the plan: {exc}."
                self.__log(f"==> {error_msg}")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=error_msg,
                    )
                )
                current_step -= 1
                continue

            self.__log(f"==> Executing the planned actions: {plan}...")
            for action_plan in plan:
                action_name: str = action_plan.get("action", "")
                action_args: dict[str, Any] = action_plan.get("args", {})
                self.__execute_action(action_name, action_args)

        self.__log("Materialization completed successfully!")
        self.last_intermediate_table_ids = set(
            i.doc_id for i in self.state.intermediate_tables
        )
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

    def __execute_action(
        self,
        action_name: str,
        action_args: dict[str, Any],
    ):
        all_tables = (
            self.state.retrieved_tables
            + self.state.external_tables
            + list(self.state.intermediate_tables)
        )

        self.__log(f"==> Executing action {action_name}...")
        match action_name:
            case ActionNames.SITUATIONAL_ANALYSIS.value:
                message: str = action_args.get("message", "")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=f"You did a situational analysis: {message}",
                    )
                )
            case ActionNames.TABLE_RETRIEVE.value:
                prompts = action_args.get("prompts", [])
                if not isinstance(prompts, list) or not all(
                    isinstance(p, str) for p in prompts
                ):
                    error_msg = "The 'prompts' argument must be a list of strings."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if len(prompts) == 0:
                    error_msg = "The 'prompts' list cannot be empty."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                self.state.retrieved_tables = (
                    self.action_set.retrieve_multi_topic_documents(
                        prompts, RetrieverType.PNEUMA_RETRIEVER, 10, True, 5
                    )
                )

                if len(self.state.retrieved_tables) == 0:
                    error_msg = f"No tables were retrieved."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                else:
                    self.state.join_paths = self.action_set.discover_join_paths(
                        self.state.retrieved_tables
                    )
                    success_msg = f'Successfully retrieved tables. Notice that the "retrieved internal tables" have been filled.'
                    self.__log(f"==> {success_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=success_msg,
                        )
                    )

                    self.__log(
                        f"Retrieved tables:\n {[i.doc_id for i in self.state.retrieved_tables]}"
                    )
                for doc in self.state.retrieved_tables:
                    if doc.path is not None:
                        node_id = self.__create_or_get_read_node(
                            doc,
                            RetrieverType.PNEUMA_RETRIEVER,
                            self.action_set.generate_pandas_read_csv_code(doc),
                            "Retrieves an internal table from Pneuma-Retriever.",
                        )
                        if node_id is not None:
                            doc.last_node_id = node_id
            case ActionNames.WEB_SEARCH.value:
                if not self.config.ENABLE_WEB_SEARCH:
                    error_msg = f"{ActionNames.WEB_SEARCH.value} is not enabled in the configuration."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                prompt = action_args.get("prompt", "")
                web_search_results = self.action_set.retrieve_documents(
                    prompt, RetrieverType.WEB_SEARCH
                )
                if len(web_search_results) == 0:
                    error_msg = f"No relevant information was found from {ActionNames.WEB_SEARCH.value}."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                self.state.web_search_result = web_search_results[0]
                success_msg = f'Successfully retrieved information from {ActionNames.WEB_SEARCH.value}. Notice that the "{ActionNames.WEB_SEARCH.value} result" have been filled. Join paths have also been updated accordingly."'
                self.__log(f"==> {success_msg}")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=success_msg,
                    )
                )

                new_node = ProvenanceNode(
                    source_retriever=RetrieverType.WEB_SEARCH,
                    python_code=self.action_set.generate_view_textual_document_code(
                        self.state.web_search_result
                    ),
                    description=f"Searches the web using this query:\n{prompt}",
                )
                self.prov_graph.add_node(new_node, True)
                self.state.web_search_result.last_node_id = new_node.id
            case ActionNames.WEB_CRAWL.value:
                if not self.config.ENABLE_WEB_CRAWL:
                    error_msg = f"{ActionNames.WEB_CRAWL.value} is not enabled in the configuration."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                prompt = action_args.get("url", "")
                web_crawl_results = self.action_set.retrieve_documents(
                    prompt, RetrieverType.WEB_CRAWL
                )
                if len(web_crawl_results) == 0:
                    error_msg = f"No relevant information was found from {ActionNames.WEB_CRAWL.value}."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                self.state.web_crawl_result = web_crawl_results[0]
                success_msg = f'Successfully retrieved information from {ActionNames.WEB_CRAWL.value}. Notice that the "{ActionNames.WEB_CRAWL.value} result" have been filled.'
                self.__log(f"==> {success_msg}")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=success_msg,
                    )
                )

                new_node = ProvenanceNode(
                    source_retriever=RetrieverType.WEB_CRAWL,
                    python_code=self.action_set.generate_view_textual_document_code(
                        self.state.web_crawl_result
                    ),
                    description=f"Crawls this web page:\n{prompt}",
                )
                self.prov_graph.add_node(new_node, True)
                self.state.web_crawl_result.last_node_id = new_node.id
            case ActionNames.TABLE_ENUMERATION.value:
                patterns = action_args.get("patterns", [])
                if not isinstance(patterns, list) or not all(
                    isinstance(p, str) for p in patterns
                ):
                    error_msg = "The 'patterns' argument must be a list of strings."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if len(patterns) == 0:
                    error_msg = "The 'patterns' list cannot be empty."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                extra_tables: list[AbstractDocument] = (
                    self.action_set.retrieve_multi_topic_documents(
                        patterns, RetrieverType.ENUMERATOR, 20, True, 5
                    )
                )
                pattern_desc = str(patterns)

                if len(extra_tables) > 0:
                    success_msg = f'Successfully retrieved all tables that match the pattern(s) {pattern_desc}. You can use them to materialize T, even if you have not called {ActionNames.TABLE_RETRIEVE.value} before, as these tables have been included to "retrieved internal tables".'
                    self.__log(f"==> {success_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=success_msg,
                        )
                    )

                    read_code = self.action_set.generate_pandas_read_multi_doc_code(
                        extra_tables
                    )
                    new_node = ProvenanceNode(
                        source_retriever=RetrieverType.ENUMERATOR,
                        python_code=read_code,
                        description=f"Enumerates all tables whose names match this regular expression (RegEx) pattern:\n{pattern_desc}.",
                    )
                    self.prov_graph.add_node(new_node, True)

                    for extra_table in extra_tables:
                        # If extra_table already has a last_node_id that exists in graph,
                        # prefer reusing it; otherwise set to this multi-read node.
                        last_id = getattr(extra_table, "last_node_id", None)
                        if last_id is not None:
                            existing_node = self.prov_graph.get_node_by_id(last_id)
                            if existing_node is not None:
                                continue
                        extra_table.last_node_id = new_node.id

                    existing_tables = self.state.retrieved_tables
                    self.state.retrieved_tables = list(
                        set(existing_tables).union(set(extra_tables))
                    )
                    self.__log(
                        f"Retrieved tables:\n {[i.doc_id for i in self.state.retrieved_tables]}"
                    )
                else:
                    error_msg = "There are no tables that match the pattern."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
            case ActionNames.TABLE_PROJECTION.value:
                all_table_doc_ids = [i.doc_id for i in all_tables]
                for target_table_id, retrieved_table_info in action_args.items():
                    # BEGIN INPUT VALIDATION
                    if isinstance(retrieved_table_info, list):
                        if len(retrieved_table_info) == 0:
                            msg = f"Skipping {target_table_id!r}: empty list provided as value."
                            self.__log(f"==> {msg}")
                            self.llm_messages.append(
                                LLMMessage(
                                    role=Role.USER.value,
                                    content=msg,
                                )
                            )
                            continue
                        retrieved_table_info = retrieved_table_info[0]

                    if not isinstance(retrieved_table_info, dict):
                        msg = f"Invalid argument for target {target_table_id!r}: expected a dict."
                        self.__log(f"==> {msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=msg,
                            )
                        )
                        continue

                    table_id_to_project = str(
                        retrieved_table_info.get("id", "")
                    ).strip()
                    if table_id_to_project.startswith("Table "):
                        table_id_to_project = table_id_to_project[6:].strip()
                    column_mapping = retrieved_table_info.get("columns", {})
                    target_table_id = target_table_id.strip()

                    if not isinstance(column_mapping, dict):
                        error_msg = "Invalid 'columns' for table_projection: expected a dict mapping output column names to source column names."
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return
                    if len(column_mapping) == 0:
                        error_msg = "Invalid 'columns' for table_projection: empty mapping provided."
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return
                    if not all(
                        isinstance(out_col, str) and isinstance(src_col, str)
                        for out_col, src_col in column_mapping.items()
                    ):
                        error_msg = "Invalid 'columns' for table_projection: mapping must be str->str."
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return

                    output_columns = list(column_mapping.keys())
                    source_columns = list(column_mapping.values())

                    if (
                        table_id_to_project.split(".")[-1].strip('"')
                        not in all_table_doc_ids
                    ):
                        error_msg = (
                            "Invalid table ID to select. Ensure the table exists."
                        )
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return

                    if target_table_id not in self.state.T:
                        error_msg = (
                            f"Error: The ID {target_table_id} does not exist in T."
                        )
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return

                    matches = [
                        i
                        for i in all_tables
                        if i.doc_id == table_id_to_project.split(".")[-1].strip('"')
                    ]
                    if not matches:
                        error_msg = f"Table {table_id_to_project.split('.')[-1].strip('"')!r} not found in the available tables."
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return
                    # END INPUT VALIDATION

                    self.__log(
                        f"==> target_table_id: {target_table_id}; table_id_to_project: {table_id_to_project}"
                    )
                    try:
                        projected_table_sample_rows = self.action_set.project_table(
                            table_id_to_project, target_table_id, column_mapping
                        )
                    except Exception as e:
                        error_msg = f"Failed projecting columns {column_mapping!r} from table {table_id_to_project!r}: {e}"
                        self.__log(f"==> {error_msg}")
                        self.llm_messages.append(
                            LLMMessage(
                                role=Role.USER.value,
                                content=error_msg,
                            )
                        )
                        return

                    # Always create a materializer node representing the select operation
                    select_code = self.action_set.generate_table_select_code(
                        target_table_id,
                        table_id_to_project,
                        source_columns,
                    )

                    # Try to create/get a read node for the source doc to connect from
                    parent_node_id = self.__create_or_get_read_node(
                        matches[0],
                        matches[0].retriever_type,
                        self.action_set.generate_pandas_read_csv_code(matches[0]),
                        "",
                    )
                    child_node_desc = (
                        f"Projects a table\n\n"
                        f"- ID: `{table_id_to_project}`\n"
                        f"- columns: {', '.join(f'`{col}`' for col in output_columns)}\n\n"
                        f"into a target table:\n\n"
                        f"`{target_table_id}`"
                    )
                    if set(output_columns) != set(
                        self.state.T[target_table_id].columns
                    ):
                        child_node_desc += " (partially)."
                    else:
                        child_node_desc += "."

                    child_node = ProvenanceNode(
                        source_retriever=RetrieverType.MATERIALIZER,
                        python_code=select_code,
                        description=child_node_desc,
                    )

                    self.prov_graph.add_node(child_node, True)
                    new_node_id = child_node.id
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
                            last_node_id=new_node_id,
                        )
                    )
                    success_msg = "Successfully projected retrieved tables in the mapping as target tables. Notice the state's intermediate tables have changed, but please CHECK if the schemas in the selected tables match, either fully or partially, with the ones in target tables."
                    self.__log(f"==> {success_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=success_msg,
                        )
                    )
            case ActionNames.EQUALITY_JOIN.value:
                left_table_id = action_args.get("left_table_id")
                right_table_id = action_args.get("right_table_id")
                left_keys = action_args.get("left_table_column_keys")
                right_keys = action_args.get("right_table_column_keys")
                result_table_id = action_args.get("result_table_id")

                if not isinstance(left_table_id, str) or not isinstance(
                    right_table_id, str
                ):
                    error_msg = "Invalid equality_join args: left_table_id and right_table_id must be strings."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return
                if not isinstance(left_keys, list) or not all(
                    isinstance(x, str) for x in left_keys
                ):
                    error_msg = "Invalid equality_join args: left_table_column_keys must be a list[str]."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return
                if not isinstance(right_keys, list) or not all(
                    isinstance(x, str) for x in right_keys
                ):
                    error_msg = "Invalid equality_join args: right_table_column_keys must be a list[str]."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return
                if not isinstance(result_table_id, str):
                    error_msg = (
                        "Invalid equality_join args: result_table_id must be a string."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return

                try:
                    joined_preview = self.action_set.join_equality(
                        left_table_id,
                        right_table_id,
                        left_keys,
                        right_keys,
                        result_table_id,
                    )
                except Exception as e:
                    error_msg = f"Failed equality_join: {e}"
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return

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

                success_msg = "Successfully performed equality join and materialized the result table."
                self.__log(f"==> {success_msg}")
                self.llm_messages.append(
                    LLMMessage(role=Role.USER.value, content=success_msg)
                )
            case ActionNames.TABLE_UNION.value:
                table_ids = action_args.get("table_ids")
                result_table_id = action_args.get("result_table_id")
                provenance_regex = action_args.get("provenance_regex")
                provenance_column_name = action_args.get("provenance_column_name")

                if not isinstance(table_ids, list) or not all(
                    isinstance(x, str) for x in table_ids
                ):
                    error_msg = (
                        "Invalid table_union args: table_ids must be a list[str]."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return
                if not isinstance(result_table_id, str):
                    error_msg = (
                        "Invalid table_union args: result_table_id must be a string."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return
                if (
                    not isinstance(provenance_column_name, str)
                    or not provenance_column_name
                ):
                    error_msg = "Invalid table_union args: provenance_column_name must be a non-empty string."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return
                if not isinstance(provenance_regex, str) or not provenance_regex:
                    error_msg = "Invalid table_union args: provenance_regex must be a non-empty string."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return

                try:
                    union_preview = self.action_set.union_tables(
                        table_ids=table_ids,
                        result_table_id=result_table_id,
                        provenance_column_name=str(provenance_column_name),
                        provenance_regex=str(provenance_regex),
                    )
                except Exception as e:
                    error_msg = f"Failed table_union: {e}"
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(role=Role.USER.value, content=error_msg)
                    )
                    return

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

                success_msg = (
                    "Successfully unioned tables and materialized the result table."
                )
                self.__log(f"==> {success_msg}")
                self.llm_messages.append(
                    LLMMessage(role=Role.USER.value, content=success_msg)
                )
            case ActionNames.SEMANTIC_COLUMN_GENERATION.value:
                table_id: str | None = action_args.get("table_id")
                new_column_name: str | None = action_args.get("new_column_name")
                src_table_columns: list[str] | None = action_args.get(
                    "relevant_columns"
                )
                instruction: str | None = action_args.get("instruction")

                if table_id is None or table_id not in [
                    i.doc_id for i in self.state.intermediate_tables
                ]:
                    error_msg = (
                        "table_id is not valid (not part of intermediate tables)."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if new_column_name is None:
                    error_msg = "new_column_name is not provided."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if src_table_columns is None:
                    error_msg = "relevant_columns is not provided."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                conditioned_table_doc = [
                    i for i in self.state.intermediate_tables if i.doc_id == table_id
                ][0]
                if conditioned_table_doc.retriever_type != RetrieverType.MATERIALIZER:
                    error_msg = "Semantic column generation is only supported for intermediate tables generated within the materialization process."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                conditioned_table_sample_rows: DataFrame = conditioned_table_doc.content

                if not set(src_table_columns) <= set(
                    list(conditioned_table_sample_rows.columns)
                ):
                    error_msg = f"relevant_columns must be a subset of the columns of table {table_id}."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if instruction is None:
                    error_msg = "instruction is not provided."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                augmented_table_sample_rows = self.action_set.generate_semantic_column(
                    conditioned_table_doc.doc_id,
                    src_table_columns,
                    new_column_name,
                    instruction,
                )
                conditioned_table_doc.content = augmented_table_sample_rows
                success_msg = f"Successfully added a new column named {new_column_name} to table with ID {table_id}."
                self.__log(f"==> {success_msg}")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=success_msg,
                    )
                )

                sem_col_code = self.action_set.generate_semantic_col_generator_code(
                    src_table_columns,
                    conditioned_table_doc,
                    new_column_name,
                    list(augmented_table_sample_rows[new_column_name]),
                )
                # Ensure we have a parent node for the conditioned table (read node)
                parent_node_id = self.__create_or_get_read_node(
                    conditioned_table_doc,
                    conditioned_table_doc.retriever_type,
                    self.action_set.generate_pandas_read_csv_code(
                        conditioned_table_doc
                    ),
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
            case ActionNames.SEMANTIC_JOIN.value:
                left_table_id: str | None = action_args.get("left_table_id")
                right_table_id: str | None = action_args.get("right_table_id")
                relevant_left_cols: list[str] | None = action_args.get(
                    "relevant_left_cols"
                )
                relevant_right_cols: list[str] | None = action_args.get(
                    "relevant_right_cols"
                )
                joined_table_id: str | None = action_args.get("joined_table_id")

                all_table_ids = [i.doc_id for i in all_tables]
                if left_table_id is None or left_table_id not in all_table_ids:
                    error_msg = "left_table_id is not valid (not part of retrieved tables or the state's intermediate tables)."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if right_table_id is None or right_table_id not in all_table_ids:
                    error_msg = "right_table_id is not valid (not part of retrieved tables or the state's intermediate tables)."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

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

                if not isinstance(left_table, DataFrame):
                    error_msg = (
                        f"left_table with ID {left_table_id} is not a DataFrame."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if not isinstance(right_table, DataFrame):
                    error_msg = (
                        f"right_table with ID {right_table_id} is not a DataFrame."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                if not isinstance(left_table_doc, AbstractDocument):
                    error_msg = f"ID {left_table_id} does not correspond to a document."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if not isinstance(right_table_doc, AbstractDocument):
                    error_msg = (
                        f"ID {right_table_id} does not correspond to a document."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                if relevant_left_cols is None:
                    error_msg = "relevant_left_cols is not provided."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if relevant_right_cols is None:
                    error_msg = "relevant_right_cols is not provided."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if not set(relevant_left_cols) <= set(list(left_table.columns)):
                    error_msg = (
                        "relevant_left_cols is not a subset of left_table's columns."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if not set(relevant_right_cols) <= set(list(right_table.columns)):
                    error_msg = (
                        "relevant_right_cols is not a subset of right_table's columns."
                    )
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                if not joined_table_id:
                    error_msg = "joined_table_id is not provided."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                joined_table_sample_rows = self.action_set.join_semantic(
                    left_table_id,
                    right_table_id,
                    relevant_left_cols,
                    relevant_right_cols,
                    joined_table_id,
                    syntactic_sim_metric=SyntacticSimMetric.JACCARD_QGRAM,
                    top_k=self.config.SEMANTIC_JOIN_TOP_K,
                )

                join_code = self.action_set.generate_semantic_join_generator_code(
                    left_table_doc,
                    right_table_doc,
                    relevant_left_cols,
                    relevant_right_cols,
                    self.config.SEMANTIC_JOIN_TOP_K,
                )
                parent_node_1_id = self.__create_or_get_read_node(
                    left_table_doc,
                    left_table_doc.retriever_type,
                    self.action_set.generate_pandas_read_csv_code(left_table_doc),
                    "",
                )
                parent_node_2_id = self.__create_or_get_read_node(
                    right_table_doc,
                    right_table_doc.retriever_type,
                    self.action_set.generate_pandas_read_csv_code(right_table_doc),
                    "",
                )

                new_node = ProvenanceNode(
                    source_retriever=RetrieverType.MATERIALIZER,
                    python_code=join_code,
                    description=f"Semantically joins `{left_table_id}` and `{right_table_id}`. For each row in the left table, keeps the `top-{self.config.SEMANTIC_JOIN_TOP_K}` matches from the right table (i.e., the most similar rows based on syntactic and semantic similarity). Similarity is computed based on these columns:\n- Left table: {', '.join(f'`{col}`' for col in relevant_left_cols)}\n- Right table: {', '.join(f'`{col}`' for col in relevant_right_cols)}",
                )
                self.prov_graph.add_node(new_node, True)
                if parent_node_1_id is not None:
                    parent_node_1 = self.prov_graph.get_node_by_id(parent_node_1_id)
                    if parent_node_1 is not None:
                        self.prov_graph.connect(parent_node_1, new_node)
                if parent_node_2_id is not None:
                    parent_node_2 = self.prov_graph.get_node_by_id(parent_node_2_id)
                    if parent_node_2 is not None:
                        self.prov_graph.connect(parent_node_2, new_node)

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
                self.__log(f"==> {success_msg}")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=success_msg,
                    )
                )
            case ActionNames.PYTHON_EXECUTOR.value:
                id_docs: dict[str, AbstractDocument] = {}
                for table_doc in all_tables:
                    id_docs[table_doc.doc_id] = table_doc
                assign_to: str | None = action_args.get("assign_to")
                if assign_to is None or assign_to.strip() == "":
                    error_msg = "'assign_to' argument is missing or empty."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                try:
                    python_code: str = parse_code(action_args.get("code", ""))
                    exec_res = self.action_set.execute_code(python_code, assign_to)
                    used_table_ids = extract_table_ids_from_code(python_code)

                    used_table_retrievers: list[RetrieverType] = []
                    parent_nodes: list[ProvenanceNode] = []
                    for used_table_id in used_table_ids:
                        try:
                            used_table_id = used_table_id.split(".")[-1].strip('"')
                            used_table_retrievers.append(
                                id_docs[used_table_id].retriever_type
                            )

                            used_table_doc = id_docs[used_table_id]
                            parent_node = self.prov_graph.get_node_by_id(
                                used_table_doc.last_node_id or ""
                            )
                            if parent_node is not None:
                                parent_nodes.append(parent_node)
                        except Exception as e:
                            self.__log(
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
                        # parent_node is already a ProvenanceNode instance
                        if parent_node is None:
                            continue
                        try:
                            self.prov_graph.connect(parent_node, new_node)
                        except Exception:
                            # best-effort: try to resolve by id and connect if present
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
                    self.__log(f"==> {success_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=success_msg,
                        )
                    )
                except Exception as exception:
                    error_msg = (
                        f"Exception occured during Python code execution: {exception}."
                    )
                    self.__log(error_msg)
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
            case ActionNames.QUERY_EXECUTOR.value:
                id_docs: dict[str, AbstractDocument] = {}
                for table_doc in all_tables:
                    id_docs[table_doc.doc_id] = table_doc

                assign_to: str | None = action_args.get("assign_to")
                if assign_to is None or assign_to.strip() == "":
                    error_msg = "'assign_to' argument is missing or empty."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

                query = action_args.get("query")
                if not isinstance(query, str) or query.strip() == "":
                    error_msg = "'query' argument is missing, empty, or not a string."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return

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
                            self.__log(
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
                    self.__log(f"==> {success_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=success_msg,
                        )
                    )
                except Exception as exception:
                    error_msg = (
                        f"Exception occured during SQL query execution: {exception}."
                    )
                    self.__log(error_msg)
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
            case ActionNames.CONTEXT_EXTRACTION.value:
                if not self.config.ENABLE_CONTEXT_EXTRACTION:
                    error_msg = f"{ActionNames.CONTEXT_EXTRACTION.value} is not enabled in the configuration."
                    self.__log(f"==> {error_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
                    return
                try:
                    python_code: str = parse_code(action_args.get("code", ""))
                    exec_res = self.action_set.execute_code(
                        python_code, "materializer_assumption_check"
                    )
                    success_msg = f"Assumption check result: {exec_res}"
                    self.__log(f"==> {success_msg}")
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=success_msg,
                        )
                    )
                    # Assumption checks may materialize either a *table* or a *view*.
                    # In DuckDB, attempting to DROP the wrong object type throws.
                    # Cleanup must be best-effort and must not turn a successful
                    # assumption_check into a failure.
                    for cleanup_stmt in (
                        "DROP VIEW IF EXISTS materializer_assumption_check;",
                        "DROP TABLE IF EXISTS materializer_assumption_check;",
                    ):
                        try:
                            self.db_api.execute_query(
                                self.user_id, self.chat_id, cleanup_stmt
                            )
                        except Exception as cleanup_exc:
                            self.__log(
                                f"Assumption check cleanup warning ({cleanup_stmt}): {cleanup_exc}"
                            )
                except Exception as exception:
                    error_msg = f"Error during assumption checking: {exception}"
                    self.__log(error_msg)
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=error_msg,
                        )
                    )
            case _:
                error_msg = f"{action_name} is not a valid action."
                self.__log(f"==> {error_msg}")
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=error_msg,
                    )
                )

    def __create_or_get_read_node(
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

    def __check_completion(self, T: dict[str, DataFrame]) -> bool:
        """Check if all target tables (T) have been materialized correctly."""
        self.__log("Checking completion...")
        all_T_ids = set(T.keys())
        id_dfs: dict[str, DataFrame] = {}

        materialized_table_ids: set[str] = set()
        for doc in self.state.intermediate_tables:
            if isinstance(doc.content, DataFrame):
                id_dfs[doc.doc_id] = doc.content
                materialized_table_ids.add(doc.doc_id)
            else:
                self.__log(
                    f"=> Warning: doc {doc.doc_id} has invalid content type {type(doc.content)}"
                )

        self.__log(f"=> all_T_ids: {all_T_ids}")
        self.__log(f"=> materialized_table_ids: {materialized_table_ids}")
        ids_complete = all_T_ids <= materialized_table_ids
        is_complete = ids_complete

        warning_messages: list[str] = []
        if not ids_complete:
            warning_msg = f"You have not materialized these tables: {all_T_ids - materialized_table_ids}"
            self.__log(f"=> {warning_msg}")
            warning_messages.append(warning_msg)

        column_issues: list[str] = []
        if ids_complete:
            for target_table_id in all_T_ids:
                df = id_dfs.get(target_table_id)
                if df is None or not isinstance(df, DataFrame):
                    issue_msg = f"- For table `{target_table_id}`: no valid table was materialized."
                    self.__log(issue_msg)
                    column_issues.append(issue_msg)
                    is_complete = False
                    continue

                self.__log(f"=> Checking the target schema {target_table_id}.")
                target_cols = set(T[target_table_id].columns)
                materialized_cols = set(df.columns)

                self.__log(f"==> target_cols {target_cols}")
                self.__log(f"==> materialized_cols {materialized_cols}")

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
                    self.__log(
                        f"==> Note: `{target_table_id}` has extra columns (allowed): {sorted(extra_cols)}"
                    )

        if ids_complete and not is_complete:
            warning_msg = f"There are issues with the columns of materialized tables: {'\n'.join(column_issues)}"
            self.__log(f"=> {warning_msg}")
            warning_messages.append(warning_msg)

        if warning_messages:
            self.llm_messages.append(
                LLMMessage(
                    role=Role.USER.value,
                    content="\n".join(warning_messages),
                )
            )
        self.__log(
            f"Completion check: {is_complete} ({len(materialized_table_ids)}/{len(all_T_ids)} tables materialized)"
        )
        return is_complete

    def __reset_materializer(self):
        """Reset the state and clear intermediate files."""
        self.__log("Resetting materializer...")
        self.state.reset()
        self.prov_graph.reset_materialization_nodes()
        self.actions = []
        self.llm_messages = []
        self.join_paths = None

        for doc_id in self.last_intermediate_table_ids:
            try:
                self.db_api.execute_query(
                    self.user_id,
                    self.chat_id,
                    f'DROP TABLE IF EXISTS "{doc_id}";',
                )
            except Exception as e:
                self.__log(
                    f"Warning: failed to delete intermediate table {doc_id} from the database: {e}"
                )
                continue
        self.last_intermediate_table_ids = set()
        self.__log("Materializer reset complete.")

    def __log(self, text: str):
        formatted_log(self.logger, "MATERIALIZER", text)
