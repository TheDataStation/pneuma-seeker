from logging import Logger
from time import time
from typing import Any, cast

from pandas import DataFrame

from pneuma_seeker.provenance.graph import ProvenanceGraph, ProvenanceNode
from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.prompt_factory import ConductorPromptFactory
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.core.materializer.main import Materializer
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.core.action import ActionExecutionStatus, ActionNames
from pneuma_seeker.shared.schemas.core.conductor import UserConductorInteraction
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
)
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.table_reader import TableReader


class Conductor:
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

        self.action_set = ActionSet(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.prov_graph,
            self.db_api,
            self.language_model_api,
        )
        self.prompt_factory = ConductorPromptFactory(self.config, self.action_set)
        self.materializer = Materializer(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.prov_graph,
            self.action_set,
            self.db_api,
            self.language_model_api,
        )
        self.table_reader = TableReader(
            self.config.OPENWEBUI_BASE_URL, self.config.OPENWEBUI_API_KEY
        )

        self.state = ConductorState()
        self.retrieved_tables: list[AbstractDocument] = []
        self.external_tables: list[AbstractDocument] = []
        self.enumerated_tables: list[AbstractDocument] = []
        self.web_search_result: AbstractDocument | None = None
        self.web_crawl_result: AbstractDocument | None = None
        self.join_paths: str | None = None

        # Short-lived state (per chat call)
        self.user_facing_response = ""
        self.is_user_facing_response = False
        self.actions: list[str] = []
        self.llm_messages: list[LLMMessage] = []

    def set_prov_graph(self, prov_graph: ProvenanceGraph) -> None:
        """Replace the provenance graph and propagate it to subcomponents.

        This is important when restoring a session from persistence; the ActionSet
        and Materializer both keep their own references.
        """
        self.prov_graph = prov_graph
        self.action_set.prov_graph = prov_graph
        self.materializer.prov_graph = prov_graph

    def chat(
        self,
        user_input: str,
        interaction_history: list[UserConductorInteraction],
        external_table_paths: list[str],
    ):
        """Processes user input and yields responses."""
        chat_start_time = time()
        self.__log(f"Processing user input: {user_input}")
        self.__reset_conductor()
        self.external_tables = self.table_reader.process_external_tables(
            external_table_paths
        )
        if len(self.external_tables) > 0:
            self.__log("External tables loaded")
            for index, doc in enumerate(self.external_tables):
                last_id = getattr(doc, "last_node_id", None)
                if (
                    last_id is not None
                    and self.prov_graph.get_node_by_id(last_id) is not None
                ):
                    continue

                new_node = ProvenanceNode(
                    source_retriever=RetrieverType.USER,
                    python_code=self.action_set.generate_read_external_tables_code(
                        index + 1, doc
                    ),
                    description="Reads a user-uploaded table.",
                )
                self.prov_graph.add_node(new_node, True)
                doc.last_node_id = new_node.id

                self.db_api.persist_df(
                    self.user_id,
                    self.chat_id,
                    cast(DataFrame, doc.content),
                    doc.doc_id,
                    True,
                )

        self.llm_messages = [
            LLMMessage(
                role=Role.SYSTEM.value,
                content=self.prompt_factory.get_sys_prompt(),
            )
        ]

        current_step = 0
        previous_step_input_tokens = 0
        previous_step_output_tokens = 0
        last_env_state_idx: int | None = None
        while (
            not self.is_user_facing_response
            and current_step < self.config.MAX_CONDUCTOR_STEPS
        ):
            current_step += 1
            yield f"LOG: [Step {current_step} / up to {self.config.MAX_CONDUCTOR_STEPS}] Planning the next sequence of actions..."
            self.__log(
                f"Asking the model to produce a sequence of actions (Current step: {current_step}/{self.config.MAX_CONDUCTOR_STEPS})..."
            )
            self.llm_messages.append(
                LLMMessage(
                    role=Role.SYSTEM.value,
                    content=self.prompt_factory.get_env_state_prompt(
                        current_step,
                        self.state,
                        interaction_history,
                        self.actions[-5:],  # only include last 5 actions for brevity
                        self.retrieved_tables,
                        user_input,
                        self.enumerated_tables,
                        self.external_tables,
                        self.web_search_result,
                        self.web_crawl_result,
                        self.join_paths,
                    ),
                )
            )
            last_env_state_idx = len(self.llm_messages) - 1

            try:
                full_response = "".join(
                    self.language_model_api.chat(
                        self.llm_messages,
                        LLMOption(json_mode=True, stream=True, top_p=0.1),
                    )
                )
            except Exception as exc:
                error_msg = f"An unexpected error occurred while generating the plan: {exc}."
                self.__log(error_msg)
                yield f"LOG: {error_msg}"
                raise exc
            self.__log(f"=> Model responded with a plan: {full_response}")
            self.llm_messages.append(
                LLMMessage(role=Role.ASSISTANT.value, content=full_response)
            )
            if last_env_state_idx is not None:
                self.llm_messages[last_env_state_idx]["content"] = (
                    self.prompt_factory.get_skeleton_env_state_prompt(
                        current_step,
                    )
                )

            try:
                self.__log("==> Parsing plan...")
                plan: list[dict[str, Any]] = parse_json(full_response).get("plan", [])

                if not isinstance(plan, list):
                    error_msg = "Plan specified is not valid (not a list)."
                    self.__log(f"=> {error_msg}")
                    raise ValueError(error_msg)
                if len(plan) == 0:
                    error_msg = "Plan specified is empty."
                    self.__log(f"=> {error_msg}")
                    raise ValueError(error_msg)
                if not all(isinstance(action_plan, dict) for action_plan in plan):
                    error_msg = (
                        "Plan specified is not valid (not all entries are objects)."
                    )
                    self.__log(f"=> {error_msg}")
                    raise ValueError(error_msg)

                plan = cast(list[dict[str, Any]], plan)
                executor_part_of_plan = False
                materializer_part_of_plan = False
                user_facing_communication_part_of_plan = False
                table_retrieve_part_of_plan = False
                assumption_check_part_of_plan = False
                for action_plan in plan:
                    assert isinstance(action_plan, dict)
                    if action_plan.get("action") is None:
                        error_msg = "Action specified is not valid (None)."
                        self.__log(f"=> {error_msg}")
                        raise ValueError(error_msg)
                    if not isinstance(action_plan.get("action"), str):
                        error_msg = f"Action specified is not valid (not a string): {action_plan.get('action')}"
                        self.__log(f"=> {error_msg}")
                        raise ValueError(error_msg)
                    if not self.action_set.is_valid_conductor_action(
                        action_plan.get("action", "")
                    ):
                        error_msg = f"Action specified is not valid: {action_plan.get('action')}"
                        self.__log(f"=> {error_msg}")
                        raise ValueError(error_msg)
                    if action_plan.get("action") == ActionNames.PYTHON_EXECUTOR.value:
                        executor_part_of_plan = True
                    if (
                        action_plan.get("action")
                        == ActionNames.USER_FACING_COMMUNICATION.value
                    ):
                        user_facing_communication_part_of_plan = True
                    if (
                        action_plan.get("action")
                        == ActionNames.MATERIALIZER.value
                    ):
                        materializer_part_of_plan = True
                    if action_plan.get("action") == ActionNames.TABLE_RETRIEVE.value:
                        table_retrieve_part_of_plan = True
                    if action_plan.get("action") == ActionNames.CONTEXT_EXTRACTION.value:
                        assumption_check_part_of_plan = True
                # Ensure there is no user-facing communication in the same plan as code execution (simply remove the user-facing part)
                if (executor_part_of_plan or materializer_part_of_plan or table_retrieve_part_of_plan or assumption_check_part_of_plan) and user_facing_communication_part_of_plan:
                    self.__log(
                        "==> Removing user-facing communication from plan due to presence of code execution or materialization or table retrieval or assumption check."
                    )
                    plan = [
                        action_plan
                        for action_plan in plan
                        if action_plan.get("action")
                        != ActionNames.USER_FACING_COMMUNICATION.value
                    ]
                self.__log("==> Plan parsed!")
                self.actions.append(str(plan))
            except Exception as exc:
                self.__log(f"=> Unexpected error occurred: {exc}")
                yield "LOG: Fixing error in produced plan..."
                self.llm_messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=f"An unexpected error occurred while processing your response: {exc}. Please fix the issue and try again.",
                    )
                )
                continue

            for action_plan in plan:
                self.__log(f"=> Executing this action: {action_plan}")
                action_name: str = action_plan.get("action", "")
                action_args: dict = action_plan.get("args", {})
                yield f"LOG: Executing action: {action_name}..."
                action_outcome, _ = self.__execute_action(action_name, action_args)
                self.llm_messages.append(
                    LLMMessage(role=Role.USER.value, content=action_outcome)
                )

            if hasattr(self.language_model_api.llm, "total_input_tokens"):
                step_input_tokens = self.language_model_api.llm.total_input_tokens - previous_step_input_tokens  # type: ignore
                previous_step_input_tokens = step_input_tokens
                self.__log(
                    f"==> [PROFILING] Total input tokens for this step: {step_input_tokens} tokens."
                )

            if hasattr(self.language_model_api.llm, "total_output_tokens"):
                step_output_tokens = self.language_model_api.llm.total_output_tokens - previous_step_output_tokens  # type: ignore
                previous_step_output_tokens = step_output_tokens
                self.__log(
                    f"==> [PROFILING] Total output tokens for this step: {step_output_tokens} tokens."
                )

        if not self.is_user_facing_response:
            self.__log("Force produce user-facing response")
            self.llm_messages.append(
                LLMMessage(
                    role=Role.SYSTEM.value,
                    content=self.prompt_factory.get_direct_response_anyway_prompt(),
                )
            )
            self.user_facing_response = "".join(
                self.language_model_api.chat(self.llm_messages, LLMOption(stream=True))
            )
            self.is_user_facing_response = True

        yield self.user_facing_response

        chat_end_time = time()
        if hasattr(self.language_model_api.llm, "total_llm_time"):
            self.__log(
                f"==> [PROFILING] Total LLM time for this chat: {self.language_model_api.llm.total_llm_time:.2f} seconds."  # type: ignore
            )
            self.__log(
                f"==> [PROFILING] Total Non-LLM time for this chat: {(chat_end_time - chat_start_time) - self.language_model_api.llm.total_llm_time:.2f} seconds."  # type: ignore
            )
        self.__log(
            f"[PROFILING] [OVERALL] Chat completed in {chat_end_time - chat_start_time:.2f} seconds."
        )
        if hasattr(self.language_model_api.llm, "total_input_tokens"):
            self.__log(
                f"==> [PROFILING] Total input tokens for this chat: {self.language_model_api.llm.total_input_tokens} tokens."  # type: ignore
            )
        if hasattr(self.language_model_api.llm, "total_output_tokens"):
            self.__log(
                f"==> [PROFILING] Total output tokens for this chat: {self.language_model_api.llm.total_output_tokens} tokens."  # type: ignore
            )

    def __execute_action(
        self, action_name: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        """Executes an action and returns the outcome message and status."""
        match action_name:
            case ActionNames.SITUATIONAL_ANALYSIS.value:
                self.__log(f"Situational Analysis request with params: {action_args}")
                message = action_args.get("message")
                if not isinstance(message, str):
                    error_msg = "=> `args` must be an object with a `message` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
                if len(message.strip()) == 0:
                    error_msg = "=> `message` must be a non-empty string"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
                success_msg = f"You did a situational analysis: {message}"
                self.__log(f"=> {success_msg}")
                return success_msg, ActionExecutionStatus.SUCCESS
            case ActionNames.USER_FACING_COMMUNICATION.value:
                self.__log(
                    f"User-Facing Communication request with params: {action_args}"
                )
                message = action_args.get("message")
                if not isinstance(message, str):
                    error_msg = "=> `args` must be an object with a `message` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
                if len(message.strip()) == 0:
                    error_msg = "=> `message` must be a non-empty string"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                self.user_facing_response = message
                self.is_user_facing_response = True
                success_msg = f"You communicated to the user: {message}"
                self.__log(f"=> {success_msg}")
                return success_msg, ActionExecutionStatus.SUCCESS
            case ActionNames.TABLE_RETRIEVE.value:
                self.__log(f"Table Retrieve request with params: {action_args}")

                if not isinstance(action_args, dict):
                    error_msg = "=> `args` must be an object with a `prompt` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                if "prompts" not in action_args:
                    error_msg = "=> `args` must have a `prompts` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                if not isinstance(action_args["prompts"], list) or not all(
                    isinstance(p, str) for p in action_args["prompts"]
                ):
                    error_msg = "=> `prompts` must be a list of strings"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                self.retrieved_tables = (
                    self.action_set.retrieve_multi_topic_documents(
                        action_args["prompts"],
                        RetrieverType.PNEUMA_RETRIEVER,
                        10,
                        True,
                        3,
                    )
                )

                self.__log(
                    f"Retrieved tables:\n {[i.doc_id for i in self.retrieved_tables]}"
                )

                try:
                    self.join_paths = self.action_set.discover_join_paths(
                        self.retrieved_tables
                    )
                except Exception as e:
                    self.__log(f"=> Error during join path extraction: {e}")

                success_msg = "Successfully retrieved tables from Table Retrieve. Notice that the `RETRIEVED TABLES` has been updated."
                self.__log(success_msg)
                return (
                    success_msg,
                    ActionExecutionStatus.SUCCESS,
                )
            case ActionNames.WEB_SEARCH.value:
                self.__log(f"Web Search request with params: {action_args}")
                if not isinstance(action_args, dict):
                    error_msg = "=> `args` must be an object with a `prompt` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
                if "prompt" not in action_args:
                    error_msg = "=> `args` must have a `prompt` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                retrieved_docs = self.action_set.retrieve_documents(
                    action_args["prompt"], RetrieverType.WEB_SEARCH
                )
                self.web_search_result = (
                    retrieved_docs[0] if len(retrieved_docs) > 0 else None
                )
                if self.web_search_result is None:
                    return (
                        "No relevant information was found from Web Search.",
                        ActionExecutionStatus.SUCCESS,
                    )
                return (
                    "Successfully retrieved information from Web Search. Notice that the `WEB SEARCH RESULT` has been updated.",
                    ActionExecutionStatus.SUCCESS,
                )
            case ActionNames.WEB_CRAWL.value:
                self.__log(f"Web Crawl request with params: {action_args}")
                if not isinstance(action_args, dict):
                    error_msg = "=> `args` must be an object with a `url` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
                if "url" not in action_args:
                    error_msg = "=> `args` must have a `url` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                retrieved_docs = self.action_set.retrieve_documents(
                    action_args["url"], RetrieverType.WEB_CRAWL
                )
                self.web_crawl_result = (
                    retrieved_docs[0] if len(retrieved_docs) > 0 else None
                )
                if self.web_crawl_result is None:
                    success_msg = "No relevant information was found from Web Crawl."
                    self.__log(success_msg)
                    return (
                        success_msg,
                        ActionExecutionStatus.SUCCESS,
                    )
                success_msg = "Successfully retrieved information from Web Crawl. Notice that the `WEB CRAWL RESULT` has been updated."
                self.__log(success_msg)
                return (
                    success_msg,
                    ActionExecutionStatus.SUCCESS,
                )
            case ActionNames.TABLE_ENUMERATION.value:
                self.__log(f"Table Enumerator request with params: {action_args}")

                if not isinstance(action_args, dict):
                    error_msg = "=> `args` must be an object with a `pattern` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                if "patterns" not in action_args:
                    error_msg = "=> `args` must have a `patterns` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                if not isinstance(action_args["patterns"], list) or not all(
                    isinstance(p, str) for p in action_args["patterns"]
                ):
                    error_msg = "=> `patterns` must be a list of strings"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                if not all(len(p.strip()) > 0 for p in action_args["patterns"]):
                    error_msg = "=> `patterns` must be a list of non-empty strings"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                self.enumerated_tables = (
                    self.action_set.retrieve_multi_topic_documents(
                        action_args["patterns"],
                        RetrieverType.ENUMERATOR,
                        20,
                        True,
                        2,
                    )
                )
                success_msg = f"Enumerated table IDs based on these patterns: {action_args['patterns']}. If there are any matches, the IDs will be reflected in `OTHER TABLE IDS WITH SIMILAR NAMING PATTERNS`."
                self.__log(success_msg)
                return (
                    success_msg,
                    ActionExecutionStatus.SUCCESS,
                )
            case ActionNames.STATE_MANIPULATION.value:
                self.__log(f"State Manipulation request with params: {action_args}")

                if not isinstance(action_args, dict):
                    error_msg = "`args` must be an object"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                T: dict[str, list[str]] | None = action_args.get("T")
                column_descriptions: dict[str, dict[str, str]] | None = action_args.get(
                    "column_descriptions"
                )
                S: str | None = action_args.get("S")

                is_T_modified = False
                if T is not None and len(T) > 0:
                    if column_descriptions is not None:
                        if self.state.T is not None and len(self.state.T) > 0:
                            for schema_id in self.state.T.keys():
                                self.db_api.execute_query(
                                    self.user_id,
                                    self.chat_id,
                                    f'DROP TABLE IF EXISTS "{schema_id}";',
                                )
                        T_docs: dict[str, AbstractDocument] = dict()
                        for schema_id in T:
                            target_schema_df = DataFrame(columns=T[schema_id])
                            T_docs[schema_id] = Table(
                                doc_id=schema_id,
                                retriever_type=RetrieverType.CONDUCTOR,
                                content=target_schema_df,
                                metadata={},
                                path=schema_id,
                            )
                            self.db_api.persist_df(
                                self.user_id,
                                self.chat_id,
                                T_docs[schema_id].content,
                                T_docs[schema_id].doc_id,
                                True,
                            )
                        self.state.T = T_docs
                        self.state.column_descriptions = column_descriptions
                        self.state.is_T_materialized = False
                        is_T_modified = True
                    else:
                        error_msg = "If you want to change T, make sure to also define column_descriptions."
                        self.__log(error_msg)
                        return (
                            error_msg,
                            ActionExecutionStatus.ERROR,
                        )

                is_S_modified = False
                if S is not None:
                    self.state.S = S
                    self.state.is_S_executed = False
                    is_S_modified = True

                if is_T_modified and is_S_modified:
                    success_msg = "Successfully modified both T and S."
                    self.__log(success_msg)
                    return (
                        success_msg,
                        ActionExecutionStatus.SUCCESS,
                    )
                if is_T_modified:
                    success_msg = "Successfully modified T."
                    self.__log(success_msg)
                    return success_msg, ActionExecutionStatus.SUCCESS
                if is_S_modified:
                    success_msg = "Successfully modified S."
                    self.__log(success_msg)
                    return success_msg, ActionExecutionStatus.SUCCESS
                error_msg = "No modification is done."
                self.__log(error_msg)
                return error_msg, ActionExecutionStatus.ERROR
            case ActionNames.MATERIALIZER.value:
                if len(self.state.T.keys()) == 0:
                    error_msg = (
                        "T has to already be defined before calling Materializer"
                    )
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                note = ""
                if isinstance(action_args, dict) and "note" in action_args:
                    note = action_args["note"]

                self.__log(f"Materializer called (note: {note})")

                self.state.T = self.__materialize_T_driver(
                    self.state.T,
                    self.state.column_descriptions,
                    self.state.S,
                    note,
                    self.external_tables,
                )
                self.state.is_T_materialized = True
                success_msg = "Successfully materialized T."
                self.__log(success_msg)
                return success_msg, ActionExecutionStatus.SUCCESS
            case ActionNames.PYTHON_EXECUTOR.value:
                self.__log(f"{ActionNames.PYTHON_EXECUTOR.value} called")
                if not self.state.is_T_materialized:
                    if len(self.state.T.keys()) > 0:
                        self.__log(
                            f"=> Self-triggered materialization from calling {ActionNames.PYTHON_EXECUTOR.value}..."
                        )
                        self.__execute_action(ActionNames.MATERIALIZER.value, {})
                    else:
                        error_msg = f"T has not been defined. Please define it first before calling {ActionNames.PYTHON_EXECUTOR.value}."
                        self.__log(f"=> {error_msg}")
                        return error_msg, ActionExecutionStatus.ERROR
                if len(self.state.S) == 0:
                    error_msg = f"S is still empty, which means there is nothing to execute. Please define S first, then ensure T has been materialized using Materializer, and finally, you can call {ActionNames.PYTHON_EXECUTOR.value} again."
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                try:
                    execution_result = self.action_set.execute_code(
                        self.state.S, "conductor_s_execution"
                    )
                    self.__log(f"Script (S) execution result: {execution_result}")

                    self.state.is_S_executed = True
                    return (
                        f"Executed S, which resulted in this output: {execution_result}",
                        ActionExecutionStatus.SUCCESS,
                    )
                except Exception as e:
                    error_msg = f"Error during script (S) execution: {e}"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
            case ActionNames.CONTEXT_EXTRACTION.value:
                self.__log(f"Assumption Check request with params: {action_args}")
                if not isinstance(action_args, dict):
                    error_msg = "=> `args` must be an object with a `code` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
                if "code" not in action_args:
                    error_msg = "=> `args` must have a `code` property"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR

                try:
                    execution_result = self.action_set.execute_code(
                        action_args["code"], "conductor_assumption_check"
                    )
                    self.__log(f"Assumption Check execution result: {execution_result}")
                    # Assumption checks may materialize either a *table* or a *view*.
                    # In DuckDB, attempting to DROP the wrong object type throws.
                    # Cleanup must be best-effort and must not turn a successful
                    # assumption_check into a failure.
                    for cleanup_stmt in (
                        "DROP VIEW IF EXISTS conductor_assumption_check;",
                        "DROP TABLE IF EXISTS conductor_assumption_check;",
                    ):
                        try:
                            self.db_api.execute_query(
                                self.user_id, self.chat_id, cleanup_stmt
                            )
                        except Exception as cleanup_exc:
                            self.__log(
                                f"Assumption Check cleanup warning ({cleanup_stmt}): {cleanup_exc}"
                            )
                    return (
                        f"Executed Assumption Check, which resulted in this output: {execution_result}",
                        ActionExecutionStatus.SUCCESS,
                    )
                except Exception as e:
                    error_msg = f"Error during Assumption Check execution: {e}"
                    self.__log(f"=> {error_msg}")
                    return error_msg, ActionExecutionStatus.ERROR
            case _:
                return (
                    f"Tool calling failed; {action_name} is unknown",
                    ActionExecutionStatus.ERROR,
                )

    def __materialize_T_driver(
        self,
        T: dict[str, AbstractDocument],
        col_descriptions: dict[str, dict[str, str]],
        S: str,
        user_side_note: str,
        external_tables: list[AbstractDocument],
    ):
        T_dfs: dict[str, DataFrame] = {}
        for T_id, T_doc in T.items():
            T_dfs[T_id] = T_doc.content

        (
            retrieved_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
            materialized_T_dfs,
        ) = self.materializer.materialize_T(
            T_dfs,
            col_descriptions,
            S,
            user_side_note,
            external_tables,
            self.retrieved_tables + self.enumerated_tables,
            self.web_search_result,
            self.web_crawl_result,
        )

        if len(retrieved_tables) > 0:
            self.retrieved_tables = retrieved_tables
        if web_search_result is not None:
            self.web_search_result = web_search_result
        if web_crawl_result is not None:
            self.web_crawl_result = web_crawl_result
        if join_paths is not None:
            self.join_paths = join_paths

        materialized_T: dict[str, AbstractDocument] = {}
        for T_id, T_df in materialized_T_dfs.items():
            materialized_T[T_id] = T[T_id]
            materialized_T[T_id].content = T_df
        return materialized_T

    def __reset_conductor(self):
        self.user_facing_response = ""
        self.is_user_facing_response = False
        self.actions = []
        self.llm_messages = []

        if hasattr(self.language_model_api.llm, "reset_metrics"):
            self.language_model_api.llm.reset_metrics()  # type: ignore

    def __log(self, text):
        formatted_log(self.logger, "CONDUCTOR", text)
