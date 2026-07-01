from json import dumps
from logging import Logger
from time import time
from typing import Any, Callable

from pandas import DataFrame
from tiktoken import encoding_for_model

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.ds_skeptic import DSSkeptic
from pneuma_seeker.services.core.conductor.models import (
    ConductorResponse,
    ConductorResponseType,
)
from pneuma_seeker.services.core.conductor.prompt_factory import ConductorPromptFactory
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.core.materializer.main import Materializer
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.str_processor import dataframe_to_preview_str
from pneuma_seeker.shared.schemas.core.action import ActionExecutionStatus, ActionNames
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    convert_retrieval_results_to_str,
)
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role


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
        frontend_callback: Callable[[ConductorResponse], None],
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.prov_graph = prov_graph
        self.db_api = db_api
        self.language_model_api = language_model_api
        self.frontend_callback = frontend_callback

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
            log_callback=lambda msg: self.frontend_callback(
                ConductorResponse(ConductorResponseType.LOG, msg)
            ),
        )
        self.ds_skeptic = DSSkeptic(self.language_model_api, self.config, self.logger)

        self.state = ConductorState()
        self.retrieved_tables: list[AbstractDocument] = []
        self.external_tables: list[AbstractDocument] = []
        self.enumerated_tables: list[AbstractDocument] = []
        self.web_search_result: AbstractDocument | None = None
        self.web_crawl_result: AbstractDocument | None = None
        self.join_paths: str | None = None

        self.user_facing_response = ""
        self.actions: list[str] = []
        self.llm_messages: list[LLMMessage] = []

        self._skeptic_rounds = 0
        self._pending_skeptic_feedback: str | None = None
        self._skeptic_pushed_back: bool = False

        self._action_handlers: dict[str, Any] = {
            ActionNames.SITUATIONAL_ANALYSIS.value: self._handle_situational_analysis,
            ActionNames.USER_FACING_COMMUNICATION.value: self._handle_user_facing_communication,
            ActionNames.TABLE_RETRIEVE.value: self._handle_table_retrieve,
            ActionNames.WEB_SEARCH.value: self._handle_web_search,
            ActionNames.WEB_CRAWL.value: self._handle_web_crawl,
            ActionNames.TABLE_ENUMERATION.value: self._handle_table_enumeration,
            ActionNames.STATE_MANIPULATION.value: self._handle_state_manipulation,
            ActionNames.MATERIALIZER.value: self._handle_materializer,
            ActionNames.PYTHON_EXECUTOR.value: self._handle_python_executor,
            ActionNames.CONTEXT_EXTRACTION.value: self._handle_context_extraction,
        }

    def chat(
        self,
        user_message: str,
        interaction_history: list[LLMMessage],
        external_table_paths: list[str],  # TODO: handle reading external tables
    ):
        """Processes user message and yields responses."""
        self._log(f"Processing user message: {user_message}")

        chat_start_time = time()
        self._reset_conductor()

        self.llm_messages = [
            LLMMessage(
                role=Role.SYSTEM.value,
                content=self.prompt_factory.get_sys_prompt(),
            )
        ]

        current_step = 0
        prev_accumulated_in_tokens = 0
        prev_accumulated_out_tokens = 0
        prev_accumulated_llm_time = 0.0

        while (
            self.user_facing_response == ""
            and current_step < self.config.MAX_CONDUCTOR_STEPS
        ):
            step_start_time = time()
            current_step += 1

            message = f"[Step {current_step} / up to {self.config.MAX_CONDUCTOR_STEPS}] Planning actions..."
            self.frontend_callback(
                ConductorResponse(ConductorResponseType.LOG, message)
            )
            self._log(message)

            curr_state_msg = LLMMessage(
                role=Role.USER.value,
                content=self.prompt_factory.get_curr_state_prompt(
                    current_step,
                    self.state,
                    interaction_history,
                    self.actions,
                    self.retrieved_tables,
                    user_message,
                    self.enumerated_tables,
                    self.external_tables,
                    self.web_search_result,
                    self.web_crawl_result,
                    self.join_paths,
                ),
            )

            try:
                llm_response = "".join(
                    self.language_model_api.chat(
                        self.llm_messages + [curr_state_msg],
                        LLMOption(json_mode=True, stream=True, top_p=0.1),
                    )
                )
            except Exception as exc:
                error_msg = (
                    f"An unexpected error occurred while generating the plan: {exc}."
                )
                self._log(error_msg)
                self.frontend_callback(
                    ConductorResponse(ConductorResponseType.LOG, error_msg)
                )
                raise exc

            self._log(f"Plan: {llm_response}")
            self.llm_messages.append(
                LLMMessage(
                    role=Role.USER.value,
                    content=self.prompt_factory.get_skeleton_curr_state_prompt(
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
                    LLMMessage(
                        role=Role.USER.value,
                        content=message,
                    )
                )
                self.frontend_callback(
                    ConductorResponse(
                        ConductorResponseType.LOG, "Fixing error in produced plan..."
                    )
                )
                continue

            for action_plan in plan:
                self._log(f"Executing action: {action_plan}")
                action_name: str = action_plan.get("action", "")
                action_args: dict = action_plan.get("args", {})

                self.frontend_callback(
                    ConductorResponse(
                        ConductorResponseType.LOG, f"Executing action: {action_name}..."
                    )
                )
                self._execute_action(user_message, action_name, action_args)

                if self._pending_skeptic_feedback is not None:
                    self.llm_messages.append(
                        LLMMessage(
                            role=Role.USER.value,
                            content=self._pending_skeptic_feedback,
                        )
                    )
                    pushed_back = self._skeptic_pushed_back
                    self._pending_skeptic_feedback = None
                    self._skeptic_pushed_back = False
                    if pushed_back:
                        break  # abort remaining plan actions

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

        if self.user_facing_response == "":
            self._log("Force produce user-facing response")
            self.llm_messages.append(
                LLMMessage(
                    role=Role.SYSTEM.value,
                    content=self.prompt_factory.get_direct_response_anyway_prompt(),
                )
            )
            self.user_facing_response = "".join(
                self.language_model_api.chat(self.llm_messages, LLMOption(stream=True))
            )
            self.user_facing_response = self.user_facing_response.strip()

        yield ConductorResponse(
            ConductorResponseType.FINAL_RESPONSE, self.user_facing_response
        )

        chat_end_time = time()
        self._log_overall_profiling(chat_start_time, chat_end_time)

    def _validate_plan(self, plan: Any) -> list[dict[str, Any]]:
        if not isinstance(plan, list):
            error_msg = "Plan specified is not valid (not a list)."
            self._log(f"=> {error_msg}")
            raise ValueError(error_msg)
        if len(plan) == 0:
            error_msg = "Plan specified is empty."
            self._log(f"=> {error_msg}")
            raise ValueError(error_msg)
        if not all(isinstance(action_plan, dict) for action_plan in plan):
            error_msg = "Plan specified is not valid (not all entries are objects)."
            self._log(f"=> {error_msg}")
            raise ValueError(error_msg)

        executor_in_plan = False
        materializer_in_plan = False
        user_facing_in_plan = False
        table_retrieve_in_plan = False
        assumption_check_in_plan = False

        for action_plan in plan:
            if action_plan.get("action") is None:
                error_msg = "Action specified is not valid (missing 'action' key)."
                self._log(f"=> {error_msg}")
                raise ValueError(error_msg)
            if not isinstance(action_plan.get("action"), str):
                error_msg = f"Action specified is not valid (not a string): {action_plan.get('action')}"
                self._log(f"=> {error_msg}")
                raise ValueError(error_msg)
            if not self.action_set.is_valid_conductor_action(
                action_plan.get("action", "")
            ):
                error_msg = (
                    f"Action specified is not valid: {action_plan.get('action')}"
                )
                self._log(f"=> {error_msg}")
                raise ValueError(error_msg)

            action = action_plan.get("action")
            if action == ActionNames.PYTHON_EXECUTOR.value:
                executor_in_plan = True
            elif action == ActionNames.USER_FACING_COMMUNICATION.value:
                user_facing_in_plan = True
            elif action == ActionNames.MATERIALIZER.value:
                materializer_in_plan = True
            elif action == ActionNames.TABLE_RETRIEVE.value:
                table_retrieve_in_plan = True
            elif action == ActionNames.CONTEXT_EXTRACTION.value:
                assumption_check_in_plan = True

        # Strip user-facing communication when it co-occurs with execution actions
        if (
            executor_in_plan
            or materializer_in_plan
            or table_retrieve_in_plan
            or assumption_check_in_plan
        ) and user_facing_in_plan:
            self._log(
                "==> Removing user-facing communication from plan due to presence of code execution or materialization or table retrieval or assumption check."
            )
            plan = [
                action_plan
                for action_plan in plan
                if action_plan.get("action")
                != ActionNames.USER_FACING_COMMUNICATION.value
            ]

        return plan

    def _execute_action(
        self, user_message: str, action_name: str, action_args: dict[str, Any]
    ) -> None:
        action_start_time = time()
        llm = self.language_model_api.llm
        llm_time_before = llm.total_llm_time

        action_json = dumps({"action": action_name, "args": action_args})
        try:
            enc = encoding_for_model("o4-mini")
            plan_tokens = len(enc.encode(action_json))
        except Exception:
            plan_tokens = len(action_json.split())

        handler = self._action_handlers.get(action_name)
        if handler is None:
            outcome = f"Tool calling failed; {action_name} is unknown"
            status = ActionExecutionStatus.ERROR
        else:
            outcome, status = handler(user_message, action_args)

        self._log_action_profiling(
            action_name,
            status,
            plan_tokens,
            time() - action_start_time,
            llm.total_llm_time - llm_time_before,
        )
        self.llm_messages.append(LLMMessage(role=Role.USER.value, content=outcome))

    def _handle_situational_analysis(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        message = action_args.get("message")
        if not isinstance(message, str):
            error_msg = "=> `args` must be an object with a `message` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if len(message.strip()) == 0:
            error_msg = "=> `message` must be a non-empty string"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        success_msg = f"You did a situational analysis: {message}"
        self._log(f"=> {success_msg}")
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_user_facing_communication(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        try:
            message = self.action_set.user_facing_communication(action_args)
        except ValueError as e:
            self._log(f"=> {e}")
            return str(e), ActionExecutionStatus.ERROR
        self.user_facing_response = message
        success_msg = f"You communicated to the user: {message}"
        self._log(success_msg)
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_table_retrieve(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"Table Retrieve request with params: {action_args}")
        if not isinstance(action_args, dict):
            error_msg = "=> `args` must be an object with a `prompt` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if "prompts" not in action_args:
            error_msg = "=> `args` must have a `prompts` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(action_args["prompts"], list) or not all(
            isinstance(p, str) for p in action_args["prompts"]
        ):
            error_msg = "=> `prompts` must be a list of strings"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        self.retrieved_tables = self.action_set.retrieve_multi_topic_documents(
            action_args["prompts"], RetrieverType.PNEUMA_RETRIEVER, 10, True, 3
        )
        self._log(f"Retrieved tables:\n {[i.doc_id for i in self.retrieved_tables]}")
        try:
            self.join_paths = self.action_set.discover_join_paths(self.retrieved_tables)
        except Exception as e:
            self._log(f"=> Error during join path extraction: {e}")
        success_msg = "Successfully retrieved tables from Table Retrieve. Notice that the `RETRIEVED TABLES` has been updated."
        self._log(success_msg)
        self._log_table_repr_tokens(self.retrieved_tables)
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_web_search(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"Web Search request with params: {action_args}")
        if not isinstance(action_args, dict):
            error_msg = "=> `args` must be an object with a `prompt` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if "prompt" not in action_args:
            error_msg = "=> `args` must have a `prompt` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        retrieved_docs = self.action_set.retrieve_documents(
            action_args["prompt"], RetrieverType.WEB_SEARCH
        )
        self.web_search_result = retrieved_docs[0] if len(retrieved_docs) > 0 else None
        if self.web_search_result is None:
            return (
                "No relevant information was found from Web Search.",
                ActionExecutionStatus.SUCCESS,
            )
        return (
            "Successfully retrieved information from Web Search. Notice that the `WEB SEARCH RESULT` has been updated.",
            ActionExecutionStatus.SUCCESS,
        )

    def _handle_web_crawl(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"Web Crawl request with params: {action_args}")
        if not isinstance(action_args, dict):
            error_msg = "=> `args` must be an object with a `url` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if "url" not in action_args:
            error_msg = "=> `args` must have a `url` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        retrieved_docs = self.action_set.retrieve_documents(
            action_args["url"], RetrieverType.WEB_CRAWL
        )
        self.web_crawl_result = retrieved_docs[0] if len(retrieved_docs) > 0 else None
        if self.web_crawl_result is None:
            success_msg = "No relevant information was found from Web Crawl."
            self._log(success_msg)
            return success_msg, ActionExecutionStatus.SUCCESS
        success_msg = "Successfully retrieved information from Web Crawl. Notice that the `WEB CRAWL RESULT` has been updated."
        self._log(success_msg)
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_table_enumeration(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"Table Enumerator request with params: {action_args}")
        if not isinstance(action_args, dict):
            error_msg = "=> `args` must be an object with a `pattern` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if "patterns" not in action_args:
            error_msg = "=> `args` must have a `patterns` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not isinstance(action_args["patterns"], list) or not all(
            isinstance(p, str) for p in action_args["patterns"]
        ):
            error_msg = "=> `patterns` must be a list of strings"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        if not all(len(p.strip()) > 0 for p in action_args["patterns"]):
            error_msg = "=> `patterns` must be a list of non-empty strings"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        self.enumerated_tables = self.action_set.retrieve_multi_topic_documents(
            action_args["patterns"], RetrieverType.ENUMERATOR, 20, True, 2
        )
        success_msg = f"Enumerated table IDs based on these patterns: {action_args['patterns']}. If there are any matches, the IDs will be reflected in `OTHER TABLE IDS WITH SIMILAR NAMING PATTERNS`."
        self._log(success_msg)
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_state_manipulation(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"State Manipulation request with params: {action_args}")
        if not isinstance(action_args, dict):
            error_msg = "`args` must be an object"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        T: dict[str, list[str]] | None = action_args.get("T")
        column_descriptions: dict[str, dict[str, str]] | None = action_args.get(
            "column_descriptions"
        )
        S: str | None = action_args.get("S")

        is_T_modified = False
        if T is not None and len(T) > 0:
            if column_descriptions is not None:
                # STATE_MANIPULATION is a pure in-memory schema operation.
                # MATERIALIZER owns the DB lifecycle of T tables (drop/create).
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
                self.state.T = T_docs
                self.state.column_descriptions = column_descriptions
                self.state.is_T_materialized = False
                is_T_modified = True
            else:
                error_msg = "If you want to change T, make sure to also define column_descriptions."
                self._log(error_msg)
                return error_msg, ActionExecutionStatus.ERROR

        is_S_modified = False
        if S is not None:
            self.state.S = S
            self.state.is_S_executed = False
            self.state.s_description = ""
            is_S_modified = True

        if is_T_modified and is_S_modified:
            success_msg = "Successfully modified both T and S."
        elif is_T_modified:
            success_msg = "Successfully modified T."
        elif is_S_modified:
            success_msg = "Successfully modified S."
        else:
            error_msg = "No modification is done."
            self._log(error_msg)
            return error_msg, ActionExecutionStatus.ERROR

        self._log(success_msg)

        if (
            self.config.ENABLE_DS_SKEPTIC
            and self._skeptic_rounds < self.config.MAX_DS_SKEPTIC_ROUNDS
        ):
            self._skeptic_rounds += 1
            message = "DS-Skeptic reviewing analysis plan..."
            self._log(message)
            self.frontend_callback(
                ConductorResponse(ConductorResponseType.LOG, message)
            )

            _DS_SKEPTIC_TABLE = "ds_skeptic_check"

            def _run_ds_skeptic_ce(uncertainties: list[dict]) -> str:
                summary, log_msgs = self.action_set.run_context_extraction(
                    uncertainties,
                    self.retrieved_tables + self.external_tables,
                    _DS_SKEPTIC_TABLE,
                )
                for log_msg in log_msgs:
                    self.frontend_callback(
                        ConductorResponse(ConductorResponseType.LOG, log_msg)
                    )
                for cleanup_stmt in (
                    f"DROP TABLE IF EXISTS {_DS_SKEPTIC_TABLE};",
                    f"DROP VIEW IF EXISTS {_DS_SKEPTIC_TABLE};",
                ):
                    try:
                        self.db_api.execute_query(
                            self.user_id, self.chat_id, cleanup_stmt
                        )
                    except Exception as cleanup_exc:
                        self._log(f"DS-Skeptic CE cleanup warning: {cleanup_exc}")
                return summary

            push_back, feedback = self.ds_skeptic.review(
                user_message,
                self.state.T,
                self.state.column_descriptions,
                self.state.S or "",
                self.retrieved_tables,
                run_ce_fn=_run_ds_skeptic_ce,
            )
            self._pending_skeptic_feedback = feedback
            self._skeptic_pushed_back = push_back
            if push_back:
                self.frontend_callback(
                    ConductorResponse(
                        ConductorResponseType.LOG,
                        "DS-Skeptic has concerns — Conductor will reconsider in the next step.",
                    )
                )

        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_materializer(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        if len(self.state.T.keys()) == 0:
            error_msg = "T has to already be defined before calling Materializer"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        note = ""
        if isinstance(action_args, dict) and "note" in action_args:
            note = action_args["note"]
        mode = (
            action_args.get("mode", "fresh")
            if isinstance(action_args, dict)
            else "fresh"
        )
        update_mode = mode == "update"
        self._log(f"Materializer called (note: {note}, mode={mode})")

        (
            retrieved_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
            materialized_T_dfs,
        ) = self.materializer.materialize_T(
            {T_id: T_doc.content for T_id, T_doc in self.state.T.items()},
            self.state.column_descriptions,
            self.state.S,
            note,
            update_mode,
            self.external_tables,
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

        for T_id, T_df in materialized_T_dfs.items():
            self.state.T[T_id].content = T_df

        self.state.is_T_materialized = True
        success_msg = "Successfully materialized T."
        self._log(success_msg)
        return success_msg, ActionExecutionStatus.SUCCESS

    def _handle_python_executor(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"{ActionNames.PYTHON_EXECUTOR.value} called")
        if not self.state.is_T_materialized:
            if len(self.state.T.keys()) > 0:
                self._log(
                    f"=> Self-triggered materialization from calling {ActionNames.PYTHON_EXECUTOR.value}..."
                )
                # Use update mode when prior intermediates exist (T schema was extended,
                # not redesigned). Fresh mode only for first-ever materialization.
                auto_mode = (
                    "update"
                    if self.materializer._saved_intermediate_tables
                    else "fresh"
                )
                self._execute_action(
                    user_message, ActionNames.MATERIALIZER.value, {"mode": auto_mode}
                )
            else:
                error_msg = f"T has not been defined. Please define it first before calling {ActionNames.PYTHON_EXECUTOR.value}."
                self._log(f"=> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR
        if len(self.state.S) == 0:
            error_msg = f"S is still empty, which means there is nothing to execute. Please define S first, then ensure T has been materialized using Materializer, and finally, you can call {ActionNames.PYTHON_EXECUTOR.value} again."
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR
        try:
            execution_result = self.action_set.execute_code(
                self.state.S, "conductor_s_execution"
            )
            execution_result_str = dataframe_to_preview_str(execution_result)
            self._log(f"Script (S) execution result: {execution_result_str}")
            self.state.is_S_executed = True
            return (
                f"Executed S, which resulted in this output: {execution_result_str}",
                ActionExecutionStatus.SUCCESS,
            )
        except Exception as e:
            error_msg = f"Error during script (S) execution: {e}"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

    def _handle_context_extraction(
        self, user_message: str, action_args: dict[str, Any]
    ) -> tuple[str, ActionExecutionStatus]:
        self._log(f"Context Extraction request with params: {action_args}")
        if not isinstance(action_args, dict):
            error_msg = "=> `args` must be an object with an `uncertainties` property"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        # Backwards compat: old-style {"code": "..."} arg
        if "code" in action_args and "uncertainties" not in action_args:
            self._log("=> Falling back to legacy code-based context extraction.")
            try:
                execution_result = self.action_set.execute_code(
                    action_args["code"], "conductor_assumption_check"
                )
                execution_result_str = dataframe_to_preview_str(execution_result)
                self._log(f"Legacy Context Extraction result: {execution_result_str}")
                for cleanup_stmt in (
                    "DROP TABLE IF EXISTS conductor_assumption_check;",
                    "DROP VIEW IF EXISTS conductor_assumption_check;",
                ):
                    try:
                        self.db_api.execute_query(
                            self.user_id, self.chat_id, cleanup_stmt
                        )
                    except Exception as cleanup_exc:
                        self._log(f"Cleanup warning ({cleanup_stmt}): {cleanup_exc}")
                return (
                    f"Executed Context Extraction, which resulted in this output: {execution_result_str}",
                    ActionExecutionStatus.SUCCESS,
                )
            except Exception as e:
                error_msg = f"Error during Context Extraction: {e}"
                self._log(f"=> {error_msg}")
                return error_msg, ActionExecutionStatus.ERROR

        uncertainties = action_args.get("uncertainties")
        if not isinstance(uncertainties, list) or len(uncertainties) == 0:
            error_msg = "=> `uncertainties` must be a non-empty list of {table_ids, question} objects"
            self._log(f"=> {error_msg}")
            return error_msg, ActionExecutionStatus.ERROR

        available_tables = self.retrieved_tables + self.external_tables
        summary, log_msgs = self.action_set.run_context_extraction(
            uncertainties, available_tables, "conductor_assumption_check"
        )
        for log_msg in log_msgs:
            self.frontend_callback(
                ConductorResponse(ConductorResponseType.LOG, log_msg)
            )
        self._log(f"Context Extraction summary: {summary}")
        for cleanup_stmt in (
            "DROP TABLE IF EXISTS conductor_assumption_check;",
            "DROP VIEW IF EXISTS conductor_assumption_check;",
        ):
            try:
                self.db_api.execute_query(self.user_id, self.chat_id, cleanup_stmt)
            except Exception as cleanup_exc:
                self._log(f"Cleanup warning ({cleanup_stmt}): {cleanup_exc}")
        return (
            f"Context Extraction result:\n{summary}",
            ActionExecutionStatus.SUCCESS,
        )

    def _reset_conductor(self):
        self.user_facing_response = ""
        self.actions = []
        self.llm_messages = []
        self._skeptic_rounds = 0
        self._pending_skeptic_feedback = None
        self._skeptic_pushed_back = False

        self.language_model_api.llm.reset_metrics()

    def _log_step_profiling(
        self, input_tokens: int, output_tokens: int, total_time: float, llm_time: float
    ) -> None:
        self._log(
            f"[PROFILING] Time taken: {total_time:.2f}s (CPU: {total_time - llm_time:.2f}s, LLM: {llm_time:.2f}s)"
        )
        self._log(
            f"[PROFILING] Input tokens: {input_tokens}, Output tokens: {output_tokens}"
        )

    def _log_overall_profiling(self, start_time: float, end_time: float) -> None:
        llm = self.language_model_api.llm
        total_time = end_time - start_time
        llm_time = llm.total_llm_time
        self._log(
            f"[PROFILING] Time taken: {total_time:.2f}s (CPU: {total_time - llm_time:.2f}s, LLM: {llm_time:.2f}s)"
        )
        self._log(
            f"[PROFILING] Input tokens: {llm.total_input_tokens}, Output tokens: {llm.total_output_tokens}"
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

    def _log_action_profiling(
        self,
        action_name: str,
        status: ActionExecutionStatus,
        plan_tokens: int,
        total_time: float,
        llm_time: float,
    ) -> None:
        tag = "OK" if status == ActionExecutionStatus.SUCCESS else "ERR"
        self._log(
            f"[PROFILING][{action_name}][{tag}] Time taken: {total_time:.2f}s (CPU: {total_time - llm_time:.2f}s, LLM: {llm_time:.2f}s)"
        )
        self._log(f"[PROFILING][{action_name}][{tag}] Plan JSON: ~{plan_tokens} tokens")

    def _log(self, text):
        formatted_log(self.logger, "Conductor", text)

    def set_prov_graph(self, prov_graph: ProvenanceGraph) -> None:
        """
        Replaces the provenance graph and propagate it to subcomponents.
        This is important when restoring a session from persistence; the ActionSet
        and Materializer both keep their own references.
        """
        self.prov_graph = prov_graph
        self.action_set.prov_graph = prov_graph
        self.materializer.prov_graph = prov_graph
