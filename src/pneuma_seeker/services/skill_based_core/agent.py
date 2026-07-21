from logging import Logger
from time import time
from typing import Callable, cast

from pandas import DataFrame

from pneuma_seeker.provenance.graph import ProvenanceGraph, ProvenanceNode
from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.skill_based_core.skills import discover_skills
from pneuma_seeker.services.skill_based_core.skills.base import SkillBase
from pneuma_seeker.services.core.conductor.models import (
    ConductorResponse,
    ConductorResponseType,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    convert_retrieval_results_to_str,
)
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role

_SYSTEM_PROMPT_TEMPLATE = """\
You are **SkillsAgent**, a data assistant that answers questions by calling skills one at a time.

# Goal
Fulfill the user's information need by calling skills sequentially. After each skill returns
a result, evaluate it and decide the next step. When you have enough to answer, call `respond`.

# Output format
Every response must be exactly one JSON object:
{{"skill": "<skill_name>", "args": {{<arguments>}}}}

# Available skills
{skills_doc}

# Skill sequencing guidelines
1. Start with `situational_analysis` to reason before acting.
2. Retrieve before transforming: call `retrieve_tables` first.
3. Probe before concluding irrelevance: use `probe_table` before dismissing a table.
4. **Call `define_target` before any operator or execution skill.** It declares (T, S) —
   the target table schemas and analysis script — making the agent's intent explicit.
   This is the relational reification step. You may refine (T, S) by calling it again.
5. Prefer operators (`project_table`, `join_tables`, `union_tables`) over `run_sql`,
   which is preferred over `run_python`, to populate T's rows after `define_target`.
6. Be assertive: use the best available proxy, disclose it clearly in `respond`.
"""


class SkillsAgent:
    """Flat single-loop alternative to Conductor + Materializer.

    Replaces the two-level Conductor → Materializer hierarchy with one agent that calls
    skills from a discovery-based registry. Skills are organized in sub-packages under
    skills/, each with a SKILLS.md (LLM prompt) and impl.py (SkillBase subclasses).
    To add a new skill group, create a new directory with those two files — no other
    changes needed.

    Intended for token-usage and runtime comparison against the original system.
    """

    MAX_STEPS = 20  # matches MAX_CONDUCTOR_STEPS + MAX_MATERIALIZER_STEPS

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
            user_id, chat_id, config, logger, prov_graph, db_api, language_model_api
        )

        # Skill registry + assembled system prompt (built once at init)
        self._skill_registry: dict[str, SkillBase]
        self._system_prompt: str
        self._skill_registry, skills_doc = discover_skills(config)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(skills_doc=skills_doc)

        # Relational state: (T, S) persists across turns exactly like Conductor.state.
        # define_target skill writes here; chat_session.py saves/restores it.
        self.state = ConductorState()

        # Remaining state that persists across turns (mirrors Conductor public attributes)
        self.retrieved_tables: list[AbstractDocument] = []
        self.enumerated_tables: list[AbstractDocument] = []
        self.external_tables: list[AbstractDocument] = []
        self.web_search_result: AbstractDocument | None = None
        self.web_crawl_result: AbstractDocument | None = None
        self.workspace_table_ids: list[str] = []
        self.join_paths: str | None = None  # compatibility stub

        # Per-call transient state
        self._done = False
        self._user_response = ""
        self._messages: list[LLMMessage] = []
        self._skill_calls: list[str] = []
        self.dataset_name = ""

    # ------------------------------------------------------------------
    # ChatSession compatibility (mirrors Conductor public interface)
    # ------------------------------------------------------------------

    def set_prov_graph(self, prov_graph: ProvenanceGraph) -> None:
        self.prov_graph = prov_graph
        self.action_set.prov_graph = prov_graph

    # ------------------------------------------------------------------
    # Public chat interface
    # ------------------------------------------------------------------

    def chat(
        self,
        user_input: str,
        interaction_history: list[LLMMessage],
        external_table_paths: list[str],  # TODO: handle reading external tables
        plan_mode: bool = False,  # not supported by SkillsAgent; accepted for signature parity with Conductor
        dataset_name: str = "",
    ):
        """Drive the skills loop; yields LOG strings then the final user response."""
        chat_start_time = time()
        self._log(f"Processing: {user_input}")
        self._reset()
        # Per-call, not construction-time: fresh on every turn, mirroring how
        # retrieved_tables/_messages/etc. are reset above — never stale.
        self.dataset_name = dataset_name

        self._messages = [
            LLMMessage(role=Role.SYSTEM.value, content=self._system_prompt)
        ]

        prev_input_tokens = 0
        prev_output_tokens = 0

        step = 0
        while not self._done and step < self.MAX_STEPS:
            step += 1
            self.frontend_callback(
                ConductorResponse(
                    ConductorResponseType.LOG,
                    f"[Step {step} / {self.MAX_STEPS}] Deciding next skill...",
                )
            )
            self._log(f"Step {step}/{self.MAX_STEPS}")

            ctx = self._build_context(step, user_input, interaction_history)
            self._messages.append(LLMMessage(role=Role.USER.value, content=ctx))
            ctx_idx = len(self._messages) - 1

            try:
                full_response = "".join(
                    self.language_model_api.chat(
                        self._messages,
                        LLMOption(json_mode=True, stream=True, top_p=0.1),
                    )
                )
            except Exception as exc:
                self._log(f"LLM error: {exc}")
                self.frontend_callback(
                    ConductorResponse(ConductorResponseType.LOG, f"LLM error: {exc}")
                )
                raise exc

            self._log(f"LLM: {full_response}")
            self._messages.append(
                LLMMessage(role=Role.ASSISTANT.value, content=full_response)
            )
            # Compress the context message to prevent unbounded context growth
            self._messages[ctx_idx]["content"] = f"Step {step} (truncated)"

            try:
                parsed = parse_json(full_response)
                skill_name: str = parsed.get("skill", "")
                skill_args: dict = parsed.get("args", {})
                if not skill_name:
                    raise ValueError("Response missing 'skill' key")
                self._skill_calls.append(skill_name)

                result_content = self._call_skill(skill_name, skill_args)
                self._messages.append(
                    LLMMessage(role=Role.USER.value, content=result_content)
                )
                self.frontend_callback(
                    ConductorResponse(
                        ConductorResponseType.LOG, f"Skill '{skill_name}' executed."
                    )
                )
            except Exception as exc:
                err = f"Error parsing/executing skill: {exc}"
                self._log(err)
                self._messages.append(LLMMessage(role=Role.USER.value, content=err))

            # Per-step token profiling (mirrors Conductor)
            llm = self.language_model_api.llm
            step_in = llm.total_input_tokens - prev_input_tokens
            prev_input_tokens = llm.total_input_tokens
            self._log(f"[PROFILING] Step {step} input tokens: {step_in}")
            step_out = llm.total_output_tokens - prev_output_tokens
            prev_output_tokens = llm.total_output_tokens
            self._log(f"[PROFILING] Step {step} output tokens: {step_out}")

        if not self._done:
            self._log("Max steps reached — forcing final response")
            self._messages.append(
                LLMMessage(
                    role=Role.SYSTEM.value,
                    content=(
                        "You have reached the maximum number of steps. "
                        "Respond to the user directly now — no JSON format required."
                    ),
                )
            )
            self._user_response = "".join(
                self.language_model_api.chat(self._messages, LLMOption(stream=True))
            )

        yield ConductorResponse(
            ConductorResponseType.FINAL_RESPONSE, self._user_response
        )

        chat_end_time = time()
        elapsed = chat_end_time - chat_start_time
        llm = self.language_model_api.llm
        self._log(f"[PROFILING] Chat completed in {elapsed:.2f}s")
        self._log(f"[PROFILING] LLM time: {llm.total_llm_time:.2f}s")
        self._log(f"[PROFILING] Non-LLM time: {elapsed - llm.total_llm_time:.2f}s")
        self._log(f"[PROFILING] Total input tokens: {llm.total_input_tokens}")
        self._log(f"[PROFILING] Total output tokens: {llm.total_output_tokens}")

    # ------------------------------------------------------------------
    # Skill dispatch
    # ------------------------------------------------------------------

    def _call_skill(self, skill_name: str, args: dict) -> str:
        """Look up skill in registry, execute it, handle terminates flag."""
        skill = self._skill_registry.get(skill_name)
        if skill is None:
            return (
                f"Unknown skill '{skill_name}'. "
                f"Available: {sorted(self._skill_registry.keys())}"
            )
        result = skill.execute(args, self)
        if result.terminates:
            self._done = True
        return result.content

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(
        self,
        step: int,
        user_input: str,
        interaction_history: list[LLMMessage],
    ) -> str:
        parts: list[str] = [
            f"Step {step} / {self.MAX_STEPS}",
            "",
            f"User question: {user_input}",
        ]
        if interaction_history:
            parts += ["", "Recent interactions:"]
            recent = interaction_history[-6:]
            for i in range(0, len(recent) - 1, 2):
                if (
                    recent[i]["role"] == Role.USER.value
                    and recent[i + 1]["role"] == Role.ASSISTANT.value
                ):
                    parts.append(
                        f'  {{"user input": {recent[i]["content"]}, "your response": {recent[i + 1]["content"]}}}'
                    )
        if self.external_tables:
            parts += ["", "External tables (user-uploaded):"]
            parts.append(convert_retrieval_results_to_str(self.external_tables))
        if self.retrieved_tables:
            parts += ["", "Retrieved tables:"]
            parts.append(convert_retrieval_results_to_str(self.retrieved_tables))
        if self.enumerated_tables:
            parts += [
                "",
                f"Enumerated table IDs: {[t.doc_id for t in self.enumerated_tables]}",
            ]
        if self.workspace_table_ids:
            parts += [
                "",
                f"Workspace tables (created so far): {self.workspace_table_ids}",
            ]
        if self.web_search_result:
            parts += ["", f"Web search result:\n{self.web_search_result}"]
        if self.web_crawl_result:
            parts += ["", f"Web crawl result:\n{self.web_crawl_result}"]
        if self._skill_calls:
            parts += ["", f"Skills called so far: {self._skill_calls}"]
        parts += ["", "Call the next skill."]
        return "\n".join(parts)

    def _reset(self) -> None:
        self._done = False
        self._user_response = ""
        self._messages = []
        self._skill_calls = []
        self.language_model_api.llm.reset_metrics()

    def _log(self, text: str) -> None:
        formatted_log(self.logger, "SKILLS_AGENT", text)
