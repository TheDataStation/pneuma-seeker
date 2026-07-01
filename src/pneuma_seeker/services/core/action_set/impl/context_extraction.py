from typing import Any, Callable

from pandas import DataFrame

from pneuma_seeker.services.core.action_set.interfaces import Action, Applicable
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_code
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.str_processor import dataframe_to_preview_str

_SHARED_PREAMBLE = f"""**{ActionNames.CONTEXT_EXTRACTION.value}**
  - Explores tables to answer specific uncertainty questions before committing to a schema or script.
  - This action is used ONLY to gather evidence, perform sanity checks, or confirm suspicions. It has no lasting side effects.
  - It MUST NOT be used to construct final outputs or pipeline tables.
  - You specify **what** to find out in natural language; a dedicated inner loop generates and executes the code for you with a focused, smaller context, which reduces hallucinations.
  - Each uncertainty entry targets one or more specific tables and asks a concrete, answerable question about them (e.g., value distributions, NULL presence, cardinality, column semantics).
  - The system returns a structured Q->A summary which you can use to inform subsequent actions.
  - **Common questions to consider:**
    - Checking whether a condition holds
    - Inspecting column value distributions or edge cases
    - Counting, filtering, sampling, or summarizing to confirm a belief
    - Verifying whether columns contain NULLs or unexpected values
    - Checking cardinality / uniqueness of key columns
    - Confirming which numeric value (0/1) maps to which label for an indicator column"""

_CONDUCTOR_SUFFIX = """
  - Args: {"uncertainties": [{"table_ids": ["<table_id_1>", "<table_id_2>"], "question": "<concrete question about these tables>"}, ...]}"""

_MATERIALIZER_SUFFIX = """
  - Args: {"uncertainties": [{"table_ids": ["<table_id_1>", "<table_id_2>"], "question": "<concrete question about these tables>"}, ...]}"""


class ContextExtraction(Action, Applicable):
    action_name = ActionNames.CONTEXT_EXTRACTION
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = "ENABLE_CONTEXT_EXTRACTION"
    order = 7
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        if agent == AgentType.MATERIALIZER:
            return _SHARED_PREAMBLE + _MATERIALIZER_SUFFIX
        return _SHARED_PREAMBLE + _CONDUCTOR_SUFFIX

    def apply(self, input: dict[str, Any]) -> tuple[str, list[str]]:
        """
        Resolves a list of natural-language uncertainty questions.

        Expected input keys:
            uncertainties     list[{table_ids: list[str], question: str}]
            available_tables  list[AbstractDocument]
            result_table_name str  — e.g. "conductor_assumption_check"
            execute_code_fn   Callable[[str, str], Any]  — injected by ActionSet

        Returns:
            (formatted_Q→A_summary, log_messages)
        """
        uncertainties: list[dict] = input.get("uncertainties", [])
        available_tables: list[AbstractDocument] = input.get("available_tables", [])
        result_table_name: str = input.get("result_table_name", "ce_result")
        execute_code_fn: Callable[[str, str], Any] = input["execute_code_fn"]

        results: list[dict] = []
        log_messages: list[str] = []

        for i, uncertainty in enumerate(uncertainties):
            table_ids: list[str] = uncertainty.get("table_ids", [])
            question: str = uncertainty.get("question", "")
            if not question:
                continue

            log_messages.append(
                f"[Context Extraction {i + 1}/{len(uncertainties)}] Resolving: "
                f"{question[:80]}{'...' if len(question) > 80 else ''}"
            )
            self.__log(f"Resolving uncertainty: {question}")

            relevant_tables = [t for t in available_tables if t.doc_id in table_ids]
            if not relevant_tables:
                relevant_tables = available_tables

            table_str = "\n\n".join(str(t) for t in relevant_tables)
            answer, retry_logs = self._resolve_single(
                question, table_str, result_table_name, execute_code_fn
            )
            log_messages.extend(retry_logs)
            results.append({"question": question, "answer": answer})

        return self._format_results(results), log_messages

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _resolve_single(
        self,
        question: str,
        table_str: str,
        result_table_name: str,
        execute_code_fn: Callable[[str, str], Any],
    ) -> tuple[str, list[str]]:
        messages = [
            LLMMessage(
                role=Role.SYSTEM.value,
                content=self._get_inner_loop_system_prompt(),
            ),
            LLMMessage(
                role=Role.USER.value,
                content=self._get_question_prompt(
                    question, table_str, result_table_name
                ),
            ),
        ]

        log_messages: list[str] = []
        last_code = ""
        for attempt in range(self.config.MAX_CONTEXT_EXTRACTION_LOOP_STEPS):
            raw_response = "".join(
                self.language_model_api.chat(messages, LLMOption(stream=False))
            )
            code = parse_code(raw_response.strip())
            last_code = code

            try:
                result = execute_code_fn(code, result_table_name)
                result_str = (
                    dataframe_to_preview_str(result)
                    if isinstance(result, DataFrame)
                    else str(result)
                )
                self.__log(f"Resolved on attempt {attempt + 1}: {result_str}")
                return result_str, log_messages
            except Exception as exc:
                self.__log(f"Attempt {attempt + 1} failed: {exc}")
                log_messages.append(
                    f"[Context Extraction] Attempt {attempt + 1} failed, retrying: {str(exc)[:120]}"
                )
                messages.append(
                    LLMMessage(role=Role.ASSISTANT.value, content=raw_response)
                )
                messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=self._get_error_retry_prompt(str(exc), code, question),
                    )
                )

        return (
            f"Could not resolve after {self.config.MAX_CONTEXT_EXTRACTION_LOOP_STEPS} attempts. Last code:\n{last_code}",
            log_messages,
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _get_inner_loop_system_prompt(self) -> str:
        max_rows = self.config.MAX_RESULT_PREVIEW_ROWS
        return f"""You are a focused data exploration assistant. Given a specific question about one or more database tables, generate Python code to answer it.

# Code Requirements
- Tables are stored in a DuckDB-based workspace database and are NOT guaranteed to fit in memory.
- To access tables, use the provided database API:
    - `db_api.execute_query(user_id, chat_id, "<SQL query>")` — returns a Pandas DataFrame
    - `db_api`, `user_id`, and `chat_id` are available as variables. Do NOT import `db_api`.
- When referencing retrieved or enumerated tables, use the dataset-qualified name as provided. For example, Table x (dataset: y) -> y."x".
- When referencing target tables in T or external tables, use the table name directly.
- Prefer SQL over Pandas for transformations. If Python processing is necessary, process in small batches and never load full tables into memory.
- When using CTEs (WITH ...), attach them directly to the SELECT of a CREATE TABLE AS statement:
    CREATE OR REPLACE TABLE <name> AS
    WITH ...
    SELECT ...
- Never use escaped newlines (\\n) inside strings. Use triple-quoted strings for multi-line SQL.
- The result MUST be materialized using:
    db_api.execute_query(user_id, chat_id, 'CREATE OR REPLACE TABLE "<result_table_name>" AS SELECT ...')
- **Result size**: The result table must contain at most {max_rows} rows. Always aggregate, filter, or add LIMIT to your result query. Never materialize a full table scan or unbounded GROUP BY without a row cap.
- **CRITICAL — column names**: Only use column names that are explicitly listed in the table schema provided. Never assume a column exists (e.g. a derived "year" column) — compute derived values inline in SQL or Python instead.
- Allowed libraries: Pandas, NumPy, SciPy.
- Do NOT create a new DuckDB connection (no `duckdb.connect()`).
- Output ONLY the Python code. No markdown fences, no explanation."""

    def _get_question_prompt(
        self, question: str, table_str: str, result_table_name: str
    ) -> str:
        return f"""## Question
{question}

## Relevant Tables
{table_str}

Generate Python code that answers the question. Materialize the result into a table named "{result_table_name}"."""

    def _get_error_retry_prompt(
        self, error: str, prior_code: str, question: str
    ) -> str:
        return f"""The previous code raised an error. Fix it and regenerate code that answers the original question.

Original question: {question}

Error: {error}

Previous code:
{prior_code}

Remember: only use column names that appear in the table schema. Compute derived columns inline — do not reference columns that don't exist in the table."""

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_results(self, results: list[dict]) -> str:
        if not results:
            return "No uncertainties were resolved."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"Q{i}: {r['question']}")
            lines.append(f"A{i}: {r['answer']}")
        return "\n".join(lines)

    def __log(self, text: str) -> None:
        formatted_log(self.logger, "CONTEXT_EXTRACTION", text)
