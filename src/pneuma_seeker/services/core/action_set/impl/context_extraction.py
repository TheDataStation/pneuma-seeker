import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import bm25s
from pandas import DataFrame

from pneuma_seeker.services.core.action_set.interfaces import Action, Applicable
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_code
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType
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


@dataclass(frozen=True)
class Resolution:
    """Holds a question this chat resolved, so a later question it answers can reuse it."""

    question: str
    tables: frozenset[str]
    versions: tuple[tuple[str, tuple], ...]  # (table id, version) when it was resolved
    code: str
    answer: str
    asked_by: str = ""  # e.g. "conductor", from the result table name


def compact(text: str, max_chars: int) -> str:
    """Returns `text` with runs of spaces collapsed (table previews pad columns), cut to `max_chars`."""
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    return text if len(text) <= max_chars else text[:max_chars] + " ...(truncated)"


class ContextExtraction(Action, Applicable):
    action_name = ActionNames.CONTEXT_EXTRACTION
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = "ENABLE_CONTEXT_EXTRACTION"
    order = 7
    show_in_prompt = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # This chat's resolved questions; the instance lives as long as the chat's
        # ActionSet, so other chats (and other runs) never see them.
        self.resolutions: list[Resolution] = []

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

        questions = [
            (i, u) for i, u in enumerate(uncertainties) if u.get("question", "")
        ]
        versions = {t.doc_id: self._version(t) for t in available_tables}

        def tables_of(uncertainty: dict) -> list[AbstractDocument]:
            table_ids: list[str] = uncertainty.get("table_ids", [])
            relevant_tables = [t for t in available_tables if t.doc_id in table_ids]
            return relevant_tables or available_tables

        reused, start_from = self._reuse(
            [(i, u["question"], {t.doc_id for t in tables_of(u)}) for i, u in questions],
            versions,
        )
        asked_by = result_table_name.split("_")[0]

        def resolve(i: int, uncertainty: dict) -> tuple[dict, list[str]]:
            question: str = uncertainty.get("question", "")
            short = f"{question[:80]}{'...' if len(question) > 80 else ''}"
            if i in reused:
                earlier = reused[i]
                self.__log(f"Reusing an earlier answer for uncertainty: {question} (answered for: {earlier.question})")
                logs = [f"[Context Extraction {i + 1}/{len(uncertainties)}] Reused an earlier answer: {short}"]
                return {"question": question, "answer": earlier.answer, "reused_from": earlier.question}, logs

            logs = [f"[Context Extraction {i + 1}/{len(uncertainties)}] Resolving: {short}"]
            self.__log(f"Resolving uncertainty: {question}")

            relevant_tables = tables_of(uncertainty)
            table_str = "\n\n".join(str(t) for t in relevant_tables)

            answer, retry_logs, code = self._resolve_single(
                question,
                table_str,
                self._question_table_name(result_table_name, i),
                execute_code_fn,
                start_from.get(i),
            )
            logs.extend(retry_logs)
            if code is not None:
                ids = frozenset(t.doc_id for t in relevant_tables)
                self.resolutions.append(Resolution(
                    question, ids, tuple(sorted((t, versions[t]) for t in ids)), code, answer, asked_by,
                ))
            return {"question": question, "answer": answer}, logs

        # Questions are independent, so they are resolved concurrently (the
        # time is spent waiting on the LLM); each writes its own result table.
        to_resolve = [q for q in questions if q[0] not in reused]
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(questions))) as pool:
                outcomes = list(pool.map(lambda q: resolve(*q), questions))
        finally:
            self._drop_question_tables(result_table_name, [i for i, _ in to_resolve])

        results = [result for result, _ in outcomes]
        log_messages = [msg for _, logs in outcomes for msg in logs]
        return self._format_results(results), log_messages

    # ------------------------------------------------------------------
    # Reuse of this chat's earlier answers
    # ------------------------------------------------------------------

    @staticmethod
    def _version(doc: AbstractDocument) -> tuple:
        """
        Returns what changes when a table is rebuilt (its shape, and for a table this
        chat built, its last transformation), so answers about an earlier version of
        it are not reused. Source tables do not change within a chat.
        """
        content = doc.content
        built = doc.retriever_type in (RetrieverType.MATERIALIZER, RetrieverType.CONDUCTOR)
        return (
            doc.last_node_id if built else None,
            getattr(content, "shape", None),
            tuple(str(c) for c in getattr(content, "columns", ())),
        )

    def _candidates(
        self, question: str, tables: set[str], versions: dict[str, tuple]
    ) -> list[int]:
        """
        Returns the indices in self.resolutions of up to CONTEXT_EXTRACTION_REUSE_TOP_K
        earlier answers that share a table with the question, every table of which is
        unchanged, best BM25 match first.
        """
        eligible = [
            k for k, r in enumerate(self.resolutions)
            if r.tables & tables and all(versions.get(t) == v for t, v in r.versions)
        ]
        if not eligible:
            return []
        retriever = bm25s.BM25()
        retriever.index(
            bm25s.tokenize([self.resolutions[k].question for k in eligible], stopwords="en", show_progress=False),
            show_progress=False,
        )
        query = bm25s.tokenize([question], stopwords="en", show_progress=False, return_ids=False)[0]
        if not query:
            return []
        scores = retriever.get_scores(query)
        ranked = sorted(zip(scores, eligible), key=lambda x: -x[0])
        return [k for score, k in ranked[: self.config.CONTEXT_EXTRACTION_REUSE_TOP_K] if score > 0]

    def _reuse(
        self, questions: list[tuple[int, str, set[str]]], versions: dict[str, tuple]
    ) -> tuple[dict[int, Resolution], dict[int, Resolution]]:
        """
        Returns ({question index: earlier answer to reuse}, {question index: earlier
        answer that covers part of it, to start from}), decided for all questions in
        one call that sees each question's candidates; ({}, {}) when there are none.
        """
        if self.config.CONTEXT_EXTRACTION_REUSE_TOP_K <= 0 or not self.resolutions:
            return {}, {}
        candidates = {i: self._candidates(q, tables, versions) for i, q, tables in questions}
        candidates = {i: c for i, c in candidates.items() if c}
        if not candidates:
            return {}, {}
        texts = {i: q for i, q, _ in questions}
        try:
            raw = "".join(self.language_model_api.chat(
                [
                    LLMMessage(role=Role.SYSTEM.value, content=self._get_reuse_system_prompt()),
                    LLMMessage(role=Role.USER.value, content=self._get_reuse_prompt(texts, candidates)),
                ],
                LLMOption(json_mode=True, stream=False, reasoning_effort=self.config.CONTEXT_EXTRACTION_REUSE_EFFORT),
            ))
            decisions = json.loads(raw).get("decisions", [])
        except Exception as exc:
            self.__log(f"Reuse check failed, resolving every question: {exc}")
            return {}, {}

        def earlier(decision: dict, key: str, i: int) -> int | None:
            try:
                k = int(decision.get(key)) - 1
            except (TypeError, ValueError):
                return None  # null: none
            return k if k in candidates.get(i, []) else None

        reused: dict[int, Resolution] = {}
        start_from: dict[int, Resolution] = {}
        decided: dict[int, str] = {}
        for decision in decisions if isinstance(decisions, list) else []:
            if not isinstance(decision, dict):
                continue
            try:
                i = int(decision.get("question")) - 1
            except (TypeError, ValueError):
                continue
            if (k := earlier(decision, "reuse", i)) is not None:
                reused[i], decided[i] = self.resolutions[k], f"reuse #{k + 1}"
            elif (k := earlier(decision, "start_from", i)) is not None:
                start_from[i], decided[i] = self.resolutions[k], f"resolve, starting from #{k + 1}"
        for i, ks in sorted(candidates.items()):
            shown = "; ".join(f"#{k + 1}: {self.resolutions[k].question[:100]}" for k in ks)
            self.__log(f"Reuse check, question {i + 1}: {texts[i][:100]} | candidates: {shown} "
                       f"| decision: {decided.get(i, 'resolve anew')}")
        self.__log(f"Reuse check: {len(reused)} of {len(questions)} question(s) answered earlier, "
                   f"{len(start_from)} started from an earlier answer")
        return reused, start_from

    def established_facts(
        self, asked_by: str, available_tables: list[AbstractDocument], max_chars: int = 800
    ) -> str:
        """
        Returns what checks by `asked_by` (e.g. "conductor") established about the
        available tables, unchanged since, as Q/A text; "" when there are none.
        """
        versions = {t.doc_id: self._version(t) for t in available_tables}
        facts = [
            r for r in self.resolutions
            if r.asked_by == asked_by and r.tables and all(versions.get(t) == v for t, v in r.versions)
        ]
        return "\n\n".join(
            f"- Tables: {', '.join(sorted(r.tables))}\n  Q: {r.question}\n  A: {compact(r.answer, max_chars)}"
            for r in facts
        )

    @staticmethod
    def _question_table_name(result_table_name: str, index: int) -> str:
        return f"{result_table_name}_{index + 1}"

    def _drop_question_tables(self, result_table_name: str, indices: list[int]) -> None:
        for i in indices:
            name = self._question_table_name(result_table_name, i)
            try:
                self.db_api.drop_relation(self.user_id, self.chat_id, name)
            except Exception as exc:
                self.__log(f"Cleanup warning ({name}): {exc}")

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _resolve_single(
        self,
        question: str,
        table_str: str,
        result_table_name: str,
        execute_code_fn: Callable[[str, str], Any],
        start_from: Resolution | None = None,
    ) -> tuple[str, list[str], str | None]:
        """
        Returns (answer, log messages, the code that produced it or None if it failed);
        `start_from` is an earlier check that covers part of the question.
        """
        messages = [
            LLMMessage(
                role=Role.SYSTEM.value,
                content=self._get_inner_loop_system_prompt(),
            ),
            LLMMessage(
                role=Role.USER.value,
                content=self._get_question_prompt(
                    question, table_str, result_table_name, start_from
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
                return result_str, log_messages, code
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
            None,
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

    @staticmethod
    def _get_reuse_system_prompt() -> str:
        return """You decide, for each new question about some tables, whether an earlier answer in this chat already answers it.

- Reuse an earlier answer only if it fully answers the new question as asked: same tables, same population, filters and definitions, and it contains every value the new question asks for.
- If the new question asks for more, for a different scope, or for a different way of computing something, do not reuse; it will be resolved anew.
- When it is resolved anew but an earlier answer covers part of it (the same join, columns or filters), name that earlier answer in "start_from"; its code is then the starting point.
- Rewording does not matter; what is asked does.

Respond with JSON only: {"decisions": [{"question": <question number>, "reuse": <earlier answer number, or null>, "start_from": <earlier answer number, or null>}, ...]}, one entry per new question."""

    def _get_reuse_prompt(
        self, questions: dict[int, str], candidates: dict[int, list[int]], max_chars: int = 1500
    ) -> str:
        def earlier(k: int) -> str:
            r = self.resolutions[k]
            return f"### Earlier answer {k + 1}\nQuestion: {r.question}\nAnswer: {compact(r.answer, max_chars)}"

        return "\n\n".join(
            f"## New question {i + 1}\n{questions[i]}\n\n" + "\n\n".join(earlier(k) for k in ks)
            for i, ks in sorted(candidates.items())
        )

    def _get_question_prompt(
        self,
        question: str,
        table_str: str,
        result_table_name: str,
        start_from: Resolution | None = None,
    ) -> str:
        earlier = (
            f"""
## Earlier Check in This Chat
It covers part of the question; build on its code (reuse its joins, filters and parsing) and compute only what is still needed.
Question: {start_from.question}
Code:
{start_from.code}
Answer: {compact(start_from.answer, 1500)}
"""
            if start_from
            else ""
        )
        return f"""## Question
{question}

## Relevant Tables
{table_str}
{earlier}
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
            if r.get("reused_from"):
                lines.append(f"A{i} (reused, answered earlier for: {r['reused_from']}): {r['answer']}")
            else:
                lines.append(f"A{i}: {r['answer']}")
        return "\n".join(lines)

    def __log(self, text: str) -> None:
        formatted_log(self.logger, "CONTEXT_EXTRACTION", text)
