import re
from abc import ABC
from enum import Enum
from typing import Any

import json

import pandas as pd
from pandas import DataFrame


class RetrieverType(Enum):
    """
    Represents all types of data in Pneuma-Seeker.
    """

    PNEUMA_RETRIEVER = "Pneuma"  # Internal tables
    CONDUCTOR = "Conductor"
    ENUMERATOR = "Enumerator"
    MATERIALIZER = "Materializer"
    DOCUMENT_DB = "Document DB"
    WEB_SEARCH = "Web Search"
    USER = "User"  # External tables
    WEB_CRAWL = "Web Crawl"


class AbstractDocument(ABC):
    """
    Represents (abstractly) the unit of information in Processor.
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: Any,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: (
            str | None
        ) = None,  # Keep track of last transformation that returns this data
    ):
        self.doc_id = doc_id
        self.retriever_type = retriever_type
        self.content = content
        self.metadata = metadata
        self.path = path
        self.last_node_id = last_node_id

    def __eq__(self, other):
        if not isinstance(other, AbstractDocument):
            return NotImplemented
        return (self.doc_id, self.retriever_type) == (
            other.doc_id,
            other.retriever_type,
        )

    def __hash__(self):
        return hash((self.doc_id, self.retriever_type))

    def __str__(self) -> str:
        return f"ID: {self.doc_id} ; Content: {self.content}"


class Knowledge(AbstractDocument):
    """
    Represents some form of knowledge from users.

    - retriever_type: RetrieverType.KNOWLEDGE_BASE
    - content: str
    - metadata: {"type": "local/global", "user": "..."}
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: str,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)


class Table(AbstractDocument):
    """
    Represents a table.

    - retriever_type: RetrieverType.PNEUMA (Pneuma is the current table discovery system)
    - content: DataFrame
    - metadata: {"table_name": "...", "dataset_name": "..."}
    """

    _NUMERIC_TYPE_PREFIXES = (
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "DECIMAL",
        "DOUBLE",
        "FLOAT",
        "REAL",
    )
    _DATE_TYPE_PREFIXES = ("DATE", "TIME", "TIMESTAMP")
    _DIGIT_RE = re.compile(r"\d+")
    _ALPHA_RE = re.compile(r"[A-Za-z]+")

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: DataFrame,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        """Wrap a DataFrame with retrieval metadata (table name, dataset name,
        DuckDB column types, etc.) used to render it as text for the LLM."""
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)

    def _duckdb_col_types(self) -> dict[str, str]:
        """Parse the DuckDB column-type JSON stored in metadata["column_types"],
        if present. Returns {} when missing/unparseable."""
        duckdb_col_types: dict[str, str] = {}
        col_types_raw = self.metadata.get("column_types")
        if isinstance(col_types_raw, str) and col_types_raw.strip():
            try:
                parsed = json.loads(col_types_raw)
                if isinstance(parsed, dict):
                    duckdb_col_types = {
                        str(k): str(v) for k, v in parsed.items() if v is not None
                    }
            except Exception:
                duckdb_col_types = {}
        return duckdb_col_types

    def _col_header_entry(
        self, col: str, dtype, duckdb_col_types: dict[str, str], series: pd.Series
    ) -> str:
        """Render one "col (type)" header entry, adding "; e.g., <value>" with
        the column's first non-null value when it isn't numeric (used by the
        "zero_rows" strategy, which has no sample rows of its own)."""
        resolved_type = duckdb_col_types.get(col, dtype)
        if self._type_family(series, duckdb_col_types.get(col)) == "numeric":
            return f"{col} ({resolved_type})"
        non_null = series.dropna()
        if non_null.empty:
            return f"{col} ({resolved_type})"
        return f"{col} ({resolved_type}; e.g., {non_null.iloc[0]})"

    def _render_header(
        self, duckdb_col_types: dict[str, str], with_examples: bool = False
    ) -> str:
        """Build the 'Table ...\ncol: col1 (type) | col2 (type) | ...' header
        shared by every strategy, including any description/keywords hints.
        When with_examples is True, non-numeric columns get an inline
        "; e.g., <value>" hint instead of a plain "(type)"."""
        table: DataFrame = self.content
        table_id = self.doc_id
        if "dataset_name" in self.metadata:
            table_id += f" (dataset: {self.metadata['dataset_name']})"

        if with_examples:
            cols = " | ".join(
                self._col_header_entry(col, dtype, duckdb_col_types, table[col])
                for col, dtype in zip(table.columns, table.dtypes)
            )
        else:
            cols = " | ".join(
                f"{col} ({duckdb_col_types.get(col, dtype)})"
                for col, dtype in zip(table.columns, table.dtypes)
            )

        if "description" in self.metadata and "keywords_existence" in self.metadata:
            return f"Table {table_id} ({self.metadata['description']}; include these keywords: {self.metadata['keywords_existence']}):\ncol: {cols}"
        elif "description" in self.metadata:
            return f"Table {table_id} ({self.metadata['description']}):\ncol: {cols}"
        elif "keywords_existence" in self.metadata:
            return (
                f"Table {table_id} "
                f"(include these keywords: {self.metadata['keywords_existence']}):\n"
                f"col: {cols}"
            )
        return f"Table {table_id}:\ncol: {cols}"

    def _type_family(self, series: pd.Series, duckdb_type: str | None) -> str:
        """Classify a column as "numeric", "date", or "categorical". Prefers
        the authoritative DuckDB type string; falls back to the pandas dtype,
        then to sniffing whether un-parsed strings look like dates."""
        if duckdb_type:
            upper = duckdb_type.upper()
            if upper.startswith(self._NUMERIC_TYPE_PREFIXES):
                return "numeric"
            if upper.startswith(self._DATE_TYPE_PREFIXES):
                return "date"
            return "categorical"
        if pd.api.types.is_numeric_dtype(series.dtype):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return "date"
        # No DuckDB type available (not every caller populates it): sniff
        # whether this looks like a date column of un-parsed strings before
        # falling back to categorical.
        non_null = series.dropna()
        if len(non_null) > 0:
            parsed_ratio = (
                pd.to_datetime(non_null, errors="coerce", format="mixed").notna().mean()
            )
            if parsed_ratio >= 0.8:
                return "date"
        return "categorical"

    def _shape_signature(self, value: str) -> str:
        """Normalize a value to its format "shape" (digit runs -> "#", letter
        runs -> "@") so format-alike values (e.g. "1", "2") group together and
        differently-shaped values (e.g. "4 or 5") stand out as distinct."""
        shape = self._DIGIT_RE.sub("#", value)
        shape = self._ALPHA_RE.sub("@", shape)
        return shape

    def _diverse_categorical_values(
        self, series: pd.Series, k: int = 3
    ) -> tuple[list[str], int]:
        """Pick up to k representative values for a categorical column: the
        single most-frequent value first, then one representative per other
        value shape (rarest shape first), plus the total distinct count so
        the caller can render an "etc. (N distinct)" suffix."""
        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            return [], 0

        counts = values.value_counts()
        n_distinct = len(counts)

        shape_groups: dict[str, list[str]] = {}
        for value in counts.index:
            shape_groups.setdefault(self._shape_signature(value), []).append(value)

        most_frequent = counts.index[0]
        representatives = [most_frequent]
        seen_shapes = {self._shape_signature(most_frequent)}

        group_freq = {
            shape: sum(counts[v] for v in vals) for shape, vals in shape_groups.items()
        }
        remaining_shapes = sorted(
            (shape for shape in shape_groups if shape not in seen_shapes),
            key=lambda shape: group_freq[shape],
        )
        for shape in remaining_shapes:
            if len(representatives) >= k:
                break
            representatives.append(shape_groups[shape][0])
            seen_shapes.add(shape)

        return representatives[:k], n_distinct

    def _numeric_range_summary(self, series: pd.Series, k: int = 3) -> str:
        """Summarize a numeric column as a "min-max" range, or as a plain
        comma-separated value list when it only has k or fewer distinct
        values (e.g. a small numeric enum like {1, 2, 3})."""
        values = series.dropna()
        if values.empty:
            return "no values"
        distinct = sorted(values.unique().tolist())
        if len(distinct) <= k:
            return ", ".join(str(v) for v in distinct)
        return f"{distinct[0]}–{distinct[-1]}"

    def _date_range_summary(self, series: pd.Series) -> str | None:
        """Summarize a date-like column as a "min-max" date range, or None if
        none of its values parse as dates."""
        parsed = pd.to_datetime(series, errors="coerce").dropna()
        if parsed.empty:
            return None
        return f"{parsed.min().date()}–{parsed.max().date()}"

    def _format_categorical_line(
        self, col: str, values: list[str], n_distinct: int
    ) -> str:
        """Render a categorical column's "col: v1, v2, etc. (N distinct)" line,
        omitting the "etc." suffix when all distinct values are shown."""
        if not values:
            return f"{col}: no values"
        rendered = ", ".join(values)
        if n_distinct > len(values):
            return f"{col}: {rendered}, etc. ({n_distinct} distinct)"
        return f"{col}: {rendered}"

    def _column_profile_lines(
        self, table: DataFrame, duckdb_col_types: dict[str, str]
    ) -> list[str]:
        """Render one summary line per column (range for numeric/date,
        diverse value list for categorical), dispatched by type family."""
        lines = []
        for col in table.columns:
            series = table[col]
            family = self._type_family(series, duckdb_col_types.get(col))
            if family == "numeric":
                lines.append(f"{col}: range {self._numeric_range_summary(series)}")
                continue
            if family == "date":
                summary = self._date_range_summary(series)
                if summary is not None:
                    lines.append(f"{col}: range {summary}")
                    continue
            values, n_distinct = self._diverse_categorical_values(series)
            lines.append(self._format_categorical_line(col, values, n_distinct))
        return lines

    def _body_column_profile(
        self, table: DataFrame, duckdb_col_types: dict[str, str]
    ) -> list[str]:
        """Body for the "column_profile" strategy: per-column value/range
        summaries instead of sample rows."""
        return ["values:"] + [
            f"  {line}" for line in self._column_profile_lines(table, duckdb_col_types)
        ]

    def _body_sample_rows(self, table: DataFrame, n: int) -> list[str]:
        """Randomly sample up to n rows (random_state=42, sorted back to
        original order) and render each as a "sample row: ..." line."""
        sample_rows = table.sample(min(n, len(table)), random_state=42).sort_index()
        lines = []
        for _, row in sample_rows.iterrows():
            row_str = " | ".join(str(row[col]) for col in table.columns)
            lines.append(f"sample row: {row_str}")
        return lines

    def _body_five_rows(
        self, table: DataFrame, duckdb_col_types: dict[str, str]
    ) -> list[str]:
        """Body for the "five_rows" strategy: up to 5 randomly sampled rows."""
        return self._body_sample_rows(table, 5)

    def _body_one_row(
        self, table: DataFrame, duckdb_col_types: dict[str, str]
    ) -> list[str]:
        """Body for the "one_row" strategy: a single randomly sampled row."""
        return self._body_sample_rows(table, 1)

    def _body_zero_rows(
        self, table: DataFrame, duckdb_col_types: dict[str, str]
    ) -> list[str]:
        """Body for the "zero_rows" strategy: no rows at all, since the
        header itself already carries a sample value per non-numeric column."""
        return []

    _STRATEGIES = {
        "column_profile": _body_column_profile,
        "five_rows": _body_five_rows,
        "one_row": _body_one_row,
        "zero_rows": _body_zero_rows,
    }

    def to_str(self, strategy: str = "one_row") -> str:
        """Render this table as text using the given strategy: "column_profile"
        (per-column summaries, no rows), "five_rows", "one_row" (plain
        random-sampled rows, default), or "zero_rows" (no rows at all; the
        header's non-numeric columns get an inline example value instead)."""
        table: DataFrame = self.content
        duckdb_col_types = self._duckdb_col_types()
        header = self._render_header(
            duckdb_col_types, with_examples=(strategy == "zero_rows")
        )

        if len(table) == 0:
            return header

        body_lines = self._STRATEGIES[strategy](self, table, duckdb_col_types)
        return "\n".join([header, *body_lines])

    def __str__(self) -> str:
        """Render this table using the "one_row" strategy."""
        return self.to_str(strategy="one_row")


class TableContext(AbstractDocument):
    """
    Represents table context.

    - retriever_type: RetrieverType.PNEUMA (Pneuma is the current table discovery system)
    - content: str
    - metadata: {"table_name": "...", "dataset_name": "...", "type": "..."}
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: str,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)


class Text(AbstractDocument):
    """
    Represents textual document.
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: str,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)


def convert_retrieval_results_to_str(retrieval_results: list[AbstractDocument]):
    representation = ""
    seen_docs: set[str] = set()
    topic_documents: dict[str, list[AbstractDocument]] = {}
    for result in retrieval_results:
        topic = result.metadata.get("topic", "unknown")
        if topic not in topic_documents:
            topic_documents[topic] = []
        topic_documents[topic].append(result)

    for topic, docs in topic_documents.items():
        representation += f"- Topic: {topic}\n"
        for doc in docs:
            if doc.doc_id in seen_docs:
                representation += f"  - Table {doc.doc_id}\n"
            else:
                representation += f"  - ```{str(doc)}```\n"
                seen_docs.add(doc.doc_id)
    return representation.strip()
