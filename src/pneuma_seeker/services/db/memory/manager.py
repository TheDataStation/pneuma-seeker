# src/pneuma_seeker/services/db/memory/manager.py
from logging import Logger
from typing import Any
from uuid import uuid4

import bm25s
import psycopg
from psycopg.types.json import Jsonb
from Stemmer import Stemmer

from pneuma_seeker.services.db.memory.models import MemoryEntryRecord
from pneuma_seeker.shared.config import Config

_UNSET = object()

_COLUMNS = (
    "memory_id",
    "memory_type",
    "group_id",
    "user_id",
    "dataset_name",
    "key_text",
    "content",
    "extra",
    "source",
    "source_user_id",
    "chat_id",
    "created_at",
    "updated_at",
)


class MemoryManager:
    """
    Postgres-backed store for all memory-layer entries (tribal knowledge, user
    preferences, agent-learning cache, table/column metadata).

    Postgres is the single source of truth for CRUD; retrieval ranking is done
    with an ephemeral, in-memory BM25S index built from the (small, pre-filtered)
    result set on every search() call rather than a persisted on-disk index —
    this avoids the write-then-rebuild staleness/no-delete problems of the old
    on-disk BM25S corpus approach while still satisfying the "no embeddings"
    requirement for ranking.
    """

    def __init__(self, config: Config, logger: Logger, dsn: str | None = None) -> None:
        self.config = config
        self.logger = logger
        self.stemmer = Stemmer("english")

        # Build DSN from config, same credentials as UserDB (same Postgres instance)
        if dsn:
            self.dsn = dsn
        else:
            self.dsn = (
                f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}"
                f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
            )

        # Create schema immediately (mirrors SessionIndex) so any code path that
        # constructs a MemoryManager directly — not just the app's startup
        # lifespan — can rely on memory_entries existing. Raises on connection
        # failure so PneumaDB can catch it and disable the memory layer instead
        # of crashing when Postgres is unavailable (e.g. local dev).
        self.init_db()

    def init_db(self) -> None:
        """Initializes schema. Safe to call on every startup — statements use IF NOT EXISTS."""
        self.__log("Initializing memory schema")
        con = self._get_connection()
        try:
            with con.transaction():
                con.execute("""
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        memory_id       VARCHAR PRIMARY KEY,
                        memory_type     VARCHAR NOT NULL,
                        group_id        VARCHAR,
                        user_id         VARCHAR,
                        dataset_name    VARCHAR,
                        key_text        VARCHAR,
                        content         TEXT NOT NULL,
                        extra           JSONB,
                        source          VARCHAR NOT NULL,
                        source_user_id  VARCHAR,
                        chat_id         VARCHAR,
                        created_at      TIMESTAMPTZ DEFAULT NOW(),
                        updated_at      TIMESTAMPTZ DEFAULT NOW()
                    );
                    """)
                con.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_entries_group
                        ON memory_entries (memory_type, group_id);
                    """)
                con.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_entries_dataset
                        ON memory_entries (memory_type, dataset_name);
                    """)
                con.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_entries_user
                        ON memory_entries (memory_type, user_id);
                    """)
        finally:
            con.close()

    def _get_connection(self) -> psycopg.Connection:
        """Opens a fresh autocommit psycopg connection (mirrors UserDB._get_connection)."""
        return psycopg.connect(self.dsn, autocommit=True)

    def create_entry(
        self,
        memory_type: str,
        content: str,
        source: str,
        source_user_id: str | None = None,
        *,
        group_id: str | None = None,
        user_id: str | None = None,
        dataset_name: str | None = None,
        key_text: str | None = None,
        extra: dict[str, Any] | None = None,
        chat_id: str | None = None,
    ) -> MemoryEntryRecord:
        """Creates a new memory entry and returns its record."""
        memory_id = str(uuid4())
        con = self._get_connection()
        try:
            row = con.execute(
                f"""
                INSERT INTO memory_entries (
                    memory_id, memory_type, group_id, user_id, dataset_name,
                    key_text, content, extra, source, source_user_id, chat_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {", ".join(_COLUMNS)};
                """,
                (
                    memory_id,
                    memory_type,
                    group_id,
                    user_id,
                    dataset_name,
                    key_text,
                    content,
                    Jsonb(extra) if extra is not None else None,
                    source,
                    source_user_id,
                    chat_id,
                ),
            ).fetchone()
        finally:
            con.close()
        assert row is not None
        return self._to_record(row)

    def get_entry(self, memory_id: str) -> MemoryEntryRecord | None:
        """Returns a single memory entry by id, or None if not found."""
        con = self._get_connection()
        try:
            row = con.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM memory_entries WHERE memory_id = %s",
                (memory_id,),
            ).fetchone()
        finally:
            con.close()
        return self._to_record(row) if row else None

    def get_by_key(
        self, memory_type: str, dataset_name: str, key_text: str
    ) -> MemoryEntryRecord | None:
        """Returns the entry exactly matching (memory_type, dataset_name, key_text), if any."""
        con = self._get_connection()
        try:
            row = con.execute(
                f"""
                SELECT {", ".join(_COLUMNS)} FROM memory_entries
                WHERE memory_type = %s AND dataset_name = %s AND key_text = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (memory_type, dataset_name, key_text),
            ).fetchone()
        finally:
            con.close()
        return self._to_record(row) if row else None

    def update_entry(self, memory_id: str, content: Any = _UNSET) -> bool:
        """Updates a memory entry's content. Returns False if the entry doesn't exist."""
        if content is _UNSET:
            return self.get_entry(memory_id) is not None

        con = self._get_connection()
        try:
            result = con.execute(
                """
                UPDATE memory_entries SET content = %s, updated_at = NOW()
                WHERE memory_id = %s
                RETURNING memory_id;
                """,
                (content, memory_id),
            ).fetchone()
            return result is not None
        finally:
            con.close()

    def delete_entry(self, memory_id: str) -> bool:
        """Deletes a memory entry. Returns False if it did not exist."""
        con = self._get_connection()
        try:
            result = con.execute(
                "DELETE FROM memory_entries WHERE memory_id = %s RETURNING memory_id;",
                (memory_id,),
            ).fetchone()
            return result is not None
        finally:
            con.close()

    def list_entries(
        self,
        memory_type: str,
        *,
        group_ids: list[str] | None = None,
        user_id: str | None = None,
        dataset_name: str | None = None,
        match_null_dataset: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Returns a paginated list of entries matching the given scope filters, newest first.
        Return shape mirrors WorkspaceManager's {"items", "has_more", "next_offset"} convention.

        `match_null_dataset`: when True, a row with dataset_name IS NULL also matches
        regardless of the `dataset_name` filter — for memory types where dataset_name is
        an optional tag rather than a partition key (tribal_knowledge/user_preference,
        which can be created without picking a dataset). Dataset-scoped types
        (agent_learning/table_metadata/column_metadata) always require an exact match,
        since those rows are never created without a dataset_name in the first place.
        """
        clauses = ["memory_type = %s"]
        params: list[Any] = [memory_type]

        if group_ids is not None:
            clauses.append("group_id = ANY(%s)")
            params.append(group_ids)
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(user_id)
        if dataset_name is not None:
            if match_null_dataset:
                clauses.append("(dataset_name = %s OR dataset_name IS NULL)")
            else:
                clauses.append("dataset_name = %s")
            params.append(dataset_name)

        where_sql = " AND ".join(clauses)
        params.extend([limit + 1, offset])

        con = self._get_connection()
        try:
            rows = con.execute(
                f"""
                SELECT {", ".join(_COLUMNS)} FROM memory_entries
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            ).fetchall()
        finally:
            con.close()

        has_more = len(rows) > limit
        rows = rows[:limit]
        return {
            "items": [self._to_record(row) for row in rows],
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
        }

    def search(
        self,
        memory_type: str,
        query: str,
        *,
        group_ids: list[str] | None = None,
        user_id: str | None = None,
        dataset_name: str | None = None,
        match_null_dataset: bool = False,
        k: int = 5,
    ) -> list[tuple[MemoryEntryRecord, float]]:
        """
        Returns up to k entries matching the scope filters, ranked by BM25 relevance
        to `query`. Builds the BM25 index in-memory for this call only — the
        candidate set (a single group/dataset/user's entries) is expected to be
        small, so there is no need to persist or incrementally maintain an index.
        Scores are normalized to [0, 1] (1 = best match in this candidate set).
        """
        candidates = self.list_entries(
            memory_type,
            group_ids=group_ids,
            user_id=user_id,
            dataset_name=dataset_name,
            match_null_dataset=match_null_dataset,
            limit=5000,
            offset=0,
        )["items"]
        if not query.strip() or not candidates:
            return [(record, 0.0) for record in candidates[:k]]

        corpus_text = [record.content for record in candidates]
        # Corpus entries are opaque index markers, not the text itself — bm25s hands
        # back whatever object was passed in at the matching position, so we only
        # need enough to map a hit back to its MemoryEntryRecord.
        corpus_json = [{"idx": i} for i in range(len(candidates))]
        corpus_tokens = bm25s.tokenize(
            corpus_text, stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        retriever = bm25s.BM25(corpus=corpus_json)
        retriever.index(corpus_tokens, show_progress=False)

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        max_k = min(k, len(candidates))
        results, scores = retriever.retrieve(query_tokens, k=max_k, show_progress=False)

        hits = results[0]
        raw_scores = scores[0]
        max_score = max(raw_scores) if len(raw_scores) else 0.0
        min_score = min(raw_scores) if len(raw_scores) else 0.0

        ranked: list[tuple[MemoryEntryRecord, float]] = []
        for hit, raw_score in zip(hits, raw_scores):
            normalized = (
                1.0
                if max_score == min_score
                else (raw_score - min_score) / (max_score - min_score)
            )
            ranked.append((candidates[int(hit["idx"])], float(normalized)))
        return ranked

    def replace_dataset_rows(
        self,
        dataset_name: str,
        memory_type: str,
        rows: list[dict[str, Any]],
        uploaded_by: str | None = None,
    ) -> int:
        """
        Replaces all entries of `memory_type` for `dataset_name` with `rows` in a single
        transaction. Each row must provide "key_text" and "content"; used by CSV upload
        for table_metadata/column_metadata, where a re-upload should fully supersede the
        previous set rather than accumulate duplicates.
        """
        con = self._get_connection()
        try:
            with con.transaction():
                con.execute(
                    "DELETE FROM memory_entries WHERE memory_type = %s AND dataset_name = %s;",
                    (memory_type, dataset_name),
                )
                for row in rows:
                    con.execute(
                        """
                        INSERT INTO memory_entries (
                            memory_id, memory_type, dataset_name, key_text, content,
                            source, source_user_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            str(uuid4()),
                            memory_type,
                            dataset_name,
                            row["key_text"],
                            row["content"],
                            "csv_upload",
                            uploaded_by,
                        ),
                    )
        finally:
            con.close()
        return len(rows)

    def _to_record(self, row: tuple) -> MemoryEntryRecord:
        return MemoryEntryRecord(**dict(zip(_COLUMNS, row)))

    def __log(self, message: str) -> None:
        self.logger.info(f"[MemoryManager] {message}")
