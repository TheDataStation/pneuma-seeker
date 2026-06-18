# src/pneuma_seeker/services/db/workspaces/session_index.py
import datetime
from logging import Logger
from typing import Any

import psycopg

from pneuma_seeker.shared.config import Config


class SessionIndex:
    """
    Postgres-backed index for chat session metadata.

    Why this exists
    ---------------
    Each chat session lives in its own DuckDB file (workspace_db_path/user_id/chat_id/ws.db).
    DuckDB is excellent for per-session query work, but listing or searching sessions requires
    opening every ws.db file on every API call — O(chats) file I/Os per request. At 1k chats
    per user that is 1k connection open/close cycles just to render the sidebar.

    This class maintains a lightweight Postgres table that is updated on every persist_session()
    call and queried by the listing and search endpoints.  A single indexed Postgres query replaces
    the O(n) filesystem scan, making both operations effectively O(log n) regardless of scale.

    The ws.db files remain the authoritative source of truth for workspace state; this table is
    purely a read-fast projection of the metadata the UI needs.
    """

    def __init__(self, config: Config, logger: Logger, dsn: str | None = None) -> None:
        self.config = config
        self.logger = logger

        # Build DSN from config, same credentials as UserDB (same Postgres instance)
        if dsn:
            self.dsn = dsn
        else:
            self.dsn = (
                f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}"
                f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
            )

        # Create schema immediately; raises on connection failure so PneumaDB can catch it
        # and fall back to the filesystem scan when Postgres is unavailable (e.g. local dev)
        self.init_db()

    def _get_connection(self) -> psycopg.Connection:
        """Opens a fresh autocommit psycopg connection (mirrors UserDB._get_connection)."""
        return psycopg.connect(self.dsn, autocommit=True)

    def init_db(self) -> None:
        """
        Creates the session_index table and its supporting indexes if they don't already exist.
        Safe to call on every startup — all statements use IF NOT EXISTS.
        """
        self.__log("Initializing session index schema")
        con = self._get_connection()
        try:
            with con.transaction():
                # pg_trgm enables fast GIN-indexed ILIKE searches.
                # Without this extension the ILIKE query on search_content falls back to
                # a full sequential scan even with an index present.
                con.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

                con.execute("""
                    CREATE TABLE IF NOT EXISTS session_index (
                        user_id        VARCHAR     NOT NULL,
                        chat_id        VARCHAR     NOT NULL,

                        -- Set once on the first persist_session call; never overwritten.
                        -- Derived into the chat title shown in the UI sidebar.
                        first_message  TEXT,

                        -- Accumulates every user + assistant message over the life of the chat.
                        -- Keeps the full message history searchable without opening ws.db files.
                        search_content TEXT        NOT NULL DEFAULT '',

                        last_active    TIMESTAMPTZ NOT NULL DEFAULT now(),
                        dataset_name   VARCHAR,
                        PRIMARY KEY (user_id, chat_id)
                    );
                """)

                # Composite index on (user_id, last_active DESC) so listing a user's sessions
                # newest-first is an O(log n) index scan rather than a full table scan.
                con.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_index_user_last_active
                        ON session_index (user_id, last_active DESC);
                """)

                # GIN trigram index converts ILIKE '%query%' from a sequential scan
                # (O(rows × content_length)) into a fast bitmap scan.
                con.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_index_search
                        ON session_index USING gin (search_content gin_trgm_ops);
                """)
        finally:
            con.close()

    def upsert(
        self,
        user_id: str,
        chat_id: str,
        new_user_message: str,
        new_assistant_message: str,
        dataset_name: str | None,
    ) -> None:
        """
        Called after every successful persist_session() to keep the index in sync.

        First call for a (user_id, chat_id) pair
        -----------------------------------------
        - Inserts a new row with first_message = new_user_message (becomes the chat title).
        - Initialises search_content with both the user and assistant messages.

        Subsequent calls
        ----------------
        - COALESCE keeps the original first_message intact (only sets it when NULL).
        - Appends new messages to search_content so the full conversation stays searchable.
        - Bumps last_active so recency ordering stays accurate.
        """
        # Both messages are concatenated into search_content so a single ILIKE can
        # match anywhere in the conversation without knowing which turn to look in.
        new_content = f"{new_user_message} {new_assistant_message}"

        con = self._get_connection()
        try:
            con.execute(
                """
                INSERT INTO session_index (user_id, chat_id, first_message, search_content, last_active, dataset_name)
                VALUES (%s, %s, %s, %s, now(), %s)
                ON CONFLICT (user_id, chat_id) DO UPDATE SET
                    -- Keep the very first message as the permanent title anchor;
                    -- COALESCE returns the first non-NULL so existing rows are unaffected.
                    first_message  = COALESCE(session_index.first_message, excluded.first_message),
                    -- Append rather than replace so old messages remain searchable
                    search_content = session_index.search_content || ' ' || excluded.search_content,
                    last_active    = now()
                """,
                (user_id, chat_id, new_user_message, new_content, dataset_name),
            )
        finally:
            con.close()

    def list_sessions(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """
        Returns a paginated list of the user's chat sessions ordered by most-recent activity.

        Replaces the O(n) filesystem scan in WorkspaceManager.get_user_chat_sessions() with a
        single O(log n) Postgres index scan.  Return shape is identical to the old method so
        the router and frontend require no changes.
        """
        con = self._get_connection()
        try:
            # Fetch limit+1 rows so we can detect whether another page exists without
            # running a separate COUNT(*) query.
            rows = con.execute(
                """
                SELECT chat_id, first_message, last_active
                FROM session_index
                WHERE user_id = %s
                ORDER BY last_active DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit + 1, offset),
            ).fetchall()
        finally:
            con.close()

        has_more = len(rows) > limit
        rows = rows[:limit]  # trim the extra probe row before formatting

        return {
            "chats": [self._format_row(row) for row in rows],
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
        }

    def search_sessions(
        self, user_id: str, query: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """
        Case-insensitive search across all messages in the user's chat sessions.

        The GIN trigram index on search_content makes this fast — ILIKE is accelerated by
        pg_trgm so performance degrades as O(matches), not O(total messages).

        Replaces the O(n × messages_per_chat) per-file ILIKE loop in
        WorkspaceManager.search_chat_sessions().  Return shape is identical.
        """
        con = self._get_connection()
        try:
            rows = con.execute(
                """
                SELECT chat_id, first_message, last_active
                FROM session_index
                WHERE user_id = %s AND search_content ILIKE %s
                ORDER BY last_active DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, f"%{query}%", limit + 1, offset),
            ).fetchall()
        finally:
            con.close()

        has_more = len(rows) > limit
        rows = rows[:limit]

        return {
            "chats": [self._format_row(row) for row in rows],
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
        }

    def delete_session(self, user_id: str, chat_id: str) -> None:
        """
        Removes a session from the index after its ws.db directory has been deleted.
        Safe to call even if the row doesn't exist (e.g. the session pre-dates this index
        and was never backfilled by the migration script).
        """
        con = self._get_connection()
        try:
            con.execute(
                "DELETE FROM session_index WHERE user_id = %s AND chat_id = %s",
                (user_id, chat_id),
            )
        finally:
            con.close()

    def _format_row(self, row: tuple) -> dict[str, Any]:
        """
        Converts a (chat_id, first_message, last_active) Postgres row into the dict
        shape the API returns, replicating the title-truncation and timestamp formatting
        that WorkspaceManager previously applied after reading each individual ws.db file.
        """
        chat_id, first_message, last_active_dt = row

        content = first_message or ""
        # Mirror the truncation logic the frontend relies on for the sidebar title
        title = f"{content[:20]}..." if len(content) > 20 else content

        # Normalise to UTC ISO-8601 with a Z suffix — same format the frontend was already receiving
        last_active_str = (
            last_active_dt.astimezone(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )
            + "Z"
            if hasattr(last_active_dt, "astimezone")
            else str(last_active_dt)
        )

        return {"id": chat_id, "title": title, "lastActive": last_active_str}

    def __log(self, message: str) -> None:
        self.logger.info(f"[SessionIndex] {message}")
