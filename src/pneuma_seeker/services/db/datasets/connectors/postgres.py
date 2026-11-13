from urllib.parse import quote

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from pneuma_seeker.services.db.datasets.connectors.base import (
    DatasetConnector,
    digest,
    quote_ident,
    source_alias,
    sql_literal,
)


class _PostgresConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    host: str
    port: int = 5432
    dbname: str
    user: str
    password: str = ""
    schema_: str = Field(default="public", alias="schema")
    sslmode: str | None = None


class PostgresConnector(DatasetConnector):
    """
    Represents one schema of a Postgres database, attached read-only through
    DuckDB's postgres extension. Changes inside the database aren't detected
    by the fingerprint, only changes to the connection settings.
    """

    type = "postgres"
    label = "Postgres"
    ConfigModel = _PostgresConfig
    SECRET_FIELDS = frozenset({"password"})

    def connection_uri(self) -> str:
        """Returns the libpq URI, with credentials percent-encoded."""
        cfg = self.cfg
        uri = (
            f"postgresql://{quote(cfg.user, safe='')}:{quote(cfg.password, safe='')}"
            f"@{cfg.host}:{cfg.port}/{quote(cfg.dbname, safe='')}"
        )
        return uri + (f"?sslmode={quote(cfg.sslmode, safe='')}" if cfg.sslmode else "")

    def scrub(self, text: str) -> str:
        return super().scrub(text.replace(self.connection_uri(), "postgresql://…"))

    def relations(self, conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, str]:
        conn.execute("INSTALL postgres; LOAD postgres")
        source = source_alias(alias)
        conn.execute(
            f"ATTACH {sql_literal(self.connection_uri())} AS {quote_ident(source)} (TYPE POSTGRES, READ_ONLY)"
        )
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = ? AND table_schema = ? ORDER BY table_name",
            [source, self.cfg.schema_],
        ).fetchall()
        schema = quote_ident(self.cfg.schema_)
        return {name: f"{quote_ident(source)}.{schema}.{quote_ident(name)}" for (name,) in rows}

    def fingerprint(self) -> str:
        cfg = self.cfg
        return "postgres:" + digest([cfg.host, str(cfg.port), cfg.dbname, cfg.user, cfg.schema_])
