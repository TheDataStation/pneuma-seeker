from pneuma_seeker.services.db.datasets.connectors.base import (
    DatasetConnector,
    DatasetUnavailable,
    quote_ident,
    source_alias,
    sql_literal,
)
from pneuma_seeker.services.db.datasets.connectors.files import (
    CSVDirConnector,
    DuckDBFileConnector,
    ParquetDirConnector,
    SQLiteFileConnector,
)
from pneuma_seeker.services.db.datasets.connectors.postgres import PostgresConnector

CONNECTOR_TYPES: dict[str, type[DatasetConnector]] = {
    cls.type: cls
    for cls in (
        ParquetDirConnector,
        CSVDirConnector,
        DuckDBFileConnector,
        SQLiteFileConnector,
        PostgresConnector,
    )
}


def build(connector_type: str, config: dict) -> DatasetConnector:
    """Builds the connector of `connector_type`. Raises ValueError for an unknown type or bad config."""
    cls = CONNECTOR_TYPES.get(connector_type)
    if cls is None:
        raise ValueError(f"Unknown connector type: {connector_type!r}")
    return cls(config)


__all__ = [
    "CONNECTOR_TYPES",
    "DatasetConnector",
    "DatasetUnavailable",
    "build",
    "quote_ident",
    "source_alias",
    "sql_literal",
]
