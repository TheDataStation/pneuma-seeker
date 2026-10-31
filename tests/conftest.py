import os
from urllib.parse import urlparse

import pytest
from testcontainers.postgres import PostgresContainer

_POSTGRES_ENV_KEYS = (
    "TEST_POSTGRES_DSN",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
)

_postgres: PostgresContainer | None = None
_original_env: dict[str, str | None] = {}


def pytest_configure(config: pytest.Config) -> None:
    """Spins up a single Postgres container for the entire test session.

    Exports both TEST_POSTGRES_DSN and the individual POSTGRES_* variables so
    that Config() (which reads those individual vars) routes to the container
    instead of any locally-running Postgres.  Without this, SessionIndex inside
    PneumaDB connects to the real DB and pollutes session_index with test data.

    This runs in pytest_configure (not a session fixture) because modules such
    as routers/chat.py build their Config at import time, and pytest imports
    test modules during collection, before any fixture runs.
    """
    global _postgres, _original_env
    _postgres = PostgresContainer("postgres:16-alpine")
    _postgres.start()
    dsn = _postgres.get_connection_url(driver=None)  # plain postgresql:// for psycopg3
    parsed = urlparse(dsn)

    # Snapshot existing values so we can restore them after the session
    _original_env = {k: os.environ.get(k) for k in _POSTGRES_ENV_KEYS}

    os.environ["TEST_POSTGRES_DSN"] = dsn
    os.environ["POSTGRES_HOST"] = parsed.hostname or "localhost"
    os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
    os.environ["POSTGRES_USER"] = parsed.username or "test"
    os.environ["POSTGRES_PASSWORD"] = parsed.password or "test"
    os.environ["POSTGRES_DB"] = parsed.path.lstrip("/")


def pytest_unconfigure(config: pytest.Config) -> None:
    if _postgres is not None:
        _postgres.stop()
    for k, v in _original_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
