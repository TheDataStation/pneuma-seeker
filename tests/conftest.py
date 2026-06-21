import os
import pytest
from urllib.parse import urlparse
from testcontainers.postgres import PostgresContainer

_POSTGRES_ENV_KEYS = (
    "TEST_POSTGRES_DSN",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
)


@pytest.fixture(scope="session", autouse=True)
def postgres_container():
    """Spins up a single Postgres container for the entire test session.

    Exports both TEST_POSTGRES_DSN and the individual POSTGRES_* variables so
    that Config() (which reads those individual vars) routes to the container
    instead of any locally-running Postgres.  Without this, SessionIndex inside
    PneumaDB connects to the real DB and pollutes session_index with test data.
    """
    with PostgresContainer("postgres:16-alpine") as pg:
        dsn = pg.get_connection_url(driver=None)  # plain postgresql:// for psycopg3
        parsed = urlparse(dsn)

        # Snapshot existing values so we can restore them after the session
        original = {k: os.environ.get(k) for k in _POSTGRES_ENV_KEYS}

        os.environ["TEST_POSTGRES_DSN"] = dsn
        os.environ["POSTGRES_HOST"] = parsed.hostname or "localhost"
        os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
        os.environ["POSTGRES_USER"] = parsed.username or "test"
        os.environ["POSTGRES_PASSWORD"] = parsed.password or "test"
        os.environ["POSTGRES_DB"] = parsed.path.lstrip("/")

        yield pg

        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
