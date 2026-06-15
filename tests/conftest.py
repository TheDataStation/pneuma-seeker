import os
import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session", autouse=True)
def postgres_container():
    """Spins up a single Postgres container for the entire test session and exposes
    its DSN via the TEST_POSTGRES_DSN environment variable so unittest.TestCase
    setUp methods can pick it up without needing pytest fixture injection."""
    with PostgresContainer("postgres:16-alpine") as pg:
        # driver=None gives a plain postgresql:// URL that psycopg3 accepts directly.
        dsn = pg.get_connection_url(driver=None)
        os.environ["TEST_POSTGRES_DSN"] = dsn
        yield pg
        del os.environ["TEST_POSTGRES_DSN"]
