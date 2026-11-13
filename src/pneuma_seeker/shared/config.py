# src/pneuma_seeker/shared/config.py
from os import getenv
from pathlib import Path

from dotenv import load_dotenv


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = getenv(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, not {value!r}")
    return value


class Config:
    def __init__(self, env_path: str | None = None) -> None:
        """Loads configuration from environment variables or a .env file."""
        if env_path:
            load_dotenv(env_path)

        # Language Model Settings
        self.LLM_PATH = getenv("LLM_PATH", "o3-2025-04-16")
        self.LLM_MAX_TOKENS = int(getenv("LLM_MAX_TOKENS", "200000"))
        # Unset: no temperature is sent and the model uses its default.
        temperature = getenv("LLM_TEMPERATURE")
        self.LLM_TEMPERATURE = float(temperature) if temperature else None
        self.EMBED_MODEL_PATH = getenv("EMBED_MODEL_PATH", "text-embedding-3-small")
        self.EMBEDDING_MAX_TOKENS = int(getenv("EMBEDDING_MAX_TOKENS", "1536"))

        # OpenAI / Azure OpenAI Settings
        self.OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")
        self.AZURE_OPENAI_API_KEY = getenv("AZURE_OPENAI_API_KEY", "")
        self.AZURE_OPENAI_ENDPOINT = getenv("AZURE_OPENAI_ENDPOINT", "")
        self.AZURE_API_VERSION = getenv("AZURE_API_VERSION", "2024-12-01-preview")
        self.USE_AZURE_LLM = getenv("USE_AZURE_LLM", "true").lower() == "true"
        self.USE_AZURE_EMBED_MODEL = (
            getenv("USE_AZURE_EMBED_MODEL", "true").lower() == "true"
        )

        # Gemini Settings
        self.GEMINI_API_KEY = getenv("GEMINI_API_KEY", "")

        # Claude Settings
        self.ANTHROPIC_API_KEY = getenv("ANTHROPIC_API_KEY", "")

        # Frontend-Backend Interaction Settings
        self.ALLOWED_ORIGINS = getenv("ALLOWED_ORIGINS", "*").split(",")
        self.TABLE_MAX_ROWS_DISPLAY = int(getenv("TABLE_MAX_ROWS_DISPLAY", "10"))
        self.SESSION_RESTORE_MAX_ROWS = int(getenv("SESSION_RESTORE_MAX_ROWS", "5000"))
        # How often to send an inert keepalive over an idle chat stream, so a
        # proxy/load balancer's idle-connection timeout doesn't kill the stream
        # while the backend is still mid-turn (e.g. waiting on one slow LLM call).
        self.STREAM_HEARTBEAT_INTERVAL_SECONDS = int(
            getenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "15")
        )

        # Core Configuration Settings
        self.MAX_CONDUCTOR_STEPS = int(getenv("MAX_CONDUCTOR_STEPS", "10"))
        self.MAX_MATERIALIZER_STEPS = int(getenv("MAX_MATERIALIZER_STEPS", "10"))
        self.ENABLE_MEMORY_PROFILING = (
            getenv("ENABLE_MEMORY_PROFILING", "false").lower() == "true"
        )

        # Action Settings
        ## Retrieval Action Settings
        self.ENABLE_WEB_SEARCH = getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
        self.ENABLE_WEB_CRAWL = getenv("ENABLE_WEB_CRAWL", "false").lower() == "true"
        self.WEB_CRAWL_MAX_CHARS = int(getenv("WEB_CRAWL_MAX_CHARS", "5000"))
        self.JOIN_PATH_EXTRACTION_ALPHA = float(
            getenv("JOIN_PATH_EXTRACTION_NAME_SIMILARITY_WEIGHT", "0.6")
        )
        self.JOIN_PATH_EXTRACTION_TOP_K = int(getenv("JOIN_PATH_EXTRACTION_TOP_K", "5"))
        self.TABLE_RETRIEVE_MAX_TOPICS = int(getenv("TABLE_RETRIEVE_MAX_TOPICS", "3"))
        self.TABLE_RETRIEVE_ENABLE_ENTITIES_RELEVANCE_BOOSTER = (
            getenv("TABLE_RETRIEVE_ENABLE_ENTITIES_RELEVANCE_BOOSTER", "true").lower()
            == "true"
        )

        ## Semantic Action Settings
        self.ENABLE_SEMANTIC_JOIN = (
            getenv("ENABLE_SEMANTIC_JOIN", "false").lower() == "true"
        )
        self.ENABLE_SEMANTIC_COL_GEN = (
            getenv("ENABLE_SEMANTIC_COL_GEN", "false").lower() == "true"
        )

        self.SEMANTIC_JOIN_TOP_K = 1
        self.SEMANTIC_JOIN_BATCH_SIZE = max(
            1, int(getenv("SEMANTIC_JOIN_BATCH_SIZE", "30"))
        )
        self.SEMANTIC_JOIN_DELIMITER = getenv("SEMANTIC_JOIN_DELIMITER", " [SEP] ")
        self.SEMANTIC_JOIN_ALPHA = float(getenv("SEMANTIC_JOIN_ALPHA", "0.5"))
        self.SEMANTIC_JOIN_MODE = getenv("SEMANTIC_JOIN_MODE", "jarowinkler")
        self.SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE = max(
            1, int(getenv("SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE", "60"))
        )
        self.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE = max(
            1, int(getenv("SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE", "20"))
        )

        ## Entity Resolution Settings
        self.ENTITY_RESOLUTION_JW_THRESHOLD = float(
            getenv("ENTITY_RESOLUTION_THRESHOLD", "0.75")
        )
        self.ENTITY_RESOLUTION_EMBEDDING_THRESHOLD = float(
            getenv("ENTITY_RESOLUTION_EMBEDDING_THRESHOLD", "0.60")
        )
        self.ENTITY_RESOLUTION_MODE = getenv("ENTITY_RESOLUTION_MODE", "jarowinkler")

        ## Other Action Settings
        self.ENABLE_CONTEXT_EXTRACTION = (
            getenv("ENABLE_CONTEXT_EXTRACTION", "true").lower() == "true"
        )
        self.MAX_CONTEXT_EXTRACTION_LOOP_STEPS = int(
            getenv("MAX_CONTEXT_EXTRACTION_LOOP_STEPS", "3")
        )
        self.MAX_RESULT_PREVIEW_ROWS = int(getenv("MAX_RESULT_PREVIEW_ROWS", "100"))
        # Max rows of model decisions (e.g., semantic join matches) embedded in
        # the end-to-end script per step; above it, the script loads the step's
        # result from a CSV snapshot instead.
        self.PROVENANCE_REPLAY_MAX_ROWS = int(
            getenv("PROVENANCE_REPLAY_MAX_ROWS", "100000")
        )
        self.ENABLE_PLAN_MODE = getenv("ENABLE_PLAN_MODE", "true").lower() == "true"
        # Multi-dataset mode: when on, a chat request with `multi_dataset: true` lets
        # Conductor/Materializer list the user's accessible datasets and retrieve
        # tables from several of them in one turn (federated over DuckDB ATTACH).
        self.ENABLE_MULTI_DATASET = (
            getenv("ENABLE_MULTI_DATASET", "true").lower() == "true"
        )

        ## Column Compaction Settings
        self.COLUMN_COMPACTION_ENABLED = (
            getenv("COLUMN_COMPACTION_ENABLED", "false").lower() == "true"
        )
        self.COLUMN_COMPACTION_SIM_THRESHOLD = float(
            getenv("COLUMN_COMPACTION_SIM_THRESHOLD", "0.6")
        )
        self.COLUMN_COMPACTION_ALPHA = float(getenv("COLUMN_COMPACTION_ALPHA", "0.85"))
        self.COLUMN_COMPACTION_USE_LLM_THRESHOLD = (
            getenv("COLUMN_COMPACTION_USE_LLM_THRESHOLD", "false").lower() == "true"
        )

        # Database Settings
        self.ENABLE_FINE_GRAINED_STATE_CHANGE_TRACKING = (
            getenv("ENABLE_FINE_GRAINED_STATE_CHANGE_TRACKING", "false").lower()
            == "true"
        )

        # Session Manager Settings
        self.SESSION_MANAGER_MAX_SESSIONS = int(
            getenv("SESSION_MANAGER_MAX_SESSIONS", "1000")
        )

        # Auth Settings
        self.AUTH_TOKEN_TTL_SECONDS = int(getenv("AUTH_TOKEN_TTL_SECONDS", "36000"))
        self.AUTH_PASSWORD_HASH_ITERATIONS = int(
            getenv("AUTH_PASSWORD_HASH_ITERATIONS", "200000")
        )

        # Tribal knowledge, retrieved per information need: "m1" | "bm25". TOP_K is
        # the "bm25" cap per need, and M1's fallback if the classifier fails.
        self.TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD = _choice(
            "TRIBAL_KNOWLEDGE_RETRIEVAL_METHOD", "m1", ("m1", "bm25")
        )
        self.TRIBAL_KNOWLEDGE_RETRIEVAL_TOP_K = int(
            getenv("TRIBAL_KNOWLEDGE_RETRIEVAL_TOP_K", "10")
        )
        self.TRIBAL_KNOWLEDGE_BATCH_TOKENS = int(  # M1 batch size (τ)
            getenv("TRIBAL_KNOWLEDGE_BATCH_TOKENS", "100")
        )

        # Table Retrieve over internal tables: "m1" (new_discovery.py) | "pneuma_retriever".
        self.TABLE_DISCOVERY_METHOD = _choice(
            "TABLE_DISCOVERY_METHOD", "m1", ("m1", "pneuma_retriever")
        )
        # M1 has no fixed k and returns tables with identical columns together, so
        # name-pattern enumeration is only needed with the Pneuma retriever.
        self.ENABLE_TABLE_ENUMERATION = self.TABLE_DISCOVERY_METHOD == "pneuma_retriever"
        self.TABLE_DISCOVERY_BATCH_TOKENS = int(  # M1 batch size (τ)
            getenv("TABLE_DISCOVERY_BATCH_TOKENS", "1500")
        )
        self.TABLE_DISCOVERY_MAX_COLUMNS = int(
            getenv("TABLE_DISCOVERY_MAX_COLUMNS", "50")
        )
        self.TABLE_DISCOVERY_GROUP_MAX_NAMES = int(
            getenv("TABLE_DISCOVERY_GROUP_MAX_NAMES", "100")
        )

        # Persisted retrieval indices (mounted by docker-compose).
        self.RETRIEVAL_INDEX_ROOT = getenv(
            "RETRIEVAL_INDEX_ROOT",
            str(Path(__file__).resolve().parent.parent / "services" / "db" / "indices"),
        )

        # M1: windows of M1_WINDOW_BATCHES batches; stop after M1_EMPTY_WINDOWS
        # empty windows or once M1_TOKEN_BUDGET tokens have been judged.
        self.M1_WINDOW_BATCHES = int(getenv("M1_WINDOW_BATCHES", "5"))
        self.M1_EMPTY_WINDOWS = int(getenv("M1_EMPTY_WINDOWS", "1"))
        self.M1_TOKEN_BUDGET = int(getenv("M1_TOKEN_BUDGET", "300000"))

        # Postgres Settings (for UserDB)
        self.POSTGRES_HOST = getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = getenv("POSTGRES_DB", "pneuma_users")
        self.POSTGRES_USER = getenv("POSTGRES_USER", "pneuma")
        self.POSTGRES_PASSWORD = getenv("POSTGRES_PASSWORD", "pneuma_password")
