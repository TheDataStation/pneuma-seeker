# src/pneuma_seeker/shared/config.py
from os import getenv

from dotenv import load_dotenv


class Config:
    def __init__(self, env_path: str | None = None) -> None:
        """Loads configuration from environment variables or a .env file."""
        if env_path:
            load_dotenv(env_path)

        # Language Model Settings
        self.LLM_PATH = getenv("LLM_PATH", "o3-2025-04-16")
        self.LLM_MAX_TOKENS = int(getenv("LLM_MAX_TOKENS", "200000"))
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
        self.USE_SKILLS_AGENT = getenv("USE_SKILLS_AGENT", "false").lower() == "true"
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
        self.ENABLE_DS_SKEPTIC = getenv("ENABLE_DS_SKEPTIC", "false").lower() == "true"
        self.MAX_DS_SKEPTIC_ROUNDS = int(getenv("MAX_DS_SKEPTIC_ROUNDS", "2"))
        self.MAX_DS_SKEPTIC_CE_CALLS = int(getenv("MAX_DS_SKEPTIC_CE_CALLS", "5"))
        self.ENABLE_PLAN_MODE = getenv("ENABLE_PLAN_MODE", "true").lower() == "true"

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

        # Postgres Settings (for UserDB)
        self.POSTGRES_HOST = getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = getenv("POSTGRES_DB", "pneuma_users")
        self.POSTGRES_USER = getenv("POSTGRES_USER", "pneuma")
        self.POSTGRES_PASSWORD = getenv("POSTGRES_PASSWORD", "pneuma_password")
