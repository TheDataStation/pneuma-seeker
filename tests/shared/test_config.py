# tests/pneuma_seeker/shared/test_config.py
import unittest
import os
import sys
from pathlib import Path
import tempfile

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from pneuma_seeker.shared.config import Config


class ConfigTests(unittest.TestCase):
    """Comprehensive tests for Config class covering all configuration paths."""

    def setUp(self):
        """Store original environment and clear config-related variables."""
        self.original_env = os.environ.copy()
        # Clear all config-related environment variables
        self.config_vars = [
            "LLM_PATH",
            "EMBED_MODEL_PATH",
            "EMBEDDING_MAX_TOKENS",
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_API_VERSION",
            "USE_AZURE_LLM",
            "USE_AZURE_EMBED_MODEL",
            "ALLOWED_ORIGINS",
            "OPENWEBUI_BASE_URL",
            "OPENWEBUI_API_KEY",
            "MAX_CONDUCTOR_STEPS",
            "MAX_MATERIALIZER_STEPS",
            "ENABLE_WEB_SEARCH",
            "ENABLE_WEB_CRAWL",
            "WEB_CRAWL_MAX_CHARS",
            "PERSIST_CHAT_SESSION",
            "SEMANTIC_JOIN_BATCH_SIZE",
            "SEMANTIC_JOIN_DELIMITER",
            "SEMANTIC_JOIN_ALPHA",
            "SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE",
            "SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE",
        ]
        for var in self.config_vars:
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    # ========== Test Default Values ==========

    def test_all_defaults_no_env_no_file(self):
        """Test that all config values have proper defaults when no env vars or file provided."""
        cfg = Config()
        
        self.assertIsInstance(cfg.LLM_PATH, str)
        self.assertIsInstance(cfg.LLM_MAX_TOKENS, int)
        self.assertIsInstance(cfg.EMBED_MODEL_PATH, str)
        self.assertIsInstance(cfg.EMBEDDING_MAX_TOKENS, int)

        self.assertIsInstance(cfg.OPENAI_API_KEY, str)
        self.assertIsInstance(cfg.AZURE_OPENAI_API_KEY, str)
        self.assertIsInstance(cfg.AZURE_OPENAI_ENDPOINT, str)
        self.assertIsInstance(cfg.AZURE_API_VERSION, str)
        self.assertIsInstance(cfg.USE_AZURE_LLM, bool)
        self.assertIsInstance(cfg.USE_AZURE_EMBED_MODEL, bool)
        
        self.assertIsInstance(cfg.ALLOWED_ORIGINS, list)
        self.assertIsInstance(cfg.OPENWEBUI_BASE_URL, str)
        self.assertIsInstance(cfg.OPENWEBUI_API_KEY, str)
        self.assertIsInstance(cfg.TABLE_MAX_ROWS_DISPLAY, int)
        
        self.assertIsInstance(cfg.MAX_CONDUCTOR_STEPS, int)
        self.assertIsInstance(cfg.MAX_MATERIALIZER_STEPS, int)
        self.assertIsInstance(cfg.PERSIST_CHAT_SESSION, bool)
        self.assertIsInstance(cfg.DATA_SOURCES, list)
        for source in cfg.DATA_SOURCES:
            self.assertIsInstance(source, str)

        self.assertIsInstance(cfg.ENABLE_WEB_SEARCH, bool)
        self.assertIsInstance(cfg.ENABLE_WEB_CRAWL, bool)
        self.assertIsInstance(cfg.WEB_CRAWL_MAX_CHARS, int)
        self.assertIsInstance(cfg.JOIN_PATH_EXTRACTION_ALPHA, float)
        self.assertIsInstance(cfg.JOIN_PATH_EXTRACTION_TOP_K, int)
        self.assertIsInstance(cfg.TABLE_RETRIEVE_MAX_TOPICS, int)
        self.assertIsInstance(cfg.TABLE_RETRIEVE_ENABLE_ENTITIES_RELEVANCE_BOOSTER, bool)
        
        self.assertIsInstance(cfg.SEMANTIC_JOIN_TOP_K, int)
        self.assertIsInstance(cfg.SEMANTIC_JOIN_BATCH_SIZE, int)
        self.assertIsInstance(cfg.SEMANTIC_JOIN_DELIMITER, str)
        self.assertIsInstance(cfg.SEMANTIC_JOIN_ALPHA, float)
        self.assertIsInstance(cfg.SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE, int)
        self.assertIsInstance(cfg.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE, int)
        
        self.assertIsInstance(cfg.ENABLE_CONTEXT_EXTRACTION, bool)

    # ========== Test Environment Variable Loading ==========

    def test_config_loads_from_environment_variables(self):
        """Test that config properly loads from environment variables."""
        os.environ["OPENAI_API_KEY"] = "test_openai_key"
        os.environ["LLM_PATH"] = "gpt-4o"
        os.environ["EMBEDDING_MAX_TOKENS"] = "2048"
        os.environ["USE_AZURE_LLM"] = "false"
        os.environ["MAX_CONDUCTOR_STEPS"] = "10"
        os.environ["TABLE_MAX_ROWS_DISPLAY"] = "15"
        cfg = Config()
        
        self.assertEqual(cfg.OPENAI_API_KEY, "test_openai_key")
        self.assertEqual(cfg.LLM_PATH, "gpt-4o")
        self.assertEqual(cfg.EMBEDDING_MAX_TOKENS, 2048)
        self.assertFalse(cfg.USE_AZURE_LLM)
        self.assertEqual(cfg.MAX_CONDUCTOR_STEPS, 10)
        self.assertEqual(cfg.TABLE_MAX_ROWS_DISPLAY, 15)

    def test_config_loads_from_env_file(self):
        """Test that config loads from .env file when path is provided."""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=env_file_key\n")
            f.write("LLM_PATH=claude-3\n")
            f.write("EMBEDDING_MAX_TOKENS=512\n")
            f.write("USE_AZURE_EMBED_MODEL=false\n")
            f.write("MAX_CONDUCTOR_STEPS=5\n")
            env_file = f.name
        
        try:
            cfg = Config(env_path=env_file)
            
            self.assertEqual(cfg.OPENAI_API_KEY, "env_file_key")
            self.assertEqual(cfg.LLM_PATH, "claude-3")
            self.assertEqual(cfg.EMBEDDING_MAX_TOKENS, 512)
            self.assertFalse(cfg.USE_AZURE_EMBED_MODEL)
            self.assertEqual(cfg.MAX_CONDUCTOR_STEPS, 5)
        finally:
            os.unlink(env_file)

    def test_env_file_path_none_uses_environment(self):
        """Test that passing None for env_path uses environment variables."""
        os.environ["OPENAI_API_KEY"] = "env_var_key"
        
        cfg = Config(env_path=None)
        
        self.assertEqual(cfg.OPENAI_API_KEY, "env_var_key")

    def test_nonexistent_env_file_doesnt_crash(self):
        """Test that providing nonexistent env file path doesn't crash."""
        cfg = Config(env_path="/nonexistent/path/.env")
        
        # Should still load defaults
        self.assertIsInstance(cfg.LLM_PATH, str)

    # ========== Test Boolean Parsing ==========

    def test_boolean_parsing_lowercase_true(self):
        """Test boolean parsing with lowercase 'true'."""
        os.environ["USE_AZURE_LLM"] = "true"
        os.environ["USE_AZURE_EMBED_MODEL"] = "true"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["ENABLE_WEB_CRAWL"] = "true"
        os.environ["PERSIST_CHAT_SESSION"] = "true"
        
        cfg = Config()
        
        self.assertTrue(cfg.USE_AZURE_LLM)
        self.assertTrue(cfg.USE_AZURE_EMBED_MODEL)
        self.assertTrue(cfg.ENABLE_WEB_SEARCH)
        self.assertTrue(cfg.ENABLE_WEB_CRAWL)
        self.assertTrue(cfg.PERSIST_CHAT_SESSION)

    def test_boolean_parsing_lowercase_false(self):
        """Test boolean parsing with lowercase 'false'."""
        os.environ["USE_AZURE_LLM"] = "false"
        os.environ["USE_AZURE_EMBED_MODEL"] = "false"
        os.environ["ENABLE_WEB_SEARCH"] = "false"
        os.environ["ENABLE_WEB_CRAWL"] = "false"
        os.environ["PERSIST_CHAT_SESSION"] = "false"
        
        cfg = Config()
        
        self.assertFalse(cfg.USE_AZURE_LLM)
        self.assertFalse(cfg.USE_AZURE_EMBED_MODEL)
        self.assertFalse(cfg.ENABLE_WEB_SEARCH)
        self.assertFalse(cfg.ENABLE_WEB_CRAWL)
        self.assertFalse(cfg.PERSIST_CHAT_SESSION)

    def test_boolean_parsing_uppercase_true(self):
        """Test boolean parsing with uppercase 'TRUE'."""
        os.environ["USE_AZURE_LLM"] = "TRUE"
        os.environ["ENABLE_WEB_SEARCH"] = "TRUE"
        
        cfg = Config()
        
        self.assertTrue(cfg.USE_AZURE_LLM)
        self.assertTrue(cfg.ENABLE_WEB_SEARCH)

    def test_boolean_parsing_mixed_case(self):
        """Test boolean parsing with mixed case."""
        os.environ["USE_AZURE_LLM"] = "True"
        os.environ["ENABLE_WEB_SEARCH"] = "False"
        
        cfg = Config()
        
        self.assertTrue(cfg.USE_AZURE_LLM)
        self.assertFalse(cfg.ENABLE_WEB_SEARCH)

    def test_boolean_parsing_invalid_value_treated_as_false(self):
        """Test that invalid boolean values are treated as false."""
        os.environ["USE_AZURE_LLM"] = "yes"
        os.environ["ENABLE_WEB_SEARCH"] = "1"
        os.environ["PERSIST_CHAT_SESSION"] = "invalid"
        
        cfg = Config()
        
        self.assertFalse(cfg.USE_AZURE_LLM)
        self.assertFalse(cfg.ENABLE_WEB_SEARCH)
        self.assertFalse(cfg.PERSIST_CHAT_SESSION)

    # ========== Test Integer Parsing ==========

    def test_integer_parsing_valid_values(self):
        """Test integer parsing with valid numeric strings."""
        os.environ["EMBEDDING_MAX_TOKENS"] = "3000"
        os.environ["MAX_CONDUCTOR_STEPS"] = "15"
        os.environ["MAX_MATERIALIZER_STEPS"] = "200"
        os.environ["WEB_CRAWL_MAX_CHARS"] = "10000"
        os.environ["SEMANTIC_JOIN_BATCH_SIZE"] = "50"
        os.environ["SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE"] = "100"
        os.environ["SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE"] = "20"
        
        cfg = Config()
        
        self.assertEqual(cfg.EMBEDDING_MAX_TOKENS, 3000)
        self.assertEqual(cfg.MAX_CONDUCTOR_STEPS, 15)
        self.assertEqual(cfg.MAX_MATERIALIZER_STEPS, 200)
        self.assertEqual(cfg.WEB_CRAWL_MAX_CHARS, 10000)
        self.assertEqual(cfg.SEMANTIC_JOIN_BATCH_SIZE, 50)
        self.assertEqual(cfg.SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE, 100)
        self.assertEqual(cfg.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE, 20)

    def test_integer_parsing_zero_and_negative_values(self):
        """Test integer parsing with zero and negative values."""
        os.environ["EMBEDDING_MAX_TOKENS"] = "0"
        os.environ["MAX_CONDUCTOR_STEPS"] = "-5"
        
        cfg = Config()
        
        self.assertEqual(cfg.EMBEDDING_MAX_TOKENS, 0)
        self.assertEqual(cfg.MAX_CONDUCTOR_STEPS, -5)

    def test_batch_size_minimum_value_enforced(self):
        """Test that batch sizes are enforced to be at least 1."""
        os.environ["SEMANTIC_JOIN_BATCH_SIZE"] = "0"
        os.environ["SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE"] = "-5"
        os.environ["SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE"] = "-10"
        
        cfg = Config()
        
        # All batch sizes should be at least 1
        self.assertGreaterEqual(cfg.SEMANTIC_JOIN_BATCH_SIZE, 1)
        self.assertGreaterEqual(cfg.SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE, 1)
        self.assertGreaterEqual(cfg.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE, 1)

    def test_batch_size_valid_positive_values(self):
        """Test batch sizes with valid positive values."""
        os.environ["SEMANTIC_JOIN_BATCH_SIZE"] = "100"
        os.environ["SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE"] = "200"
        os.environ["SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE"] = "50"
        
        cfg = Config()
        
        self.assertEqual(cfg.SEMANTIC_JOIN_BATCH_SIZE, 100)
        self.assertEqual(cfg.SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE, 200)
        self.assertEqual(cfg.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE, 50)

    # ========== Test Float Parsing ==========

    def test_float_parsing_valid_values(self):
        """Test float parsing with valid values."""
        os.environ["SEMANTIC_JOIN_ALPHA"] = "0.75"
        
        cfg = Config()
        
        self.assertAlmostEqual(cfg.SEMANTIC_JOIN_ALPHA, 0.75)

    def test_float_parsing_edge_values(self):
        """Test float parsing with edge values."""
        os.environ["SEMANTIC_JOIN_ALPHA"] = "0.0"
        cfg1 = Config()
        self.assertAlmostEqual(cfg1.SEMANTIC_JOIN_ALPHA, 0.0)
        
        os.environ["SEMANTIC_JOIN_ALPHA"] = "1.0"
        cfg2 = Config()
        self.assertAlmostEqual(cfg2.SEMANTIC_JOIN_ALPHA, 1.0)
        
        os.environ["SEMANTIC_JOIN_ALPHA"] = "0.123456"
        cfg3 = Config()
        self.assertAlmostEqual(cfg3.SEMANTIC_JOIN_ALPHA, 0.123456)

    # ========== Test String Parsing ==========

    def test_string_values_with_special_characters(self):
        """Test string values containing special characters."""
        os.environ["OPENAI_API_KEY"] = "sk-test_key-with-dashes_and_underscores"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
        os.environ["OPENWEBUI_BASE_URL"] = "http://localhost:3000/api/v1/"
        
        cfg = Config()
        
        self.assertEqual(cfg.OPENAI_API_KEY, "sk-test_key-with-dashes_and_underscores")
        self.assertEqual(cfg.AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com/")
        self.assertEqual(cfg.OPENWEBUI_BASE_URL, "http://localhost:3000/api/v1/")

    def test_empty_string_values(self):
        """Test that empty string values are preserved."""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        
        cfg = Config()
        
        self.assertEqual(cfg.OPENAI_API_KEY, "")
        self.assertEqual(cfg.AZURE_OPENAI_API_KEY, "")

    def test_string_with_whitespace(self):
        """Test strings with leading/trailing whitespace."""
        os.environ["OPENAI_API_KEY"] = "  test_key  "
        
        cfg = Config()
        
        # Environment variables preserve whitespace
        self.assertEqual(cfg.OPENAI_API_KEY, "  test_key  ")

    # ========== Test List Parsing ==========

    def test_allowed_origins_single_value(self):
        """Test ALLOWED_ORIGINS with single value."""
        os.environ["ALLOWED_ORIGINS"] = "*"
        
        cfg = Config()
        
        self.assertEqual(cfg.ALLOWED_ORIGINS, ["*"])

    def test_allowed_origins_multiple_values(self):
        """Test ALLOWED_ORIGINS with multiple comma-separated values."""
        os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,https://example.com,https://app.example.com"
        
        cfg = Config()
        
        self.assertEqual(cfg.ALLOWED_ORIGINS, [
            "http://localhost:3000",
            "https://example.com",
            "https://app.example.com"
        ])

    def test_allowed_origins_with_spaces(self):
        """Test ALLOWED_ORIGINS parsing with spaces around commas."""
        os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000, https://example.com , https://app.example.com"
        
        cfg = Config()
        
        # Spaces are preserved in split()
        self.assertEqual(cfg.ALLOWED_ORIGINS, [
            "http://localhost:3000",
            " https://example.com ",
            " https://app.example.com"
        ])

    def test_allowed_origins_empty_string(self):
        """Test ALLOWED_ORIGINS with empty string."""
        os.environ["ALLOWED_ORIGINS"] = ""
        
        cfg = Config()
        
        self.assertEqual(cfg.ALLOWED_ORIGINS, [""])

    def test_semantic_join_delimiter_default(self):
        """Test SEMANTIC_JOIN_DELIMITER default value."""
        cfg = Config()
        
        self.assertIsInstance(cfg.SEMANTIC_JOIN_DELIMITER, str)

    def test_semantic_join_delimiter_custom(self):
        """Test SEMANTIC_JOIN_DELIMITER with custom value."""
        os.environ["SEMANTIC_JOIN_DELIMITER"] = " | "
        
        cfg = Config()
        
        self.assertEqual(cfg.SEMANTIC_JOIN_DELIMITER, " | ")

    # ========== Test All Values Together ==========

    def test_comprehensive_custom_config(self):
        """Test comprehensive configuration with all custom values."""
        # Set all environment variables
        os.environ["LLM_PATH"] = "gpt-4o-mini"
        os.environ["EMBED_MODEL_PATH"] = "text-embedding-3-large"
        os.environ["EMBEDDING_MAX_TOKENS"] = "3072"
        os.environ["OPENAI_API_KEY"] = "sk-custom-key"
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://custom.openai.azure.com/"
        os.environ["AZURE_API_VERSION"] = "2025-01-01"
        os.environ["USE_AZURE_LLM"] = "false"
        os.environ["USE_AZURE_EMBED_MODEL"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,https://prod.example.com"
        os.environ["OPENWEBUI_BASE_URL"] = "http://openwebui:8080/"
        os.environ["OPENWEBUI_API_KEY"] = "openwebui-key"
        os.environ["MAX_CONDUCTOR_STEPS"] = "10"
        os.environ["MAX_MATERIALIZER_STEPS"] = "150"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["ENABLE_WEB_CRAWL"] = "false"
        os.environ["WEB_CRAWL_MAX_CHARS"] = "8000"
        os.environ["PERSIST_CHAT_SESSION"] = "false"
        os.environ["SEMANTIC_JOIN_BATCH_SIZE"] = "40"
        os.environ["SEMANTIC_JOIN_DELIMITER"] = " || "
        os.environ["SEMANTIC_JOIN_ALPHA"] = "0.75"
        os.environ["SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE"] = "80"
        os.environ["SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE"] = "15"
        
        cfg = Config()
        
        # Verify all values
        self.assertEqual(cfg.LLM_PATH, "gpt-4o-mini")
        self.assertEqual(cfg.EMBED_MODEL_PATH, "text-embedding-3-large")
        self.assertEqual(cfg.EMBEDDING_MAX_TOKENS, 3072)
        self.assertEqual(cfg.OPENAI_API_KEY, "sk-custom-key")
        self.assertEqual(cfg.AZURE_OPENAI_API_KEY, "azure-key")
        self.assertEqual(cfg.AZURE_OPENAI_ENDPOINT, "https://custom.openai.azure.com/")
        self.assertEqual(cfg.AZURE_API_VERSION, "2025-01-01")
        self.assertFalse(cfg.USE_AZURE_LLM)
        self.assertFalse(cfg.USE_AZURE_EMBED_MODEL)
        self.assertEqual(cfg.ALLOWED_ORIGINS, ["http://localhost:3000", "https://prod.example.com"])
        self.assertEqual(cfg.OPENWEBUI_BASE_URL, "http://openwebui:8080/")
        self.assertEqual(cfg.OPENWEBUI_API_KEY, "openwebui-key")
        self.assertEqual(cfg.MAX_CONDUCTOR_STEPS, 10)
        self.assertEqual(cfg.MAX_MATERIALIZER_STEPS, 150)
        self.assertTrue(cfg.ENABLE_WEB_SEARCH)
        self.assertFalse(cfg.ENABLE_WEB_CRAWL)
        self.assertEqual(cfg.WEB_CRAWL_MAX_CHARS, 8000)
        self.assertFalse(cfg.PERSIST_CHAT_SESSION)
        self.assertEqual(cfg.SEMANTIC_JOIN_BATCH_SIZE, 40)
        self.assertEqual(cfg.SEMANTIC_JOIN_DELIMITER, " || ")
        self.assertAlmostEqual(cfg.SEMANTIC_JOIN_ALPHA, 0.75)
        self.assertEqual(cfg.SEMANTIC_COL_GEN_ROW_PROCESSING_BATCH_SIZE, 80)
        self.assertEqual(cfg.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE, 15)

    def test_data_sources_is_list(self):
        """Test that DATA_SOURCES is always a list."""
        cfg = Config()
        
        self.assertIsInstance(cfg.DATA_SOURCES, list)

    def test_semantic_join_top_k_is_hardcoded(self):
        """Test that SEMANTIC_JOIN_TOP_K is hardcoded to 1."""
        cfg = Config()
        
        self.assertEqual(cfg.SEMANTIC_JOIN_TOP_K, 1)

    # ========== Test Edge Cases ==========

    def test_multiple_config_instances_independent(self):
        """Test that multiple Config instances can have different values."""
        os.environ["OPENAI_API_KEY"] = "first_key"
        cfg1 = Config()
        
        os.environ["OPENAI_API_KEY"] = "second_key"
        cfg2 = Config()
        
        # cfg1 should still have the old value (captured at init)
        self.assertEqual(cfg1.OPENAI_API_KEY, "first_key")
        self.assertEqual(cfg2.OPENAI_API_KEY, "second_key")

    def test_config_with_very_large_integer_values(self):
        """Test config with very large integer values."""
        os.environ["EMBEDDING_MAX_TOKENS"] = "999999"
        os.environ["MAX_CONDUCTOR_STEPS"] = "1000000"
        
        cfg = Config()
        
        self.assertEqual(cfg.EMBEDDING_MAX_TOKENS, 999999)
        self.assertEqual(cfg.MAX_CONDUCTOR_STEPS, 1000000)

    def test_config_value_types(self):
        """Test that config values have correct types."""
        cfg = Config()
        
        # String types
        self.assertIsInstance(cfg.LLM_PATH, str)
        self.assertIsInstance(cfg.EMBED_MODEL_PATH, str)
        self.assertIsInstance(cfg.OPENAI_API_KEY, str)
        self.assertIsInstance(cfg.AZURE_API_VERSION, str)
        self.assertIsInstance(cfg.SEMANTIC_JOIN_DELIMITER, str)
        
        # Integer types
        self.assertIsInstance(cfg.EMBEDDING_MAX_TOKENS, int)
        self.assertIsInstance(cfg.MAX_CONDUCTOR_STEPS, int)
        self.assertIsInstance(cfg.MAX_MATERIALIZER_STEPS, int)
        self.assertIsInstance(cfg.WEB_CRAWL_MAX_CHARS, int)
        self.assertIsInstance(cfg.SEMANTIC_JOIN_TOP_K, int)
        self.assertIsInstance(cfg.SEMANTIC_JOIN_BATCH_SIZE, int)
        
        # Float types
        self.assertIsInstance(cfg.SEMANTIC_JOIN_ALPHA, float)
        
        # Boolean types
        self.assertIsInstance(cfg.USE_AZURE_LLM, bool)
        self.assertIsInstance(cfg.USE_AZURE_EMBED_MODEL, bool)
        self.assertIsInstance(cfg.ENABLE_WEB_SEARCH, bool)
        self.assertIsInstance(cfg.ENABLE_WEB_CRAWL, bool)
        self.assertIsInstance(cfg.PERSIST_CHAT_SESSION, bool)
        
        # List types
        self.assertIsInstance(cfg.ALLOWED_ORIGINS, list)
        self.assertIsInstance(cfg.DATA_SOURCES, list)


if __name__ == "__main__":
    unittest.main()
