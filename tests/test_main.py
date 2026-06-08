# tests/test_main.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class TestMainApplication(unittest.TestCase):
    """Integration and initialization tests for the core FastAPI entrypoint."""

    def setUp(self):
        # Patch Config initialization to prevent it from crashing on missing local paths
        self.config_patcher = patch("pneuma_seeker.shared.config.Config")
        self.mock_config_cls = self.config_patcher.start()

        # Stub out an expected value for CORS origins
        self.mock_config_instance = MagicMock()
        self.mock_config_instance.ALLOWED_ORIGINS = ["http://localhost:3000", "*"]
        self.mock_config_cls.return_value = self.mock_config_instance

        # Import the app instance after patching config to apply mocked values during setup
        from pneuma_seeker.main import app

        self.app = app
        self.client = TestClient(self.app)

    def tearDown(self):
        self.config_patcher.stop()

    def test_root_endpoint_returns_ok_status(self):
        """Verifies that the base URL / successfully signals application availability."""
        # Act
        response = self.client.get("/")

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_expected_routers_are_registered(self):
        """Ensures that the auth, chat, and indexing router groups are attached to the app layout."""
        # Extract path definitions from registered routes
        registered_paths = {route.path for route in self.app.routes}

        # Check explicitly for router prefix mappings
        self.assertTrue(
            any(path.startswith("/auth") for path in registered_paths),
            "Auth router missing.",
        )
        self.assertTrue(
            any(path.startswith("/chat") for path in registered_paths),
            "Chat router missing.",
        )
        self.assertTrue(
            any(path.startswith("/index") for path in registered_paths),
            "Indexing router missing.",
        )


if __name__ == "__main__":
    unittest.main()
