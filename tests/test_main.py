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
            any(path.startswith("/api/auth") for path in registered_paths),
            "Auth router missing.",
        )
        self.assertTrue(
            any(path.startswith("/api/chat") for path in registered_paths),
            "Chat router missing.",
        )
        self.assertTrue(
            any(path.startswith("/api/index") for path in registered_paths),
            "Indexing router missing.",
        )

    def test_all_router_routes_live_under_api_prefix(self):
        """Every route contributed by the auth/chat/indexing routers must be
        under /api — the routes contributed by those three routers are
        exactly the ones known to potentially collide with a same-named
        frontend page route (e.g. the chat router's DELETE /{chat_id}
        colliding with the frontend's own GET /chat/{chatId} page route).
        Framework-level routes (docs/openapi/health check) are intentionally
        excluded — they aren't part of any application router."""
        framework_paths = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc", "/"}
        registered_paths = {route.path for route in self.app.routes}
        app_router_paths = registered_paths - framework_paths

        self.assertTrue(app_router_paths, "Expected at least one application route.")
        for path in app_router_paths:
            self.assertTrue(
                path.startswith("/api/"),
                f"Route {path!r} is not namespaced under /api — it risks "
                "colliding with a frontend page route of the same name.",
            )

    def test_chat_router_does_not_register_a_bare_chat_id_route(self):
        """Regression: a production hard refresh on a chat page (GET
        /chat/{chatId}, served by the frontend) got misrouted to the backend
        and matched the chat router's DELETE /{chat_id} route (mounted at
        bare /chat/{chat_id} before the /api prefix was added), returning
        405 Method Not Allowed instead of the chat page."""
        registered_paths = {route.path for route in self.app.routes}
        self.assertNotIn("/chat/{chat_id}", registered_paths)
        self.assertIn("/api/chat/{chat_id}", registered_paths)


if __name__ == "__main__":
    unittest.main()
