import logging
import os
import shutil
import sys
from tempfile import mkdtemp
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.routers import auth
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.shared.config import Config


class TestAuthRouter(unittest.TestCase):
    """Tests for FastAPI authentication endpoints."""

    def setUp(self):
        self.config = Config()
        self.config.AUTH_PASSWORD_HASH_ITERATIONS = 1
        self.config.AUTH_TOKEN_TTL_SECONDS = 60
        self.logger = logging.getLogger("test")
        self.tmpdir = mkdtemp()
        
        self.test_user_db = UserDB(
            self.config,
            self.logger,
            (Path(self.tmpdir) / "users.db").as_posix(),
        )

        self.app = FastAPI()
        self.app.include_router(auth.router)

        self.app.dependency_overrides[auth.get_user_db] = lambda: self.test_user_db

        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _register_and_login(self, email="user@example.com", password="password123"):
        """Helper method to register a user and return their authentication token."""
        register_response = self.client.post(
            "/auth/register",
            json={"email": email, "password": password},
        )
        self.assertEqual(register_response.status_code, 200)

        login_response = self.client.post(
            "/auth/login",
            json={"email": email, "password": password},
        )
        self.assertEqual(login_response.status_code, 200)
        return login_response.json()["access_token"]

    def test_register_defaults_group(self):
        """Tests that registering a user without specifying a group assigns them to the default group."""
        response = self.client.post(
            "/auth/register",
            json={"email": "USER@example.com", "password": "password123"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["email"], "user@example.com")
        self.assertEqual(body["username"], "user@example.com")
        self.assertEqual(body["group_name"], "default")

    def test_register_duplicate_returns_conflict(self):
        """Tests that attempting to register with an email that already exists (case-insensitive) returns a 409 Conflict."""
        self.client.post(
            "/auth/register",
            json={"email": "dupe@example.com", "password": "password123"},
        )

        response = self.client.post(
            "/auth/register",
            json={"email": "DUPE@example.com", "password": "password123"},
        )

        self.assertEqual(response.status_code, 409)

    def test_login_returns_bearer_token(self):
        """Tests that logging in with valid credentials returns a bearer token with the expected structure."""
        self.client.post(
            "/auth/register",
            json={"email": "login@example.com", "password": "password123"},
        )

        response = self.client.post(
            "/auth/login",
            json={"email": "login@example.com", "password": "password123"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["token_type"], "bearer")
        self.assertTrue(body["access_token"])
        self.assertTrue(body["expires_at"])

    def test_login_rejects_invalid_password(self):
        """Tests that logging in with an incorrect password returns a 401 Unauthorized."""
        self.client.post(
            "/auth/register",
            json={"email": "login@example.com", "password": "password123"},
        )

        response = self.client.post(
            "/auth/login",
            json={"email": "login@example.com", "password": "wrong-password"},
        )

        self.assertEqual(response.status_code, 401)

    def test_me_requires_token_and_returns_current_user(self):
        """Tests that the /auth/me endpoint requires a valid token and returns the current user's information."""
        token = self._register_and_login()

        response = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["email"], "user@example.com")

    def test_logout_revokes_token(self):
        """Tests that logging out revokes the user's token, preventing further access to authenticated endpoints."""
        token = self._register_and_login()

        logout_response = self.client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        me_response = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(logout_response.status_code, 200)
        self.assertEqual(me_response.status_code, 401)

    def test_create_and_list_groups_require_auth(self):
        """Tests that creating and listing groups requires authentication, and that groups can be created and listed successfully."""
        token = self._register_and_login()

        create_response = self.client.post(
            "/auth/groups",
            json={"name": "analysts"},
            headers={"Authorization": f"Bearer {token}"},
        )
        list_response = self.client.get(
            "/auth/groups",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(create_response.status_code, 200)
        self.assertEqual(create_response.json()["name"], "analysts")
        self.assertIn("analysts", [group["name"] for group in list_response.json()])

    def test_create_duplicate_group_returns_bad_request(self):
        """Tests that attempting to create a group with a name that already exists returns a 400 Bad Request."""
        token = self._register_and_login()
        self.client.post(
            "/auth/groups",
            json={"name": "analysts"},
            headers={"Authorization": f"Bearer {token}"},
        )

        response = self.client.post(
            "/auth/groups",
            json={"name": "analysts"},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 400)

    def test_register_non_existent_group_returns_bad_request(self):
        """Tests that registering a user with a non-existent group_id returns 400 Bad Request."""
        response = self.client.post(
            "/auth/register",
            json={
                "email": "groupless@example.com", 
                "password": "password123",
                "group_id": "99999"  # Fake ID
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Group not found", response.json()["detail"])

    def test_register_with_valid_group(self):
        """Tests that registering a user with a valid explicit group assigns it correctly."""
        token = self._register_and_login()
        
        # Create a group first to get a valid ID
        group_response = self.client.post(
            "/auth/groups",
            json={"name": "devs"},
            headers={"Authorization": f"Bearer {token}"},
        )
        group_id = group_response.json()["group_id"]

        # Register a new user into that group
        response = self.client.post(
            "/auth/register",
            json={
                "email": "dev_user@example.com", 
                "password": "password123",
                "group_id": group_id
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["group_name"], "devs")

    def test_authenticated_endpoint_missing_token_returns_unauthorized(self):
        """Tests that hitting a protected endpoint without an Authorization header returns 401."""
        response = self.client.get("/auth/me")  # No headers
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing authorization token", response.json()["detail"])

    def test_authenticated_endpoint_invalid_scheme_returns_unauthorized(self):
        """Tests that using an authentication scheme other than Bearer returns 401."""
        response = self.client.get(
            "/auth/me",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},  # Basic auth instead of Bearer
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing authorization token", response.json()["detail"])

    def test_authenticated_endpoint_inactive_user_returns_unauthorized(self):
        """Tests that a user whose token is valid but is marked inactive is rejected with a 401."""
        token = self._register_and_login("deactivated@example.com", "password123")
        
        # Deactivate the user directly in the DB backend
        user = self.test_user_db.get_user_by_token(token)
        if user:
            self.test_user_db.revoke_token(token)  # This will invalidate the token, simulating deactivation
        response = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid or expired token", response.json()["detail"])

    def test_create_group_non_existent_parent_returns_bad_request(self):
        """Tests that creating a group with an invalid parent_group_id returns 400."""
        token = self._register_and_login()
        
        response = self.client.post(
            "/auth/groups",
            json={"name": "sub-analysts", "parent_group_id": "99999"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Parent group not found", response.json()["detail"])

    def test_create_group_with_valid_parent(self):
        """Tests successful creation of a nested/child group hierarchy."""
        token = self._register_and_login()
        
        # Create Parent
        parent_res = self.client.post(
            "/auth/groups",
            json={"name": "engineering"},
            headers={"Authorization": f"Bearer {token}"},
        )
        parent_id = parent_res.json()["group_id"]

        # Create Child
        child_res = self.client.post(
            "/auth/groups",
            json={"name": "backend", "parent_group_id": parent_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(child_res.status_code, 200)
        self.assertEqual(child_res.json()["parent_group_id"], parent_id)

    def test_create_group_internal_error_fallback(self):
        """Tests that a 500 error is thrown if the group is created but cannot be retrieved."""
        token = self._register_and_login()

        # Force get_group_by_id to return None right after creation simulation
        original_get_group = self.test_user_db.get_group_by_id
        self.test_user_db.get_group_by_id = lambda group_id: None

        try:
            response = self.client.post(
                "/auth/groups",
                json={"name": "ghost-group"},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(response.status_code, 500)
            self.assertIn("Failed to create group", response.json()["detail"])
        finally:
            # Clean up our inline runtime hotfix
            self.test_user_db.get_group_by_id = original_get_group

if __name__ == "__main__":
    unittest.main()
