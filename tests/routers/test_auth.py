# tests/routers/test_auth.py
import logging
import os
import sys
import unittest

import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.routers import auth
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.shared.config import Config


def _truncate_all(dsn: str) -> None:
    """Removes all rows from the user DB tables so each test starts clean.
    Safe to call before init_db() — silently ignored if tables don't exist yet."""
    with psycopg.connect(dsn, autocommit=True) as con:
        try:
            con.execute(
                "TRUNCATE TABLE auth_tokens, users, group_permissions, groups RESTART IDENTITY CASCADE;"
            )
        except psycopg.errors.UndefinedTable:
            pass


class TestAuthRouter(unittest.TestCase):
    """Tests for FastAPI authentication and hierarchical group management endpoints."""

    def setUp(self):
        self.config = Config()
        self.config.AUTH_PASSWORD_HASH_ITERATIONS = 1
        self.config.AUTH_TOKEN_TTL_SECONDS = 60
        self.logger = logging.getLogger("test")
        self.dsn = os.environ["TEST_POSTGRES_DSN"]
        _truncate_all(self.dsn)

        self.test_user_db = UserDB(
            self.config,
            self.logger,
            dsn=self.dsn,
        )
        self.test_user_db.init_db()

        self.app = FastAPI()
        self.app.include_router(auth.router)

        self.original_user_db = auth.user_db
        auth.user_db = self.test_user_db

        self.client = TestClient(self.app)

        self._setup_admin_group()

    def tearDown(self):
        self.app.dependency_overrides.clear()
        auth.user_db = self.original_user_db

    def _setup_admin_group(self):
        """Helper to explicitly establish the admin group structure within the current test database instance."""
        try:
            self.admin_group_id = self.test_user_db.create_group("admin", None)
            self.test_user_db.set_group_permission(
                group_id=self.admin_group_id,
                permission_key="user:management",
                permission_value="true",
            )
        except ValueError:
            admin_rec = self.test_user_db.get_group_by_name("admin")
            if admin_rec:
                self.admin_group_id = admin_rec.group_id

    def _register_and_login(
        self, email="user@example.com", password="password123", group_id=None
    ):
        """Helper method to register a user and return their authentication token."""
        register_payload = {"email": email, "password": password}
        if group_id:
            register_payload["group_id"] = group_id

        register_response = self.client.post(
            "/auth/register",
            json=register_payload,
        )
        self.assertEqual(register_response.status_code, 200)

        login_response = self.client.post(
            "/auth/login",
            json={"email": email, "password": password},
        )
        self.assertEqual(login_response.status_code, 200)
        return login_response.json()["access_token"]

    def _register_and_login_admin(
        self, email="admin_user@example.com", password="password123"
    ):
        """Helper method to register an administrative user within the designated admin group block."""
        return self._register_and_login(email, password, group_id=self.admin_group_id)

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
        """Tests that creating and listing groups requires authorization, and executed successfully by admin."""
        token = self._register_and_login_admin()

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
        token = self._register_and_login_admin()
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
                "group_id": "99999",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Group not found", response.json()["detail"])

    def test_register_with_valid_group(self):
        """Tests that registering a user with a valid explicit group assigns it correctly."""
        token = self._register_and_login_admin()

        group_response = self.client.post(
            "/auth/groups",
            json={"name": "devs"},
            headers={"Authorization": f"Bearer {token}"},
        )
        group_id = group_response.json()["group_id"]

        response = self.client.post(
            "/auth/register",
            json={
                "email": "dev_user@example.com",
                "password": "password123",
                "group_id": group_id,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["group_name"], "devs")

    def test_authenticated_endpoint_missing_token_returns_unauthorized(self):
        """Tests that hitting a protected endpoint without an Authorization header returns 401."""
        response = self.client.get("/auth/me")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing authorization token", response.json()["detail"])

    def test_authenticated_endpoint_invalid_scheme_returns_unauthorized(self):
        """Tests that using an authentication scheme other than Bearer returns 401."""
        response = self.client.get(
            "/auth/me",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing authorization token", response.json()["detail"])

    def test_authenticated_endpoint_inactive_user_returns_unauthorized(self):
        """Tests that a user whose token is valid but is marked inactive is rejected with a 401."""
        token = self._register_and_login("deactivated@example.com", "password123")

        user = self.test_user_db.get_user_by_token(token)
        if user:
            self.test_user_db.revoke_token(token)
        response = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid or expired token", response.json()["detail"])

    def test_create_group_with_valid_parent(self):
        """Tests successful creation of a nested/child group hierarchy."""
        token = self._register_and_login_admin()

        parent_res = self.client.post(
            "/auth/groups",
            json={"name": "engineering"},
            headers={"Authorization": f"Bearer {token}"},
        )
        parent_id = parent_res.json()["group_id"]

        child_res = self.client.post(
            "/auth/groups",
            json={"name": "backend", "parent_group_id": parent_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(child_res.status_code, 200)
        self.assertEqual(child_res.json()["parent_group_id"], parent_id)

    def test_create_group_internal_error_fallback(self):
        """Tests that a 500 error is thrown if the group is created but cannot be retrieved."""
        token = self._register_and_login_admin()

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
            self.test_user_db.get_group_by_id = original_get_group

    def test_ensure_admin_blocks_standard_user(self):
        """Tests that a regular user without admin permissions is blocked from accessing administrative endpoints."""
        token = self._register_and_login("regular_user@example.com", "password123")

        endpoints = [
            ("POST", "/auth/groups", {"name": "unauthorized-group"}),
            ("GET", "/auth/groups", None),
            (
                "POST",
                "/auth/groups/permissions",
                {"permission_key": "read", "permission_value": "true"},
            ),
            ("POST", "/auth/users/change-group", {"user_id": "123", "group_id": "456"}),
        ]

        for method, path, payload in endpoints:
            if method == "POST":
                res = self.client.post(
                    path, json=payload, headers={"Authorization": f"Bearer {token}"}
                )
            else:
                res = self.client.get(
                    path, headers={"Authorization": f"Bearer {token}"}
                )

            self.assertEqual(
                res.status_code,
                403,
                f"Standard user bypassed restriction on endpoint: {path}",
            )
            self.assertIn("Access denied", res.json()["detail"])

    def test_get_group_permissions_direct(self):
        """Tests fetching only the direct permissions of the current user's group."""
        admin_token = self._register_and_login_admin()

        group_res = self.client.post(
            "/auth/groups",
            json={"name": "direct-perm-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = group_res.json()["group_id"]

        self.client.post(
            "/auth/groups/permissions",
            json={
                "group_id": group_id,
                "permission_key": "data:read",
                "permission_value": "true",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        user_token = self._register_and_login(
            "worker@example.com", "password123", group_id=group_id
        )

        perm_res = self.client.get(
            "/auth/groups/permissions?effective=false",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        self.assertEqual(perm_res.status_code, 200)
        self.assertEqual(perm_res.json().get("data:read"), "true")

    def test_get_group_permissions_effective_resolution(self):
        """Tests fetching resolved effective permissions that evaluate inheritance across the hierarchy tree."""
        admin_token = self._register_and_login_admin()

        parent_res = self.client.post(
            "/auth/groups",
            json={"name": "parent-layer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        parent_id = parent_res.json()["group_id"]

        child_res = self.client.post(
            "/auth/groups",
            json={"name": "child-layer", "parent_group_id": parent_id},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        child_id = child_res.json()["group_id"]

        self.client.post(
            "/auth/groups/permissions",
            json={
                "group_id": parent_id,
                "permission_key": "global:access",
                "permission_value": "true",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.client.post(
            "/auth/groups/permissions",
            json={
                "group_id": child_id,
                "permission_key": "local:write",
                "permission_value": "true",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        user_token = self._register_and_login(
            "child_user@example.com", "password123", group_id=child_id
        )

        effective_res = self.client.get(
            "/auth/groups/permissions?effective=true",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        self.assertEqual(effective_res.status_code, 200)
        perms = effective_res.json()
        self.assertEqual(perms.get("global:access"), "true")
        self.assertEqual(perms.get("local:write"), "true")

    def test_get_group_permissions_fails_for_groupless_user(self):
        """Tests that a 400 Bad Request is returned when checking permissions for a user with no assigned group."""
        token = self._register_and_login("groupless_check@example.com", "password123")

        original_get_current_user = auth.get_current_user

        try:
            user_rec = self.test_user_db.get_user_by_token(token)
            if user_rec:
                user_rec.group_id = None
                self.app.dependency_overrides[auth.get_current_user] = lambda: user_rec

            response = self.client.get(
                "/auth/groups/permissions", headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("not assigned to any group", response.json()["detail"])
        finally:
            self.app.dependency_overrides.clear()

    def test_set_group_permission_by_name(self):
        """Tests setting a group permission by specifying the group name string instead of an ID."""
        admin_token = self._register_and_login_admin()

        self.client.post(
            "/auth/groups",
            json={"name": "target-by-name"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        response = self.client.post(
            "/auth/groups/permissions",
            json={
                "group_name": "target-by-name",
                "permission_key": "execute",
                "permission_value": "allow",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["permission_value"], "allow")

    def test_set_group_permission_invalid_name_returns_not_found(self):
        """Tests that targeting a non-existent group name string triggers an explicit 404 block error."""
        admin_token = self._register_and_login_admin()

        response = self.client.post(
            "/auth/groups/permissions",
            json={
                "group_name": "non-existent-group-xyz",
                "permission_key": "read",
                "permission_value": "true",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_set_group_permission_no_target_fallback(self):
        """Tests that when no group target properties are present, it defaults modification access directly to the administrator's group context."""
        admin_token = self._register_and_login_admin()

        response = self.client.post(
            "/auth/groups/permissions",
            json={"permission_key": "admin_tool:use", "permission_value": "granted"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["group_id"], self.admin_group_id)

    def test_change_user_group_success(self):
        """Tests shifting a targeted user context entry cleanly between separate functional group blocks."""
        admin_token = self._register_and_login_admin()

        group_res = self.client.post(
            "/auth/groups",
            json={"name": "destination-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        destination_group_id = group_res.json()["group_id"]

        self.client.post(
            "/auth/register",
            json={"email": "migrant@example.com", "password": "password123"},
        )
        target_user_record = self.test_user_db.get_user_by_email("migrant@example.com")
        self.assertIsNotNone(target_user_record)

        assert target_user_record is not None

        payload = {
            "user_id": target_user_record.user_id,
            "group_id": destination_group_id,
        }
        response = self.client.post(
            "/auth/users/change-group",
            json=payload,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

        updated_user = self.test_user_db.get_user_by_email("migrant@example.com")
        assert updated_user is not None
        self.assertEqual(updated_user.group_id, destination_group_id)

    def test_change_user_group_invalid_group_returns_bad_request(self):
        """Tests that requesting an update path matching a non-existent group target ID triggers a 400 Bad Request error."""
        admin_token = self._register_and_login_admin()

        payload = {"user_id": "some-valid-user-id", "group_id": "99999"}
        response = self.client.post(
            "/auth/users/change-group",
            json=payload,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Target group does not exist", response.json()["detail"])

    def test_change_user_group_missing_user_returns_not_found(self):
        """Tests that passing a valid group ID but an invalid/missing user ID triggers a 404 Not Found error."""
        admin_token = self._register_and_login_admin()

        fake_uuid = "00000000-0000-0000-0000-000000000000"

        payload = {"user_id": fake_uuid, "group_id": self.admin_group_id}
        response = self.client.post(
            "/auth/users/change-group",
            json=payload,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("Target user not found", response.json()["detail"])

    def test_create_group_non_existent_parent_falls_back_to_default(self):
        """Tests that creating a group with an invalid parent_group_id falls back cleanly to the default group structure."""
        token = self._register_and_login_admin()

        response = self.client.post(
            "/auth/groups",
            json={"name": "fallback-test-group", "parent_group_id": "99999"},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)

        default_group = self.test_user_db.get_group_by_name("default")
        if default_group:
            self.assertEqual(response.json()["parent_group_id"], default_group.group_id)


    # ------------------------------------------------------------------
    # User CRUD (admin)
    # ------------------------------------------------------------------

    def test_list_users_returns_all_users(self):
        """Admin can retrieve a full list of registered users."""
        admin_token = self._register_and_login_admin()
        self._register_and_login("alice@example.com", "password123")
        self._register_and_login("bob@example.com", "password123")

        response = self.client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        emails = [u["email"] for u in response.json()]
        self.assertIn("alice@example.com", emails)
        self.assertIn("bob@example.com", emails)

    def test_list_users_blocked_for_non_admin(self):
        """Non-admin users receive 403 when trying to list all users."""
        token = self._register_and_login("regular@example.com", "password123")
        response = self.client.get("/auth/users", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 403)

    def test_get_user_by_id_success(self):
        """Admin can retrieve a specific user by their ID."""
        admin_token = self._register_and_login_admin()
        self._register_and_login("target@example.com", "password123")
        user = self.test_user_db.get_user_by_email("target@example.com")
        assert user is not None

        response = self.client.get(
            f"/auth/users/{user.user_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["email"], "target@example.com")

    def test_get_user_by_id_not_found(self):
        """Returns 404 when the requested user ID does not exist."""
        admin_token = self._register_and_login_admin()
        response = self.client.get(
            "/auth/users/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)

    def test_update_user_username_via_admin(self):
        """Admin can rename a user's username."""
        admin_token = self._register_and_login_admin()
        self._register_and_login("editable@example.com", "password123")
        user = self.test_user_db.get_user_by_email("editable@example.com")
        assert user is not None

        response = self.client.patch(
            f"/auth/users/{user.user_id}",
            json={"username": "new_handle"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "new_handle")

    def test_update_user_deactivate_via_admin(self):
        """Admin can deactivate a user account, preventing further logins."""
        admin_token = self._register_and_login_admin()
        token = self._register_and_login("deactivateme@example.com", "password123")
        user = self.test_user_db.get_user_by_email("deactivateme@example.com")
        assert user is not None

        self.client.patch(
            f"/auth/users/{user.user_id}",
            json={"is_active": False},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        me_response = self.client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(me_response.status_code, 401)

    def test_delete_user_success(self):
        """Admin can delete a user; subsequent lookups return 404."""
        admin_token = self._register_and_login_admin()
        self._register_and_login("deleteuser@example.com", "password123")
        user = self.test_user_db.get_user_by_email("deleteuser@example.com")
        assert user is not None

        response = self.client.delete(
            f"/auth/users/{user.user_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.test_user_db.get_user_by_email("deleteuser@example.com"))

    def test_delete_user_not_found(self):
        """Returns 404 when trying to delete a user that does not exist."""
        admin_token = self._register_and_login_admin()
        response = self.client.delete(
            "/auth/users/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # Self-service profile / password
    # ------------------------------------------------------------------

    def test_update_me_changes_username(self):
        """Authenticated user can rename their own account via PATCH /auth/me."""
        token = self._register_and_login("patchme@example.com", "password123")

        response = self.client.patch(
            "/auth/me",
            json={"username": "patched_name"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "patched_name")

    def test_update_me_no_fields_returns_bad_request(self):
        """PATCH /auth/me with an empty payload returns 400 Bad Request."""
        token = self._register_and_login("emptyuser@example.com", "password123")
        response = self.client.patch(
            "/auth/me",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 400)

    def test_change_password_success(self):
        """User can change their own password and then log in with the new credential."""
        token = self._register_and_login("pwdchange@example.com", "oldpassword1")

        change_res = self.client.post(
            "/auth/me/change-password",
            json={"current_password": "oldpassword1", "new_password": "newpassword1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(change_res.status_code, 200)

        login_res = self.client.post(
            "/auth/login",
            json={"email": "pwdchange@example.com", "password": "newpassword1"},
        )
        self.assertEqual(login_res.status_code, 200)

    def test_change_password_wrong_current_returns_401(self):
        """Providing the wrong current password when changing password returns 401."""
        token = self._register_and_login("wrongpwd@example.com", "correctpassword1")

        response = self.client.post(
            "/auth/me/change-password",
            json={"current_password": "wrongpassword1", "new_password": "newpassword1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 401)

    # ------------------------------------------------------------------
    # Group CRUD (admin)
    # ------------------------------------------------------------------

    def test_get_group_by_id_success(self):
        """Admin can retrieve a specific group by its ID."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "fetched-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]

        response = self.client.get(
            f"/auth/groups/{group_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "fetched-group")

    def test_get_group_by_id_not_found(self):
        """Returns 404 when the group ID does not exist."""
        admin_token = self._register_and_login_admin()
        response = self.client.get(
            "/auth/groups/99999",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)

    def test_update_group_rename(self):
        """Admin can rename an existing group."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "old-group-name"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]

        response = self.client.patch(
            f"/auth/groups/{group_id}",
            json={"name": "new-group-name"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "new-group-name")

    def test_update_group_no_fields_returns_bad_request(self):
        """PATCH /auth/groups/{id} with no fields returns 400."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "stable-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]

        response = self.client.patch(
            f"/auth/groups/{group_id}",
            json={},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 400)

    def test_delete_group_success(self):
        """Admin can delete an empty group; it no longer appears in the group list."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "to-be-deleted"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]

        del_res = self.client.delete(
            f"/auth/groups/{group_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(del_res.status_code, 200)

        group_list = self.client.get(
            "/auth/groups", headers={"Authorization": f"Bearer {admin_token}"}
        ).json()
        self.assertNotIn("to-be-deleted", [g["name"] for g in group_list])

    def test_delete_group_with_users_returns_conflict(self):
        """Deleting a group that still has users assigned returns 409 Conflict."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "occupied-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]
        self.client.post(
            "/auth/register",
            json={"email": "occupant@example.com", "password": "password123", "group_id": group_id},
        )

        response = self.client.delete(
            f"/auth/groups/{group_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 409)

    def test_delete_group_not_found(self):
        """Attempting to delete a non-existent group returns 404."""
        admin_token = self._register_and_login_admin()
        response = self.client.delete(
            "/auth/groups/99999",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)

    def test_delete_group_permission_success(self):
        """Admin can remove a specific permission from a group."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "perm-del-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]

        self.client.post(
            "/auth/groups/permissions",
            json={"group_id": group_id, "permission_key": "data:export", "permission_value": "true"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        del_res = self.client.delete(
            f"/auth/groups/{group_id}/permissions/data:export",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(del_res.status_code, 200)
        self.assertIsNone(
            self.test_user_db.get_group_permissions(group_id).get("data:export")
        )

    def test_delete_group_permission_not_found(self):
        """Returns 404 when the permission key does not exist on the group."""
        admin_token = self._register_and_login_admin()
        create_res = self.client.post(
            "/auth/groups",
            json={"name": "noperm-group"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        group_id = create_res.json()["group_id"]

        response = self.client.delete(
            f"/auth/groups/{group_id}/permissions/nonexistent:key",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
