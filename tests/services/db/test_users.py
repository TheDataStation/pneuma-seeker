import logging
import os
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.shared.config import Config


class TestUserDBUsers(unittest.TestCase):
    """Tests for DuckDB-backed user authentication data."""

    def setUp(self):
        self.config = Config()
        self.config.AUTH_PASSWORD_HASH_ITERATIONS = 1
        self.config.AUTH_TOKEN_TTL_SECONDS = 60
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.users_db_path = Path(self.tmpdir) / "users.db"
        self.db = UserDB(self.config, self.logger, self.users_db_path.as_posix())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_creates_default_group(self):
        """The default 'default' group should be created on DB initialization."""
        group = self.db.get_group_by_name("default")

        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.name, "default")
        self.assertIsNone(group.parent_group_id)

    def test_create_user_defaults_group(self):
        """Creating a user without specifying a group should assign them to the default group."""
        user = self.db.create_user("USER@Example.COM", "password123")
        default_group = self.db.get_group_by_name("default")

        assert default_group is not None

        self.assertEqual(user.email, "user@example.com")
        self.assertEqual(user.username, "user@example.com")
        self.assertEqual(user.group_id, default_group.group_id)
        self.assertTrue(user.is_active)

    def test_create_user_uses_explicit_group_and_username(self):
        """Creating a user with an explicit group and username should assign them correctly."""
        group_id = self.db.create_group("analysts")

        user = self.db.create_user(
            "analyst@example.com",
            "password123",
            username="Analyst",
            group_id=group_id,
        )

        self.assertEqual(user.username, "Analyst")
        self.assertEqual(user.group_id, group_id)

    def test_create_user_rejects_unknown_group(self):
        """Creating a user with a non-existent group_id should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.db.create_user(
                "missing-group@example.com",
                "password123",
                group_id="missing-group-id",
            )

    def test_duplicate_user_raises_value_error_case_insensitive(self):
        """Creating a user with an email that already exists (case-insensitive) should raise a ValueError."""
        self.db.create_user("dupe@example.com", "password123")

        with self.assertRaises(ValueError):
            self.db.create_user("DUPE@example.com", "password123")

    def test_password_is_hashed_and_verifiable(self):
        """Passwords should be stored as hashes, and the verify_user method should validate them correctly."""
        user = self.db.create_user("hash@example.com", "password123")
        record = self.db.get_user_with_password("hash@example.com")

        assert record is not None

        self.assertNotEqual(record["password_hash"], "password123")
        self.assertIsNotNone(self.db.verify_user("hash@example.com", "password123"))
        self.assertIsNone(self.db.verify_user("hash@example.com", "wrong-password"))
        self.assertEqual(record["user_id"], user.user_id)

    def test_inactive_user_cannot_verify(self):
        """Users with is_active = FALSE should not be able to verify successfully."""
        user = self.db.create_user("inactive@example.com", "password123")
        con = self.db._get_connection()
        try:
            con.execute(
                "UPDATE users SET is_active = FALSE WHERE user_id = ?",
                (user.user_id,),
            )
        finally:
            con.close()

        self.assertIsNone(self.db.verify_user("inactive@example.com", "password123"))


class TestUserDBGroups(unittest.TestCase):
    """Tests for group hierarchy and inherited permissions."""

    def setUp(self):
        self.config = Config()
        self.config.AUTH_PASSWORD_HASH_ITERATIONS = 1
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = UserDB(
            self.config,
            self.logger,
            (Path(self.tmpdir) / "users.db").as_posix(),
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_group_with_parent(self):
        """Creating a group with a valid parent_group_id should set the parent-child relationship correctly."""
        parent_id = self.db.create_group("company")
        child_id = self.db.create_group("research", parent_group_id=parent_id)

        child = self.db.get_group_by_id(child_id)
        assert child is not None
        self.assertEqual(child.parent_group_id, parent_id)

    def test_create_group_rejects_missing_parent(self):
        """Creating a group with a parent_group_id that doesn't exist should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.db.create_group("orphan", parent_group_id="missing")

    def test_create_group_rejects_duplicate_name(self):
        """Creating a group with a name that already exists (case-insensitive) should raise a ValueError."""
        self.db.create_group("duplicate")

        with self.assertRaises(ValueError):
            self.db.create_group("duplicate")

    def test_list_group_ancestors_returns_root_to_child(self):
        """The list_group_ancestors method should return the correct lineage of groups from root to the specified group."""
        parent_id = self.db.create_group("company")
        child_id = self.db.create_group("research", parent_group_id=parent_id)
        grandchild_id = self.db.create_group("nlp", parent_group_id=child_id)

        lineage = self.db.list_group_ancestors(grandchild_id)

        self.assertEqual([group.name for group in lineage], ["company", "research", "nlp"])

    def test_effective_permissions_inherit_and_child_overrides(self):
        """Child groups should inherit permissions from their ancestors, but can override them with their own settings."""
        parent_id = self.db.create_group("company")
        child_id = self.db.create_group("research", parent_group_id=parent_id)

        self.db.set_group_permission(parent_id, "datasets.read", "true")
        self.db.set_group_permission(parent_id, "datasets.write", "false")
        self.db.set_group_permission(child_id, "datasets.write", "true")

        permissions = self.db.get_effective_group_permissions(child_id)

        self.assertEqual(
            permissions,
            {
                "datasets.read": "true",
                "datasets.write": "true",
            },
        )

    def test_set_permission_rejects_missing_group(self):
        """Setting a permission for a non-existent group should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.db.set_group_permission("missing", "datasets.read", "true")


class TestUserDBTokens(unittest.TestCase):
    """Tests for auth token lifecycle."""

    def setUp(self):
        self.config = Config()
        self.config.AUTH_PASSWORD_HASH_ITERATIONS = 1
        self.config.AUTH_TOKEN_TTL_SECONDS = 60
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = UserDB(
            self.config,
            self.logger,
            (Path(self.tmpdir) / "users.db").as_posix(),
        )
        self.user = self.db.create_user("token@example.com", "password123")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_token_and_get_user_by_token(self):
        """Creating a session token for a user and then retrieving the user by that token should work correctly."""
        token_payload = self.db.create_session_token(self.user.user_id)

        user = self.db.get_user_by_token(token_payload["token"])

        assert user is not None
        self.assertEqual(user.user_id, self.user.user_id)
        self.assertEqual(token_payload["token_type"], "bearer")

    def test_revoke_token(self):
        """Revoking a token should prevent it from being used to retrieve a user."""
        token = self.db.create_session_token(self.user.user_id)["token"]

        self.db.revoke_token(token)

        self.assertIsNone(self.db.get_user_by_token(token))

    def test_expired_token_is_removed_and_rejected(self):
        """Expired tokens should be removed from the database and rejected when attempting to retrieve a user."""
        token = self.db.create_session_token(self.user.user_id)["token"]
        con = self.db._get_connection()
        try:
            con.execute(
                "UPDATE auth_tokens SET expires_at = ? WHERE token = ?",
                (datetime.now(UTC) - timedelta(seconds=1), token),
            )
        finally:
            con.close()

        self.assertIsNone(self.db.get_user_by_token(token))

        con = self.db._get_connection()
        try:
            count = con.execute(
                "SELECT COUNT(*) FROM auth_tokens WHERE token = ?",
                (token,),
            ).fetchone()
            assert count is not None
            count = count[0]
        finally:
            con.close()
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
