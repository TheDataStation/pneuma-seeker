# src/pneuma_seeker/services/db/users/manager.py
import os
import secrets
from datetime import UTC, datetime, timedelta
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from logging import Logger
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from pneuma_seeker.services.db.users.models import GroupPermissionRecord, GroupRecord, UserRecord
from pneuma_seeker.shared.config import Config


class UserDB:
    """Manages users, groups, and auth tokens stored in DuckDB."""

    def __init__(
        self,
        config: Config,
        logger: Logger,
        users_db_path: str | None = None,
    ) -> None:
        self.config = config
        self.logger = logger

        if users_db_path:
            self.users_db_path = Path(users_db_path)
        else:
            self.users_db_path = Path(__file__).resolve().parent / "users.db"

        os.makedirs(self.users_db_path.parent, exist_ok=True)
    
    def init_db(self) -> None:
        """Initializes schema and default settings. Call this ONLY once on app startup."""
        self._init_schema()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Returns a new connection to the DuckDB database."""
        return duckdb.connect(self.users_db_path.as_posix(), read_only=False)

    def _init_schema(self) -> None:
        """Initializes the database schema if it doesn't already exist, and ensures the default group is present."""
        con = self._get_connection()
        try:
            con.begin()
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS groups (
                    group_id        VARCHAR PRIMARY KEY,
                    name            VARCHAR UNIQUE,
                    parent_group_id VARCHAR,
                    FOREIGN KEY (parent_group_id) REFERENCES groups(group_id),
                    created_at      TIMESTAMP DEFAULT now()
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id            VARCHAR PRIMARY KEY,
                    email              VARCHAR UNIQUE,
                    username           VARCHAR,
                    password_hash      VARCHAR,
                    password_salt      VARCHAR,
                    password_iterations INTEGER,
                    group_id           VARCHAR,
                    is_active          BOOLEAN DEFAULT TRUE,
                    created_at         TIMESTAMP DEFAULT now(),
                    FOREIGN KEY (group_id) REFERENCES groups(group_id)
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS group_permissions (
                    group_id          VARCHAR,
                    permission_key    VARCHAR,
                    permission_value  VARCHAR,
                    created_at        TIMESTAMP DEFAULT now(),
                    updated_at        TIMESTAMP DEFAULT now(),
                    PRIMARY KEY (group_id, permission_key),
                    FOREIGN KEY (group_id) REFERENCES groups(group_id)
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token       VARCHAR PRIMARY KEY,
                    user_id     VARCHAR,
                    created_at  TIMESTAMP DEFAULT now(),
                    expires_at  TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

        self._ensure_default_group()

    def _ensure_default_group(self) -> str:
        """Ensures the default group exists and returns its ID."""
        self.__log("Ensuring default group exists")
        existing = self.get_group_by_name("default")
        if existing:
            return existing.group_id
        return self.create_group("default")

    def _hash_password(self, password: str) -> tuple[str, str, int]:
        """Hashes the password using PBKDF2 with a random salt and returns the hash, salt, and iteration count."""
        self.__log("Hashing password")
        iterations = max(100_000, self.config.AUTH_PASSWORD_HASH_ITERATIONS)
        salt = secrets.token_bytes(16)
        derived = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return derived.hex(), salt.hex(), iterations

    def _verify_password(
        self, password: str, hash_hex: str, salt_hex: str, iterations: int
    ) -> bool:
        """Verifies the password by hashing it with the provided salt and iterations, and comparing it to the stored hash."""
        self.__log(f"Verifying password")
        derived = pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            iterations,
        ).hex()
        return compare_digest(derived, hash_hex)

    def _normalize_email(self, email: str) -> str:
        """Normalizes the email by stripping whitespace and converting to lowercase."""
        self.__log(f"Normalizing email: {email}")
        return email.strip().lower()

    def create_group(self, name: str, parent_group_id: str | None = None) -> str:
        """Creates a new group, ensuring the parent group (if provided) exists, and returns the created group's ID. The group name must be unique (case-insensitive), and cannot be empty or whitespace."""
        self.__log(f"Creating group with name: {name}")
        name = name.strip()
        if not name:
            raise ValueError("Group name is required")
        if parent_group_id and not self.get_group_by_id(parent_group_id):
            raise ValueError("Parent group not found")

        group_id = str(uuid4())
        con = self._get_connection()
        try:
            con.execute(
                """
                INSERT INTO groups (group_id, name, parent_group_id)
                VALUES (?, ?, ?);
                """,
                (group_id, name, parent_group_id),
            )
            return group_id
        except Exception as e:
            if "UNIQUE" in str(e).upper():
                raise ValueError("Group already exists") from e
            raise
        finally:
            con.close()

    def get_group_by_name(self, name: str) -> GroupRecord | None:
        """Returns the group's record by name, or None if not found."""
        self.__log(f"Retrieving group by name: {name}")
        con = self._get_connection()
        try:
            row = con.execute(
                """
                SELECT group_id, name, parent_group_id
                FROM groups
                WHERE name = ?
                """,
                (name,),
            ).fetchone()
            if not row:
                return None
            return GroupRecord(group_id=row[0], name=row[1], parent_group_id=row[2])
        finally:
            con.close()

    def get_group_by_id(self, group_id: str) -> GroupRecord | None:
        """Returns the group's record by ID, or None if not found."""
        self.__log(f"Retrieving group by ID: {group_id}")
        con = self._get_connection()
        try:
            row = con.execute(
                """
                SELECT group_id, name, parent_group_id
                FROM groups
                WHERE group_id = ?
                """,
                (group_id,),
            ).fetchone()
            if not row:
                return None
            return GroupRecord(group_id=row[0], name=row[1], parent_group_id=row[2])
        finally:
            con.close()

    def list_groups(self) -> list[GroupRecord]:
        """Returns a list of all groups."""
        self.__log("Listing all groups")
        con = self._get_connection()
        try:
            rows = con.execute(
                """
                SELECT group_id, name, parent_group_id
                FROM groups
                ORDER BY name
                """
            ).fetchall()
            return [
                GroupRecord(group_id=row[0], name=row[1], parent_group_id=row[2])
                for row in rows
            ]
        finally:
            con.close()

    def list_group_ancestors(self, group_id: str) -> list[GroupRecord]:
        """Returns the group lineage from root-most parent to the requested group."""
        self.__log(f"Listing group ancestors for group_id: {group_id}")
        lineage: list[GroupRecord] = []
        seen: set[str] = set()
        current = self.get_group_by_id(group_id)
        while current:
            if current.group_id in seen:
                raise ValueError("Group hierarchy contains a cycle")
            seen.add(current.group_id)
            lineage.append(current)
            current = (
                self.get_group_by_id(current.parent_group_id)
                if current.parent_group_id
                else None
            )
        return list(reversed(lineage))

    def set_group_permission(
        self, group_id: str, permission_key: str, permission_value: str
    ) -> GroupPermissionRecord:
        """Sets a permission for the given group, creating or updating the record as needed. The group_id must exist, and the permission_key must be a non-empty string."""
        self.__log(f"Setting permission for group_id: {group_id}")
        if not self.get_group_by_id(group_id):
            raise ValueError("Group not found")
        permission_key = permission_key.strip()
        if not permission_key:
            raise ValueError("Permission key is required")

        con = self._get_connection()
        try:
            con.execute(
                """
                INSERT INTO group_permissions (
                    group_id,
                    permission_key,
                    permission_value
                ) VALUES (?, ?, ?)
                ON CONFLICT (group_id, permission_key) DO UPDATE SET
                    permission_value = excluded.permission_value,
                    updated_at = now();
                """,
                (group_id, permission_key, permission_value),
            )
            return GroupPermissionRecord(
                group_id=group_id,
                permission_key=permission_key,
                permission_value=permission_value,
            )
        finally:
            con.close()

    def get_group_permissions(self, group_id: str) -> dict[str, str]:
        """Returns the permissions directly assigned to the given group as a dictionary of key-value pairs."""
        self.__log(f"Retrieving permissions for group_id: {group_id}")
        con = self._get_connection()
        try:
            rows = con.execute(
                """
                SELECT permission_key, permission_value
                FROM group_permissions
                WHERE group_id = ?
                ORDER BY permission_key
                """,
                (group_id,),
            ).fetchall()
            return {row[0]: row[1] for row in rows}
        finally:
            con.close()

    def get_effective_group_permissions(self, group_id: str) -> dict[str, str]:
        """
        Returns permissions inherited through the group hierarchy.

        Parent permissions are applied first; child permissions with the same key
        override their ancestors.
        """
        self.__log(f"Calculating effective permissions for group_id: {group_id}")
        effective: dict[str, str] = {}
        for group in self.list_group_ancestors(group_id):
            effective.update(self.get_group_permissions(group.group_id))
        return effective

    def create_user(
        self,
        email: str,
        password: str,
        username: str | None = None,
        group_id: str | None = None,
    ) -> UserRecord:
        """Creates a new user with the given email and password, optionally assigning them to a group. The email must be unique (case-insensitive), and the group_id must exist if provided. The password is hashed securely before storage."""
        self.__log(f"Creating user with email: {email}")
        email = self._normalize_email(email)
        username = username or email
        group_id = group_id or self._ensure_default_group()
        if not self.get_group_by_id(group_id):
            raise ValueError("Group not found")
        password_hash, password_salt, iterations = self._hash_password(password)
        user_id = str(uuid4())

        con = self._get_connection()
        try:
            con.execute(
                """
                INSERT INTO users (
                    user_id,
                    email,
                    username,
                    password_hash,
                    password_salt,
                    password_iterations,
                    group_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    user_id,
                    email,
                    username,
                    password_hash,
                    password_salt,
                    iterations,
                    group_id,
                ),
            )
        except Exception as e:
            if "UNIQUE" in str(e).upper():
                raise ValueError("User already exists") from e
            raise
        finally:
            con.close()

        return UserRecord(
            user_id=user_id,
            email=email,
            username=username,
            group_id=group_id,
            is_active=True,
        )

    def get_user_by_email(self, email: str) -> UserRecord | None:
        """Returns the user's record by email, or None if not found."""
        self.__log(f"Retrieving user with email: {email}")
        email = self._normalize_email(email)
        con = self._get_connection()
        try:
            row = con.execute(
                """
                SELECT user_id, email, username, group_id, is_active
                FROM users
                WHERE email = ?
                """,
                (email,),
            ).fetchone()
            if not row:
                return None
            return UserRecord(
                user_id=row[0],
                email=row[1],
                username=row[2],
                group_id=row[3],
                is_active=bool(row[4]),
            )
        finally:
            con.close()

    def get_user_with_password(self, email: str) -> dict[str, Any] | None:
        """Returns the user's record including password hash and salt, or None if not found."""
        self.__log(f"Retrieving user with email: {email}")
        email = self._normalize_email(email)
        con = self._get_connection()
        try:
            row = con.execute(
                """
                SELECT user_id, email, username, password_hash, password_salt, password_iterations, group_id, is_active
                FROM users
                WHERE email = ?
                """,
                (email,),
            ).fetchone()
            if not row:
                return None
            return {
                "user_id": row[0],
                "email": row[1],
                "username": row[2],
                "password_hash": row[3],
                "password_salt": row[4],
                "password_iterations": row[5],
                "group_id": row[6],
                "is_active": bool(row[7]),
            }
        finally:
            con.close()

    def verify_user(self, email: str, password: str) -> UserRecord | None:
        """Verifies the user's credentials and returns their record if valid, or None if invalid."""
        self.__log(f"Verifying user with email: {email}")
        record = self.get_user_with_password(email)
        if not record:
            return None
        if not record["is_active"]:
            return None
        if not self._verify_password(
            password,
            record["password_hash"],
            record["password_salt"],
            record["password_iterations"],
        ):
            return None
        return UserRecord(
            user_id=record["user_id"],
            email=record["email"],
            username=record["username"],
            group_id=record["group_id"],
            is_active=record["is_active"],
        )

    def create_session_token(self, user_id: str) -> dict[str, Any]:
        """Creates a new session token for the user, stores it in the database with an expiration time, and returns the token information."""
        self.__log(f"Creating session token for user_id: {user_id}")
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(
            seconds=self.config.AUTH_TOKEN_TTL_SECONDS
        )
        con = self._get_connection()
        try:
            con.execute(
                """
                INSERT INTO auth_tokens (token, user_id, expires_at)
                VALUES (?, ?, ?);
                """,
                (token, user_id, expires_at),
            )
        finally:
            con.close()
        return {
            "token": token,
            "expires_at": expires_at,
            "token_type": "bearer",
        }

    def get_user_by_token(self, token: str) -> UserRecord | None:
        """Returns the user associated with the given session token if it's valid and not expired, or None if the token is invalid or expired. Expired tokens are also removed from the database."""
        self.__log(f"Validating token: {token}")
        now = datetime.now(UTC)
        con = self._get_connection()
        try:
            con.execute(
                """
                DELETE FROM auth_tokens
                WHERE expires_at IS NOT NULL AND expires_at < ?;
                """,
                (now,),
            )
            row = con.execute(
                """
                SELECT u.user_id, u.email, u.username, u.group_id, u.is_active
                FROM auth_tokens t
                JOIN users u ON u.user_id = t.user_id
                WHERE t.token = ?
                """,
                (token,),
            ).fetchone()
            if not row:
                return None
            return UserRecord(
                user_id=row[0],
                email=row[1],
                username=row[2],
                group_id=row[3],
                is_active=bool(row[4]),
            )
        finally:
            con.close()

    def revoke_token(self, token: str) -> None:
        """Revokes the given session token by removing it from the database."""
        self.__log(f"Revoking token: {token}")
        con = self._get_connection()
        try:
            con.execute("DELETE FROM auth_tokens WHERE token = ?;", (token,))
        finally:
            con.close()

    def __log(self, message: str):
        """Helper logging method."""
        self.logger.info(f"[UserDB] {message}")
