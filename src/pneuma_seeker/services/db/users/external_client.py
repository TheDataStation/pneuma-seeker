# src/pneuma_seeker/services/db/users/external_client.py — ExternalUserDB,
# used in place of UserDB when config.AUTH_BACKEND == "external" (see
# factory.py and the migration plan). Proxies every call to Agentic
# Catalog's own identity/router.py endpoints instead of touching Postgres
# directly. UserDB itself is never modified — this is a separate class with
# (almost exactly) the same public method surface, swapped in wholesale via
# the factory rather than by branching inside UserDB.
#
# Trust model: methods that carry their own proof (a real user's own bearer
# token, or an email+password pair) use exactly that as the call's
# authentication — get_user_by_token/revoke_token forward the token itself;
# verify_user/create_user need no token at all (the password/registration
# itself is the proof). Every other method (group/user CRUD and lookups)
# has no per-call caller identity of its own to forward — Pneuma-Seeker has
# always already authorized the original human caller (via get_current_user/
# ensure_admin) before ever reaching one of these, so they're all sent with
# the static AGENTIC_CATALOG_SERVICE_TOKEN instead, which Agentic Catalog
# treats as unconditionally trusted for exactly these admin-gated routes —
# see identity/router.py's own module docstring on the other side.
#
# login(email, password) is NOT part of UserDB's original surface — it's
# additive, used only by routers/auth.py's login handler's explicit
# backend branch (see that file): local mode composes verify_user() +
# create_session_token() as two separate primitives, but there is
# deliberately no "mint a token for an arbitrary user_id, no password"
# endpoint exposed externally (that really would be dangerous over a
# network boundary), so external mode needs the one combined call instead.
from datetime import datetime
from typing import Any

import httpx

from pneuma_seeker.services.db.users.models import GroupPermissionRecord, GroupRecord, UserRecord
from pneuma_seeker.shared.config import Config

_TIMEOUT = 10.0


class ExternalUserDB:
    def __init__(self, config: Config, logger) -> None:
        self.config = config
        self.logger = logger
        self.base_url = config.AUTH_BACKEND_EXTERNAL_URL.rstrip("/")
        self._service_headers = {"Authorization": f"Bearer {config.AGENTIC_CATALOG_SERVICE_TOKEN}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict | None = None,
        json: dict | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        with httpx.Client(timeout=_TIMEOUT) as client:
            return client.request(method, f"{self.base_url}{path}", headers=headers, json=json, params=params)

    def _user_from_response(self, body: dict) -> UserRecord:
        return UserRecord(
            user_id=body["user_id"],
            email=body["email"],
            username=body["username"],
            group_id=body["group_id"],
            is_active=body["is_active"],
        )

    def _group_from_response(self, body: dict) -> GroupRecord:
        return GroupRecord(group_id=body["group_id"], name=body["name"], parent_group_id=body["parent_group_id"])

    # -- carries its own proof (a real bearer token, or credentials) --

    def get_user_by_token(self, token: str) -> UserRecord | None:
        resp = self._request("GET", "/me", headers={"Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
            return None
        return self._user_from_response(resp.json())

    def revoke_token(self, token: str) -> None:
        self._request("POST", "/logout", headers={"Authorization": f"Bearer {token}"})

    def verify_user(self, email: str, password: str) -> UserRecord | None:
        resp = self._request("POST", "/verify-credentials", json={"email": email, "password": password})
        if resp.status_code != 200:
            return None
        return self._user_from_response(resp.json())

    def login(self, email: str, password: str) -> dict[str, Any] | None:
        """Not part of UserDB's surface — see module docstring."""
        resp = self._request("POST", "/login", json={"email": email, "password": password})
        if resp.status_code != 200:
            return None
        body = resp.json()
        return {
            "token": body["access_token"],
            "token_type": body["token_type"],
            "expires_at": datetime.fromisoformat(body["expires_at"]),
        }

    def create_user(self, email: str, password: str, username: str | None = None, group_id: str | None = None) -> UserRecord:
        resp = self._request(
            "POST", "/register", json={"email": email, "password": password, "group_id": group_id}
        )
        if resp.status_code == 409:
            raise ValueError("User already exists")
        if resp.status_code != 200:
            raise ValueError(resp.json().get("detail", "Failed to create user"))
        return self._user_from_response(resp.json())

    # -- no caller-specific proof of their own: uses the service token --

    def get_group_by_id(self, group_id: str) -> GroupRecord | None:
        resp = self._request("GET", f"/groups/{group_id}", headers=self._service_headers)
        if resp.status_code != 200:
            return None
        return self._group_from_response(resp.json())

    def get_group_by_name(self, name: str) -> GroupRecord | None:
        for group in self.list_groups():
            if group.name == name:
                return group
        return None

    def list_groups(self) -> list[GroupRecord]:
        resp = self._request("GET", "/groups", headers=self._service_headers)
        resp.raise_for_status()
        return [self._group_from_response(g) for g in resp.json()]

    def create_group(self, name: str, parent_group_id: str | None = None) -> str:
        resp = self._request(
            "POST", "/groups", headers=self._service_headers, json={"name": name, "parent_group_id": parent_group_id}
        )
        if resp.status_code != 200:
            raise ValueError(resp.json().get("detail", "Failed to create group"))
        return resp.json()["group_id"]

    def update_group(self, group_id: str, **kwargs: Any) -> bool:
        """kwargs: name/parent_group_id — only include what should actually
        change, mirroring UserDB.update_group's _UNSET-omission semantics
        (identity/router.py's UpdateGroupRequest only updates fields
        actually present in the JSON body)."""
        resp = self._request("PATCH", f"/groups/{group_id}", headers=self._service_headers, json=kwargs)
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise ValueError(resp.json().get("detail", "Failed to update group"))
        return True

    def delete_group(self, group_id: str) -> bool:
        resp = self._request("DELETE", f"/groups/{group_id}", headers=self._service_headers)
        if resp.status_code == 409:
            raise ValueError(resp.json().get("detail", "Cannot delete group"))
        return resp.status_code == 200

    def delete_group_permission(self, group_id: str, permission_key: str) -> bool:
        resp = self._request(
            "DELETE", f"/groups/{group_id}/permissions/{permission_key}", headers=self._service_headers
        )
        return resp.status_code == 200

    def list_group_ancestors(self, group_id: str) -> list[GroupRecord]:
        """Same root-first walk IdentityDB.list_group_ancestors does
        locally, just built from repeated get_group_by_id proxy calls
        instead of local rows — no dedicated endpoint needed for this."""
        lineage: list[GroupRecord] = []
        seen: set[str] = set()
        current = self.get_group_by_id(group_id)
        while current:
            if current.group_id in seen:
                raise ValueError("Group hierarchy contains a cycle")
            seen.add(current.group_id)
            lineage.append(current)
            current = self.get_group_by_id(current.parent_group_id) if current.parent_group_id else None
        return list(reversed(lineage))

    def set_group_permission(self, group_id: str, permission_key: str, permission_value: str) -> GroupPermissionRecord:
        resp = self._request(
            "POST",
            "/groups/permissions",
            headers=self._service_headers,
            json={"group_id": group_id, "permission_key": permission_key, "permission_value": permission_value},
        )
        if resp.status_code != 200:
            raise ValueError(resp.json().get("detail", "Failed to set group permission"))
        return GroupPermissionRecord(group_id=group_id, permission_key=permission_key, permission_value=permission_value)

    def get_group_permissions(self, group_id: str) -> dict[str, str]:
        resp = self._request("GET", f"/groups/{group_id}/permissions", headers=self._service_headers)
        resp.raise_for_status()
        return resp.json()

    def get_effective_group_permissions(self, group_id: str) -> dict[str, str]:
        resp = self._request(
            "GET", f"/groups/{group_id}/permissions", headers=self._service_headers, params={"effective": "true"}
        )
        resp.raise_for_status()
        return resp.json()

    def list_users(self) -> list[UserRecord]:
        resp = self._request("GET", "/users", headers=self._service_headers)
        resp.raise_for_status()
        return [self._user_from_response(u) for u in resp.json()]

    def get_user_by_id(self, user_id: str) -> UserRecord | None:
        resp = self._request("GET", f"/users/{user_id}", headers=self._service_headers)
        if resp.status_code != 200:
            return None
        return self._user_from_response(resp.json())

    def get_user_by_email(self, email: str) -> UserRecord | None:
        for user in self.list_users():
            if user.email == email:
                return user
        return None

    def update_user(
        self,
        user_id: str,
        username: str | None = None,
        group_id: str | None = None,
        is_active: bool | None = None,
    ) -> bool:
        body: dict[str, Any] = {}
        if username is not None:
            body["username"] = username
        if group_id is not None:
            body["group_id"] = group_id
        if is_active is not None:
            body["is_active"] = is_active
        if not body:
            return False
        resp = self._request("PATCH", f"/users/{user_id}", headers=self._service_headers, json=body)
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise ValueError(resp.json().get("detail", "Failed to update user"))
        return True

    def update_password(self, user_id: str, new_password: str) -> bool:
        resp = self._request(
            "PATCH", f"/users/{user_id}", headers=self._service_headers, json={"new_password": new_password}
        )
        return resp.status_code == 200

    def delete_user(self, user_id: str) -> bool:
        resp = self._request("DELETE", f"/users/{user_id}", headers=self._service_headers)
        return resp.status_code == 200
