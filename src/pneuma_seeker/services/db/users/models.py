from dataclasses import dataclass


@dataclass
class UserRecord:
    user_id: str
    email: str
    username: str
    group_id: str | None
    is_active: bool


@dataclass
class GroupRecord:
    group_id: str
    name: str
    parent_group_id: str | None


@dataclass
class GroupPermissionRecord:
    group_id: str
    permission_key: str
    permission_value: str
