from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, EmailStr

from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class EndpointTag(Enum):
    INDEXING = "indexing"
    CORE = "core"
    CHAT = "chat"
    AUTH = "auth"


class PermissionKey(Enum):
    ADMIN = "admin"  # This is a special permission that grants all access
    DATASET_ACCESS_PREFIX = "dataset:access"
    USER_MANAGEMENT = "user:management"
    INDEXING_MANAGEMENT = "indexing:management"


class ChatHistoryResponse(BaseModel):
    user_id: str
    chat_id: str
    messages: list[LLMMessage]
    dataset_name: str | None = None


class IndexDatasetRequest(BaseModel):
    dataset_name: str = Field(min_length=1)
    connector_config: dict[str, Any]
    overwrite: bool
    schema_summaries: list[dict[str, Any]] | None = None


class IndexDatasetResponse(BaseModel):
    run_id: str
    dataset_name: str
    latest_metadata: dict[str, Any] | None


class DatasetMetadataResponse(BaseModel):
    dataset_name: str
    latest_metadata: dict[str, Any]


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    group_id: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    user_id: str
    email: EmailStr
    username: str
    group_id: str | None
    group_name: str | None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str


class GroupCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    parent_group_id: str | None = None


class GroupResponse(BaseModel):
    group_id: str
    name: str
    parent_group_id: str | None


class SetPermissionRequest(BaseModel):
    permission_key: str = Field(
        ...,
        description="The permission key/string (e.g., 'user:management' or 'dataset:access:archeology')",
    )
    permission_value: str = Field(
        ..., description="The value assigned (e.g., 'true', 'read')"
    )
    group_id: str | None = Field(
        None, description="Target group ID. If omitted, defaults to the caller's group."
    )
    group_name: str | None = Field(
        None, description="Target group name (used if group_id is omitted)."
    )


class UpdateUserGroupRequest(BaseModel):
    user_id: str = Field(..., description="The ID of the user being modified")
    group_id: str = Field(..., description="The new group ID to assign to this user")
