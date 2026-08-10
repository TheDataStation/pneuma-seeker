from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, EmailStr

from pneuma_seeker.shared.schemas.language_model.message import LLMMessage


class EndpointTag(Enum):
    INDEXING = "indexing"
    CORE = "core"
    CHAT = "chat"
    AUTH = "auth"
    MEMORY = "memory"


class PermissionKey(Enum):
    ADMIN = "admin"  # This is a special permission that grants all access
    DATASET_ACCESS_PREFIX = "dataset:access"
    USER_MANAGEMENT = "user:management"
    INDEXING_MANAGEMENT = "indexing:management"
    MEMORY_MANAGEMENT = (
        "memory:management"  # Gates admin-only actions like metadata CSV upload
    )


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


class DatasetQueryRequest(BaseModel):
    """Body for POST /chat/dataset_query/{dataset_name} — a single read-only
    SELECT/WITH...SELECT against exactly one dataset's own DB file. See
    DatasetManager.query_sql for the safety checks (single statement,
    read-only connection, capped rows)."""

    sql: str = Field(min_length=1)
    params: list[Any] | None = None
    limit: int = 500


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
    is_active: bool


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


class UpdateUserRequest(BaseModel):
    username: str | None = None
    is_active: bool | None = None
    group_id: str | None = None


class UpdateMeRequest(BaseModel):
    username: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class UpdateGroupRequest(BaseModel):
    name: str | None = None
    parent_group_id: str | None = None


class MemoryEntryResponse(BaseModel):
    memory_id: str
    memory_type: str
    content: str
    source: str
    group_id: str | None = None
    user_id: str | None = None
    dataset_name: str | None = None
    key_text: str | None = None
    source_user_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MemoryEntryListResponse(BaseModel):
    items: list[MemoryEntryResponse]
    has_more: bool
    next_offset: int | None = None


class CreateMemoryEntryRequest(BaseModel):
    memory_type: str = Field(
        ...,
        description="tribal_knowledge | user_preference | table_metadata | column_metadata",
    )
    content: str = Field(min_length=1)
    dataset_name: str | None = None
    key_text: str | None = Field(
        None,
        description="Required for table_metadata (table name) / column_metadata (table.column)",
    )


class UpdateMemoryEntryRequest(BaseModel):
    content: str = Field(min_length=1)


class MemoryMetadataUploadResponse(BaseModel):
    dataset_name: str
    memory_type: str
    rows_stored: int
