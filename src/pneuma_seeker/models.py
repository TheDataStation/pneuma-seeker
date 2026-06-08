from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, EmailStr


class EndpointTag(Enum):
    INDEXING = "indexing"
    CORE = "core"
    CHAT = "chat"
    AUTH = "auth"


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


