from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    TRIBAL_KNOWLEDGE = "tribal_knowledge"
    USER_PREFERENCE = "user_preference"
    AGENT_LEARNING = "agent_learning"
    TABLE_METADATA = "table_metadata"
    COLUMN_METADATA = "column_metadata"


class MemorySource(str, Enum):
    USER = "user"
    LLM_EXTRACTION = "llm_extraction"
    AGENT_LEARNING = "agent_learning"
    CSV_UPLOAD = "csv_upload"


@dataclass
class MemoryEntryRecord:
    memory_id: str
    memory_type: str
    content: str
    source: str
    group_id: str | None = None
    user_id: str | None = None
    dataset_name: str | None = None
    key_text: str | None = None
    extra: dict[str, Any] | None = None
    source_user_id: str | None = None
    chat_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
