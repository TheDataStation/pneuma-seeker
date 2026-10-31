# src/pneuma_seeker/routers/memory.py
from pandas import read_csv
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from pneuma_seeker.models import (
    CreateMemoryEntryRequest,
    EndpointTag,
    MemoryEntryListResponse,
    MemoryEntryResponse,
    MemoryMetadataUploadResponse,
    PermissionKey,
    UpdateMemoryEntryRequest,
)
from pneuma_seeker.routers.auth import get_current_user, get_current_user_permissions
from pneuma_seeker.services.db.memory.manager import MemoryManager
from pneuma_seeker.services.db.memory.models import MemoryEntryRecord, MemoryType
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.services.db.users.models import UserRecord
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

router = APIRouter(
    prefix="/memory",
    tags=[EndpointTag.MEMORY],
)

config = Config("../../../.env")
logger = setup_logger("Memory Router")
user_db = UserDB(config, logger)

# A Postgres hiccup at import time must not take down the entire app (auth,
# chat, indexing routers are all imported in the same statement in main.py) —
# so this mirrors PneumaDB's graceful-degradation of the memory layer.
try:
    memory_manager: MemoryManager | None = MemoryManager(config, logger)
except Exception as e:
    logger.warning(f"[Memory Router] Postgres memory store unavailable: {e}")
    memory_manager = None


def get_memory_manager() -> MemoryManager:
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory layer is unavailable.")
    return memory_manager

_DATASET_SCOPED_TYPES = {
    MemoryType.AGENT_LEARNING.value,
    MemoryType.TABLE_METADATA.value,
    MemoryType.COLUMN_METADATA.value,
}
_MANUALLY_CREATABLE_TYPES = {
    MemoryType.TRIBAL_KNOWLEDGE.value,
    MemoryType.USER_PREFERENCE.value,
    MemoryType.TABLE_METADATA.value,
    MemoryType.COLUMN_METADATA.value,
}


def _is_admin(current_user: UserRecord) -> bool:
    if not current_user.group_id:
        return False
    perms = user_db.get_effective_group_permissions(current_user.group_id)
    admin_value = perms.get(PermissionKey.ADMIN.value, False)
    is_admin = admin_value.lower() == "true" if isinstance(admin_value, str) else bool(admin_value)
    if is_admin:
        return True
    memory_mgmt = perms.get(PermissionKey.MEMORY_MANAGEMENT.value, "false")
    return str(memory_mgmt).lower() == "true"


def _ancestor_group_ids(group_id: str | None) -> list[str]:
    if not group_id:
        return []
    return [g.group_id for g in user_db.list_group_ancestors(group_id)]


def _to_response(record: MemoryEntryRecord) -> MemoryEntryResponse:
    return MemoryEntryResponse(
        memory_id=record.memory_id,
        memory_type=record.memory_type,
        content=record.content,
        source=record.source,
        group_id=record.group_id,
        user_id=record.user_id,
        dataset_name=record.dataset_name,
        key_text=record.key_text,
        source_user_id=record.source_user_id,
        created_at=record.created_at.isoformat() if record.created_at else None,
        updated_at=record.updated_at.isoformat() if record.updated_at else None,
    )


@router.get("/entries", response_model=MemoryEntryListResponse)
def list_entries(
    memory_type: str,
    dataset_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
    current_user: UserRecord = Depends(get_current_user),
    mm: MemoryManager = Depends(get_memory_manager),
):
    """Lists memory entries of a given type visible to the current user."""
    if memory_type not in {t.value for t in MemoryType}:
        raise HTTPException(status_code=400, detail=f"Unknown memory_type '{memory_type}'")

    if memory_type in _DATASET_SCOPED_TYPES:
        if not dataset_name:
            raise HTTPException(
                status_code=400, detail="dataset_name is required for this memory_type"
            )
        result = mm.list_entries(
            memory_type, dataset_name=dataset_name, limit=limit, offset=offset
        )
    elif memory_type == MemoryType.USER_PREFERENCE.value:
        result = mm.list_entries(
            memory_type, user_id=current_user.user_id, limit=limit, offset=offset
        )
    else:  # tribal_knowledge — visible to the user's group and its ancestors
        group_ids = _ancestor_group_ids(current_user.group_id)
        result = mm.list_entries(
            memory_type, group_ids=group_ids, limit=limit, offset=offset
        )

    return MemoryEntryListResponse(
        items=[_to_response(r) for r in result["items"]],
        has_more=result["has_more"],
        next_offset=result["next_offset"],
    )


@router.post("/entries", response_model=MemoryEntryResponse)
def create_entry(
    payload: CreateMemoryEntryRequest,
    current_user: UserRecord = Depends(get_current_user),
    mm: MemoryManager = Depends(get_memory_manager),
):
    """Manually creates a memory entry. agent_learning entries are system-generated only."""
    if payload.memory_type not in _MANUALLY_CREATABLE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"memory_type '{payload.memory_type}' cannot be created manually",
        )

    if payload.memory_type in _DATASET_SCOPED_TYPES:
        if not payload.dataset_name or not payload.key_text:
            raise HTTPException(
                status_code=400,
                detail="dataset_name and key_text are required for this memory_type",
            )
        record = mm.create_entry(
            payload.memory_type,
            payload.content,
            "user",
            current_user.user_id,
            dataset_name=payload.dataset_name,
            key_text=payload.key_text,
        )
    elif payload.memory_type == MemoryType.USER_PREFERENCE.value:
        record = mm.create_entry(
            payload.memory_type,
            payload.content,
            "user",
            current_user.user_id,
            user_id=current_user.user_id,
            dataset_name=payload.dataset_name,
        )
    else:  # tribal_knowledge
        if not current_user.group_id:
            raise HTTPException(status_code=400, detail="User belongs to no group")
        record = mm.create_entry(
            payload.memory_type,
            payload.content,
            "user",
            current_user.user_id,
            group_id=current_user.group_id,
            dataset_name=payload.dataset_name,
        )

    return _to_response(record)


@router.patch("/entries/{memory_id}", response_model=MemoryEntryResponse)
def update_entry(
    memory_id: str,
    payload: UpdateMemoryEntryRequest,
    current_user: UserRecord = Depends(get_current_user),
    mm: MemoryManager = Depends(get_memory_manager),
):
    """Edits a memory entry's content."""
    entry = mm.get_entry(memory_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    _ensure_can_modify(entry, current_user)

    mm.update_entry(memory_id, content=payload.content)
    updated = mm.get_entry(memory_id)
    assert updated is not None
    return _to_response(updated)


@router.delete("/entries/{memory_id}")
def delete_entry(
    memory_id: str,
    current_user: UserRecord = Depends(get_current_user),
    mm: MemoryManager = Depends(get_memory_manager),
):
    """Deletes a memory entry."""
    entry = mm.get_entry(memory_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    _ensure_can_modify(entry, current_user)

    mm.delete_entry(memory_id)
    return {"detail": "Memory entry deleted successfully."}


def _ensure_can_modify(entry: MemoryEntryRecord, current_user: UserRecord) -> None:
    """Everyone in the relevant group/dataset can modify — this only rejects cross-user/group edits."""
    if entry.memory_type == MemoryType.USER_PREFERENCE.value:
        if entry.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not your preference entry")
    elif entry.memory_type == MemoryType.TRIBAL_KNOWLEDGE.value:
        if entry.group_id != current_user.group_id:
            raise HTTPException(status_code=403, detail="Not your group's entry")


@router.post("/metadata/upload", response_model=MemoryMetadataUploadResponse)
def upload_metadata_csv(
    dataset_name: str = Form(...),
    memory_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: UserRecord = Depends(get_current_user),
    mm: MemoryManager = Depends(get_memory_manager),
):
    """
    Admin-only: uploads a CSV of table or column descriptions for a dataset,
    replacing any previously uploaded rows of that memory_type for that dataset.

    - table_metadata expects columns: table_name, description
    - column_metadata expects columns: table_name, column_name, description
    """
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")

    if memory_type not in (MemoryType.TABLE_METADATA.value, MemoryType.COLUMN_METADATA.value):
        raise HTTPException(
            status_code=400,
            detail="memory_type must be table_metadata or column_metadata",
        )

    try:
        df = read_csv(BytesIO(file.file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    rows: list[dict[str, str]] = []
    if memory_type == MemoryType.TABLE_METADATA.value:
        if "table_name" not in df.columns or "description" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must have 'table_name' and 'description' columns",
            )
        for _, row in df.iterrows():
            rows.append({"key_text": str(row["table_name"]), "content": str(row["description"])})
    else:
        required = {"table_name", "column_name", "description"}
        if not required.issubset(set(df.columns)):
            raise HTTPException(
                status_code=400,
                detail="CSV must have 'table_name', 'column_name', and 'description' columns",
            )
        for _, row in df.iterrows():
            key_text = f"{row['table_name']}.{row['column_name']}"
            rows.append({"key_text": key_text, "content": str(row["description"])})

    rows_stored = mm.replace_dataset_rows(
        dataset_name, memory_type, rows, uploaded_by=current_user.user_id
    )
    return MemoryMetadataUploadResponse(
        dataset_name=dataset_name, memory_type=memory_type, rows_stored=rows_stored
    )
