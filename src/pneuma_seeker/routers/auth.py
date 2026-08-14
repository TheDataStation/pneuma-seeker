# src/pneuma_seeker/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pneuma_seeker.models import (
    ChangePasswordRequest,
    EndpointTag,
    GroupCreateRequest,
    GroupResponse,
    LoginRequest,
    PermissionKey,
    RegisterRequest,
    SetPermissionRequest,
    TokenResponse,
    UpdateGroupRequest,
    UpdateMeRequest,
    UpdateUserGroupRequest,
    UpdateUserRequest,
    UserResponse,
)
from pneuma_seeker.services.db.users.external_client import ExternalUserDB
from pneuma_seeker.services.db.users.factory import get_user_db
from pneuma_seeker.services.db.users.manager import GroupRecord, UserDB, UserRecord
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

router = APIRouter(
    prefix="/auth",
    tags=[EndpointTag.AUTH],
)

bearer_scheme = HTTPBearer(auto_error=False)

config = Config("../../../.env")
logger = setup_logger("Auth Router")
user_db = get_user_db(config, logger)


def _group_to_response(group: GroupRecord | None) -> GroupResponse | None:
    """Converts a GroupRecord to a GroupResponse, or returns None if the group is None."""
    if not group:
        return None
    return GroupResponse(
        group_id=group.group_id,
        name=group.name,
        parent_group_id=group.parent_group_id,
    )


def _user_to_response(user: UserRecord, user_db: UserDB) -> UserResponse:
    """Converts a UserRecord to a UserResponse, including group information if available."""
    group = user_db.get_group_by_id(user.group_id) if user.group_id else None
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        username=user.username,
        group_id=user.group_id,
        group_name=group.name if group else None,
        is_active=user.is_active,
    )


def _get_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Extracts the token from the Authorization header, ensuring it's a Bearer token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
        )
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )
    return credentials.credentials


def get_current_user(
    token: str = Depends(_get_token),
) -> UserRecord:
    """Validates the token and returns the associated user, or raises an HTTPException if invalid."""
    user = user_db.get_user_by_token(token)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return user


def get_current_user_group(
    current_user: UserRecord = Depends(get_current_user),
) -> GroupRecord | None:
    """Returns the group record for the current user, or None if the user has no group."""
    if not current_user.group_id:
        return None
    return user_db.get_group_by_id(current_user.group_id)


def get_current_user_permissions(
    current_user: UserRecord = Depends(get_current_user),
) -> dict[str, str]:
    """Returns the effective permissions for the current user's group, or an empty dict if no group."""
    if not current_user.group_id:
        return {}
    return user_db.get_effective_group_permissions(current_user.group_id)


@router.post("/register", response_model=UserResponse)
def register(
    payload: RegisterRequest,
):
    """Registers a new user, ensuring the email is unique and the group (if provided) exists."""
    if payload.group_id:
        group = user_db.get_group_by_id(payload.group_id)
        if not group:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Group not found",
            )
    try:
        user = user_db.create_user(
            email=payload.email,
            password=payload.password,
            group_id=payload.group_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return _user_to_response(user, user_db)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
):
    """Authenticates the user and returns a session token if successful, or raises an HTTPException if invalid.

    External mode uses ExternalUserDB.login() (one combined call) instead of
    verify_user()+create_session_token() (two local primitives) — see
    external_client.py's module docstring for why those can't compose the
    same way across the service boundary.
    """
    if isinstance(user_db, ExternalUserDB):
        token_payload = user_db.login(payload.email, payload.password)
        if not token_payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
    else:
        user = user_db.verify_user(payload.email, payload.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        token_payload = user_db.create_session_token(user.user_id)

    return TokenResponse(
        access_token=token_payload["token"],
        token_type=token_payload["token_type"],
        expires_at=token_payload["expires_at"].isoformat(),
    )


@router.post("/logout")
def logout(
    token: str = Depends(_get_token),
    _user: UserRecord = Depends(get_current_user),
):
    """Revokes the user's session token, effectively logging them out."""
    user_db.revoke_token(token)
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(
    current_user: UserRecord = Depends(get_current_user),
):
    """Returns the current authenticated user's information, or raises an HTTPException if the token is invalid."""
    return _user_to_response(current_user, user_db)


@router.get("/groups/permissions", response_model=dict[str, str])
def get_my_group_permissions(
    effective: bool = False,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Returns all permissions for the group that the authenticated user belongs to.
    
    Pass `?effective=true` to fetch full inherited permissions down the lineage tree, 
    or leave it false to get only direct group permissions.
    """
    if not current_user.group_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The authenticated user is not assigned to any group.",
        )

    try:
        if effective:
            permissions = user_db.get_effective_group_permissions(current_user.group_id)
        else:
            permissions = user_db.get_group_permissions(current_user.group_id)
            
        return permissions
    except ValueError as exc:
        # Catch potential hierarchy cycle loops if present in list_group_ancestors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def ensure_admin(email: str) -> None:
    """Verifies if the user has the explicit user:management permission flag set to 'true'."""
    user = user_db.get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: User not found."
        )
    if not user.group_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: User belongs to no group."
        )

    admin_group = user_db.get_group_by_name("admin")
    if admin_group:
        if user.group_id == admin_group.group_id:
            return  # User is in the admin group, grant access immediately

    effective_perms = user_db.get_effective_group_permissions(user.group_id)
    if effective_perms.get(PermissionKey.USER_MANAGEMENT.value) != "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: required admin permissions not found in user's effective permissions."
        )


@router.post("/groups", response_model=GroupResponse)
def create_group(
    payload: GroupCreateRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Administrative endpoint to create a new user group with an optional parent group. This endpoint is protected and requires the user to have administrative permissions. It validates the parent group if provided, creates the new group, and returns its details, or raises an HTTPException if any validation fails.
    """
    ensure_admin(current_user.email)
    if payload.parent_group_id:
        parent = user_db.get_group_by_id(payload.parent_group_id)
        if not parent:
            parent = user_db.get_group_by_name("default")
            if not parent:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent group not found, and default group does not exist. Contact administrator.",
                )
            payload.parent_group_id = parent.group_id
    try:
        group_id = user_db.create_group(payload.name, payload.parent_group_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    group = user_db.get_group_by_id(group_id)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create group",
        )
    return _group_to_response(group)


@router.get("/groups", response_model=list[GroupResponse])
def list_groups(
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Administrative endpoint to list all groups in the system. This endpoint is protected and requires the user to have administrative permissions. It returns a list of groups with their details, or raises an HTTPException if the user does not have access.
    """
    ensure_admin(current_user.email)
    return [_group_to_response(group) for group in user_db.list_groups()]


@router.post("/groups/permissions")
def set_group_permission(
    payload: SetPermissionRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Administrative endpoint to set a specific permission key-value pair for a target group, which can be specified either by group_id or group_name. If no target group is specified, the permission will be set for the current user's group. Raises HTTPExceptions for various error conditions such as missing target group, invalid permission keys, or insufficient permissions.
    """
    ensure_admin(current_user.email)
    target_group_id = current_user.group_id

    is_targeting_custom = bool(payload.group_id or payload.group_name)
    if is_targeting_custom:        
        if payload.group_id:
            target_group_id = payload.group_id
        elif payload.group_name:
            group_rec = user_db.get_group_by_name(payload.group_name)
            if not group_rec:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Target group name '{payload.group_name}' not found."
                )
            target_group_id = group_rec.group_id

    if not target_group_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No target group could be determined."
        )

    try:
        user_db.set_group_permission(
            group_id=target_group_id,
            permission_key=payload.permission_key,
            permission_value=payload.permission_value,
        )
        return {
            "status": "success",
            "group_id": target_group_id,
            "permission_key": payload.permission_key,
            "permission_value": payload.permission_value
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/users/change-group")
def change_user_group(
    payload: UpdateUserGroupRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Administrative endpoint to move a user to a different group.
    """
    ensure_admin(current_user.email)

    if not user_db.get_group_by_id(payload.group_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target group does not exist."
        )
    
    target_user = user_db.get_user_by_id(payload.user_id)
    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target user not found."
        )
    update_status = user_db.update_user(target_user.user_id, group_id=payload.group_id)
    if not update_status:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user group."
        )
    return {
        "status": "success",
        "message": f"User {payload.user_id} moved to group {payload.group_id}."
    }


# ---------------------------------------------------------------------------
# User CRUD (admin)
# ---------------------------------------------------------------------------

@router.get("/users", response_model=list[UserResponse])
def list_users(
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to list all registered users."""
    ensure_admin(current_user.email)
    return [_user_to_response(u, user_db) for u in user_db.list_users()]


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to retrieve a single user by ID."""
    ensure_admin(current_user.email)
    user = user_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return _user_to_response(user, user_db)


@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    payload: UpdateUserRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to update a user's username, group, or active status."""
    ensure_admin(current_user.email)
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")

    if "group_id" in fields and payload.group_id and not user_db.get_group_by_id(payload.group_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Group not found.")

    try:
        ok = user_db.update_user(
            user_id,
            username=payload.username if "username" in fields else None,
            group_id=payload.group_id if "group_id" in fields else None,
            is_active=payload.is_active if "is_active" in fields else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    user = user_db.get_user_by_id(user_id)
    assert user is not None
    return _user_to_response(user, user_db)


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to permanently delete a user account."""
    ensure_admin(current_user.email)
    ok = user_db.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return {"status": "success", "message": f"User {user_id} deleted."}


# ---------------------------------------------------------------------------
# Self-service profile / password
# ---------------------------------------------------------------------------

@router.patch("/me", response_model=UserResponse)
def update_me(
    payload: UpdateMeRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """Allows the authenticated user to update their own username."""
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")

    user_db.update_user(
        current_user.user_id,
        username=payload.username if "username" in fields else None,
    )
    user = user_db.get_user_by_id(current_user.user_id)
    assert user is not None
    return _user_to_response(user, user_db)


@router.post("/me/change-password")
def change_password(
    payload: ChangePasswordRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """Allows the authenticated user to change their own password after verifying the current one."""
    verified = user_db.verify_user(current_user.email, payload.current_password)
    if not verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )
    user_db.update_password(current_user.user_id, payload.new_password)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Group CRUD (admin)
# ---------------------------------------------------------------------------

@router.get("/groups/{group_id}/permissions", response_model=dict[str, str])
def get_group_permissions_by_id(
    group_id: str,
    effective: bool = False,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to fetch direct (or effective) permissions for any group by ID."""
    ensure_admin(current_user.email)
    if not user_db.get_group_by_id(group_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    if effective:
        return user_db.get_effective_group_permissions(group_id)
    return user_db.get_group_permissions(group_id)


@router.get("/groups/{group_id}", response_model=GroupResponse)
def get_group(
    group_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to retrieve a single group by ID."""
    ensure_admin(current_user.email)
    group = user_db.get_group_by_id(group_id)
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    return _group_to_response(group)


@router.patch("/groups/{group_id}", response_model=GroupResponse)
def update_group(
    group_id: str,
    payload: UpdateGroupRequest,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to rename a group or change its parent."""
    ensure_admin(current_user.email)
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")

    kwargs: dict = {}
    if "name" in fields:
        kwargs["name"] = payload.name
    if "parent_group_id" in fields:
        kwargs["parent_group_id"] = payload.parent_group_id

    try:
        ok = user_db.update_group(group_id, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")

    group = user_db.get_group_by_id(group_id)
    assert group is not None
    return _group_to_response(group)


@router.delete("/groups/{group_id}")
def delete_group(
    group_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to delete a group. Fails if users or child groups are still assigned to it."""
    ensure_admin(current_user.email)
    try:
        ok = user_db.delete_group(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    return {"status": "success", "message": f"Group {group_id} deleted."}


@router.delete("/groups/{group_id}/permissions/{permission_key}")
def delete_group_permission(
    group_id: str,
    permission_key: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """Administrative endpoint to remove a specific permission from a group."""
    ensure_admin(current_user.email)
    if not user_db.get_group_by_id(group_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    removed = user_db.delete_group_permission(group_id, permission_key)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Permission '{permission_key}' not found on group.",
        )
    return {"status": "success", "group_id": group_id, "permission_key": permission_key}
