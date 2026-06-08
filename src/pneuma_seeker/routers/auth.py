from logging import Logger

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pneuma_seeker.model import EndpointTag, GroupCreateRequest, GroupResponse, LoginRequest, RegisterRequest, TokenResponse, UserResponse
from pneuma_seeker.services.db.users.manager import GroupRecord, UserDB, UserRecord
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

router = APIRouter(
    prefix="/auth",
    tags=[EndpointTag.AUTH],
)

bearer_scheme = HTTPBearer(auto_error=False)


def get_config() -> Config:
    """Returns a Config instance initialized from the .env file."""
    return Config("../../../.env")


def get_logger() -> Logger:
    """Returns a Logger instance for the Auth Router."""
    return setup_logger("Auth Router")


def get_user_db(
    config: Config = Depends(get_config), 
    logger: Logger = Depends(get_logger)
) -> UserDB:
    """Returns a UserDB instance initialized with the provided Config and Logger."""
    return UserDB(config, logger)


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
    user_db: UserDB = Depends(get_user_db)
) -> UserRecord:
    """Validates the token and returns the associated user, or raises an HTTPException if invalid."""
    user = user_db.get_user_by_token(token)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return user


@router.post("/register", response_model=UserResponse)
def register(payload: RegisterRequest, user_db: UserDB = Depends(get_user_db)):
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
def login(payload: LoginRequest, user_db: UserDB = Depends(get_user_db)):
    """Authenticates the user and returns a session token if successful, or raises an HTTPException if invalid."""
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
    user_db: UserDB = Depends(get_user_db),
):
    """Revokes the user's session token, effectively logging them out."""
    user_db.revoke_token(token)
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(
    current_user: UserRecord = Depends(get_current_user), 
    user_db: UserDB = Depends(get_user_db)
):
    """Returns the current authenticated user's information, or raises an HTTPException if the token is invalid."""
    return _user_to_response(current_user, user_db)


@router.post("/groups", response_model=GroupResponse)
def create_group(
    payload: GroupCreateRequest,
    _current_user: UserRecord = Depends(get_current_user),
    user_db: UserDB = Depends(get_user_db),
):
    """Creates a new group, ensuring the parent group (if provided) exists, and returns the created group's information."""
    if payload.parent_group_id:
        parent = user_db.get_group_by_id(payload.parent_group_id)
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent group not found",
            )
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
    _current_user: UserRecord = Depends(get_current_user),
    user_db: UserDB = Depends(get_user_db),
):
    """Returns a list of all groups, or raises an HTTPException if the user is not authenticated."""
    return [_group_to_response(group) for group in user_db.list_groups()]
