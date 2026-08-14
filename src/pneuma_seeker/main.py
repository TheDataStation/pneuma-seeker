# src/pneuma_seeker/main.py
from contextlib import asynccontextmanager
from os import getenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pneuma_seeker.routers import auth, chat, indexing, memory
from pneuma_seeker.services.db.memory.manager import MemoryManager
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block executes ONCE when the server boots up
    config = Config("../../.env")
    logger = setup_logger("App Bootstrapping")

    if config.AUTH_BACKEND not in ("local", "external"):
        raise RuntimeError(f"AUTH_BACKEND must be 'local' or 'external', got {config.AUTH_BACKEND!r}")
    if config.AUTH_BACKEND == "external" and not (
        config.AUTH_BACKEND_EXTERNAL_URL and config.AGENTIC_CATALOG_SERVICE_TOKEN
    ):
        raise RuntimeError(
            "AUTH_BACKEND_EXTERNAL_URL and AGENTIC_CATALOG_SERVICE_TOKEN must both be set "
            "when AUTH_BACKEND=external."
        )

    logger.info("Ensuring database schema is initialized...")

    # Always runs against THIS service's own local Postgres, regardless of
    # AUTH_BACKEND — deliberately unconditional (see users/manager.py's
    # UserDB, never modified by this feature) so the local path stays ready
    # to serve immediately if AUTH_BACKEND is ever flipped back to "local"
    # from "external" — a real rollback shouldn't also require re-running
    # bootstrap by hand.
    db_initializer = UserDB(config, logger)
    db_initializer.init_db()  # Initializes schema safely before any requests arrive

    memory_initializer = MemoryManager(config, logger)
    memory_initializer.init_db()

    admin_group = db_initializer.get_group_by_name("admin")
    if admin_group is None:
        raise RuntimeError(
            "Admin group was not created successfully during initialization"
        )

    admin_email = getenv("ADMIN_EMAIL", "pneuma_admin@pneuma.com")
    admin_password = getenv("ADMIN_PASSWORD")

    if admin_password is None:
        raise RuntimeError(
            "ADMIN_PASSWORD environment variable must be set for secure admin user creation"
        )

    admin_user = db_initializer.get_user_by_email(admin_email)

    if admin_user is None:
        logger.info("Admin user not found, creating default admin user...")

        db_initializer.create_user(
            email=admin_email,
            password=admin_password,
            group_id=admin_group.group_id,
        )

        admin_user = db_initializer.get_user_by_email(admin_email)

        if admin_user is None:
            raise RuntimeError(
                "Admin user was not created successfully during initialization"
            )

    yield


app = FastAPI(title="Pneuma-Seeker", lifespan=lifespan)

# All backend routes live under /api so a reverse proxy can route by a single
# unambiguous prefix ("/api/* -> backend, everything else -> frontend").
# Without this, backend routes could collide with frontend page routes of the
# same name (e.g. chat.router's DELETE /chat/{chat_id} vs. the frontend's own
# GET /chat/{chatId} page route) — a real production incident where a hard
# refresh on a chat page got misrouted to the backend and hit that DELETE
# route's path pattern, returning 405 Method Not Allowed for the GET.
app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(indexing.router, prefix="/api")
app.include_router(memory.router, prefix="/api")

logger = setup_logger()
config = Config("../../.env")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok"}
