# src/pneuma_seeker/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pneuma_seeker.routers import auth, chat, indexing
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block executes ONCE when the server boots up
    config = Config("../../.env")
    logger = setup_logger("App Bootstrapping")

    logger.info("Ensuring database schema is initialized...")
    db_initializer = UserDB(config, logger)
    db_initializer.init_db() # Initializes schema safely before any requests arrive
    
    yield
    # Any cleanup code goes here

app = FastAPI(title="Pneuma-Seeker", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(indexing.router)

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
