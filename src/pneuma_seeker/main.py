# src/pneuma_seeker/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pneuma_seeker.routers import auth, chat, indexing
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

app = FastAPI(title="Pneuma-Seeker")

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
