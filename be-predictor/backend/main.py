from __future__ import annotations

import os
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .config import get_settings
from .db.session import engine, Base, SessionLocal
from .db import models as orm
from .services.ml_service import load_model
from .routers import predict as predict_router
from .routers import chat as chat_router
from .routers import health as health_router
from .routers import auth as auth_router

# Load .env file from the backend directory
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(env_path)

# If .env not found in backend directory, try parent directory
if not env_path.exists():
    parent_env_path = backend_dir.parent / ".env"
    print(f"🔍 .env not found in backend, trying parent directory: {parent_env_path}")
    load_dotenv(parent_env_path)

# Debug: Check if .env file exists and log environment variables
print(f"🔍 Looking for .env file at: {env_path}")
print(f"🔍 .env file exists: {env_path.exists()}")
print(f"🔍 DEEPSEEK_API_KEY from env: {os.getenv('DEEPSEEK_API_KEY', 'NOT_FOUND')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    # Log effective DB connection info (masked) to verify .env loading
    try:
        settings = get_settings()
        db_url = settings.database_url
        # mask password
        masked = db_url
        if "@" in db_url and "://" in db_url:
            scheme_sep = db_url.split("://", 1)
            creds_and_rest = scheme_sep[1]
            if "@" in creds_and_rest and ":" in creds_and_rest.split("@", 1)[0]:
                user = creds_and_rest.split(":", 1)[0]
                rest_after_user = creds_and_rest.split(":", 1)[1]
                if "@" in rest_after_user:
                    rest = rest_after_user.split("@", 1)[1]
                    masked = f"{scheme_sep[0]}://{user}:***@{rest}"
        logging.info(f"Using DATABASE_URL: {masked}")
    except Exception:
        pass
    Base.metadata.create_all(bind=engine)
    # seed guest user (id=1) if not present for anonymous predictions linkage
    db = SessionLocal()
    try:
        guest = db.query(orm.User).filter(orm.User.id == 1).first()
        if guest is None:
            guest = orm.User(id=1, username="guest", email=None, password="guest")
            db.add(guest)
            db.commit()
    finally:
        db.close()
    load_model()
    yield
    # shutdown


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Diabetes Predictor API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predict_router.router)
    app.include_router(chat_router.router)
    app.include_router(health_router.router)
    app.include_router(auth_router.router, prefix="/auth", tags=["authentication"])
    return app


app = create_app()


