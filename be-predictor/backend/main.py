from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from config import get_settings
from db.session import engine, Base, SessionLocal
from db import models as orm
from services.ml_service import load_model
from routers import predict as predict_router
from routers import chat as chat_router
from routers import health as health_router
from routers import auth as auth_router


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
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


