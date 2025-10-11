import os
from functools import lru_cache


class Settings:
    database_url: str
    model_path: str
    allowed_origins: list[str]
    log_level: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

    def __init__(self) -> None:
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql+psycopg://postgres:richie@localhost:5432/Diabetes"
        )
        self.model_path = os.getenv("MODEL_PATH", "./backend/models/model.joblib")
        self.allowed_origins = [
            origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if origin.strip()
        ]
        self.log_level = os.getenv("LOG_LEVEL", "info")
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()


