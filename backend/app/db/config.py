# backend/app/db/config.py
from __future__ import annotations
import logging
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env before reading settings
load_dotenv()

# Create dedicated logger for this module
logger = logging.getLogger("rag.db.config")

class Settings(BaseSettings):
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_user: str = Field("postgres", env="DB_USER")
    db_password: str = Field("postgres", env="DB_PASSWORD")
    db_name: str = Field("ragdb", env="DB_NAME")

    embedding_dim: int = Field(768, env="EMBEDDING_DIM")
    data_path: str | None = Field(None, env="DATA_PATH")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

# Instantiate settings once
settings = Settings()

# Mask password for safe logging
masked = settings.database_url.replace(settings.db_password, "*****")
logger.info(f"Connecting to database using DSN: {masked}")
