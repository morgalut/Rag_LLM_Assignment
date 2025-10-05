from __future__ import annotations
import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # DB
    db_host: str = Field(default=os.getenv("DB_HOST", "localhost"))
    db_port: int = Field(default=int(os.getenv("DB_PORT", "5432")))
    db_name: str = Field(default=os.getenv("DB_NAME", "ragdb"))
    db_user: str = Field(default=os.getenv("DB_USER", "raguser"))
    db_password: str = Field(default=os.getenv("DB_PASSWORD", "ragpass"))

    # App
    k_default: int = Field(default=int(os.getenv("K_DEFAULT", "8")))
    temperature: float = Field(default=float(os.getenv("TEMPERATURE", "0.2")))
    embedding_dim: int = Field(default=int(os.getenv("EMBEDDING_DIM", "384")))
    data_path: str | None = Field(default=os.getenv("DATA_PATH"))
    tz: str = Field(default=os.getenv("TZ", "Asia/Jerusalem"))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

settings = Settings()
