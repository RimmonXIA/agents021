from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")

    chat_model: str = Field("deepseek-chat", alias="ARCH_CHAT_MODEL")
    data_dir: str = Field("data", alias="DATA_DIR")
    max_retries: int = Field(3, alias="ARCH_CHAT_MAX_RETRIES")
    real_search: bool = Field(False, alias="ARCH_CHAT_REAL_SEARCH")
    verbose: bool = Field(True, alias="ARCH_CHAT_VERBOSE")

    @property
    def memory_db_path(self) -> str:
        return str(Path(self.data_dir) / "arch_chat_memory.db")

    @property
    def lancedb_dir(self) -> str:
        return str(Path(self.data_dir) / "arch_chat_lancedb")

    @property
    def sessions_db_path(self) -> str:
        return str(Path(self.data_dir) / "arch_chat_sessions.db")


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
