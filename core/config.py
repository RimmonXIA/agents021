from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management for the Trinity system.
    Loads from environment variables or .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys
    deepseek_api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")

    # Model Configuration
    planner_model: str = Field("deepseek-reasoner", alias="PLANNER_MODEL")
    subagent_model: str = Field("deepseek-chat", alias="SUBAGENT_MODEL")
    eo_model: str = Field("deepseek-reasoner", alias="EO_MODEL")

    # Database Paths
    data_dir: str = Field("data", alias="DATA_DIR")
    sqlite_db_name: str = Field("trinity_memory.db", alias="SQLITE_DB_NAME")
    lancedb_name: str = Field("trinity_lancedb", alias="LANCEDB_NAME")

    @property
    def sqlite_db_path(self) -> str:
        return str(Path(self.data_dir) / self.sqlite_db_name)

    @property
    def lancedb_dir(self) -> str:
        return str(Path(self.data_dir) / self.lancedb_name)

    # System Behavior
    verbose: bool = Field(True, alias="VERBOSE")
    max_retries: int = Field(3, alias="MAX_RETRIES")

settings = Settings()  # type: ignore[call-arg]
