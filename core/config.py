from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management for the Trinity system.
    Loads from environment variables or .env file.

    Model IDs: `planner_model`, `subagent_model`, `eo_model` (see templates for usage).
    Paths: `data_dir` + `sqlite_db_name` / `lancedb_name` for SQLite trajectories and LanceDB skills.
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
    max_concurrency: int = Field(5, alias="MAX_CONCURRENCY")
    eo_multi_pass: bool = Field(True, alias="EO_MULTI_PASS")
    eo_quality_gate: bool = Field(True, alias="EO_QUALITY_GATE")
    eo_min_quality_score: float = Field(0.65, alias="EO_MIN_QUALITY_SCORE")
    eo_max_output_chars: int = Field(1200, alias="EO_MAX_OUTPUT_CHARS")
    skill_dedup_v2: bool = Field(True, alias="SKILL_DEDUP_V2")
    memory_tiering_enabled: bool = Field(False, alias="MEMORY_TIERING_ENABLED")
    memory_router_stage: str = Field("observe", alias="MEMORY_ROUTER_STAGE")
    memory_hot_max_items: int = Field(5, alias="MEMORY_HOT_MAX_ITEMS")
    skill_write_gate_stage: str = Field("observe", alias="SKILL_WRITE_GATE_STAGE")
    skill_deprecate_threshold: float = Field(0.45, alias="SKILL_DEPRECATE_THRESHOLD")
    rollout_precision_at_3_min: float = Field(0.60, alias="ROLLOUT_PRECISION_AT_3_MIN")
    rollout_recall_at_3_min: float = Field(0.70, alias="ROLLOUT_RECALL_AT_3_MIN")
    rollout_dead_end_improvement_min: float = Field(0.20, alias="ROLLOUT_DEAD_END_IMPROVEMENT_MIN")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def normalized_memory_router_stage(self) -> str:
        stage = self.memory_router_stage.strip().lower()
        return stage if stage in {"observe", "soft", "hard"} else "observe"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def normalized_skill_write_gate_stage(self) -> str:
        stage = self.skill_write_gate_stage.strip().lower()
        return stage if stage in {"observe", "soft", "hard"} else "observe"

settings = Settings()  # type: ignore[call-arg]
