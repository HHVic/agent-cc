from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # LLM
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen-max"

    # Database
    database_url: str = "postgresql://postgres:postgres123@localhost:5432/agent_cc"

    # App
    app_secret_key: str = "change-me-in-production"
    debug: bool = False

    # Clarification Agent defaults
    max_exploration_rounds: int = 3
    max_clarification_rounds: int = 3
    min_questions_threshold: int = 3
    coverage_threshold: float = 0.7
    question_quality_threshold: float = 0.6

    @property
    def async_database_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
