"""全局配置。

统一从项目根目录的 .env 读取，并同时兼容 ``LLM_*`` 与 ``DEEPSEEK_*`` 两套变量名，
方便直接复用已有的 DeepSeek 配置。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """运行时配置（环境变量 / .env 优先级高于这里的默认值）。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    APP_NAME: str = "DataAnalysisAgent"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # ---------- 大模型 ----------
    LLM_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    LLM_BASE_URL: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("LLM_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    LLM_MODEL: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("LLM_MODEL", "DEEPSEEK_MODEL"),
    )
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: float = 120.0
    LLM_MAX_RETRIES: int = 2
    # 结构化输出方式：function_calling / json_schema / json_mode
    # DeepSeek 目前只支持 function_calling（json_schema 会返回 400）
    LLM_STRUCTURED_METHOD: str = "function_calling"

    # ---------- 上传 / 存储 ----------
    MAX_FILE_SIZE: int = 200 * 1024 * 1024
    ALLOWED_FILE_TYPES: str = ".csv,.xlsx,.xls,.txt,.tsv"
    STORAGE_DIR: Path = Path("./storage")
    UPLOAD_SUBDIR: str = "uploads"
    ARTIFACT_SUBDIR: str = "artifacts"
    REPORT_SUBDIR: str = "reports"
    META_DB_NAME: str = "agent.db"
    CHECKPOINT_DB_NAME: str = "checkpoints.db"
    MAX_ROWS: int = 200_000

    # ---------- Agent 行为 ----------
    MAX_TOOL_ROUNDS: int = 8
    HISTORY_WINDOW: int = 24
    CHART_DPI: int = 120
    CHART_THEME: str = "seaborn-v0_8-darkgrid"

    # ---------- 校验 ----------
    @field_validator("STORAGE_DIR", mode="before")
    @classmethod
    def _strip_quotes(cls, v):
        return v.strip().strip("'\"") if isinstance(v, str) else v

    # ---------- 派生路径 ----------
    def _ensure(self, sub: str) -> Path:
        root = self.STORAGE_DIR
        root = root if root.is_absolute() else PROJECT_ROOT / root
        path = (root / sub).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def storage_path(self) -> Path:
        return self._ensure("")

    @property
    def upload_path(self) -> Path:
        return self._ensure(self.UPLOAD_SUBDIR)

    @property
    def artifact_path(self) -> Path:
        return self._ensure(self.ARTIFACT_SUBDIR)

    @property
    def report_path(self) -> Path:
        return self._ensure(self.REPORT_SUBDIR)

    @property
    def meta_db_path(self) -> Path:
        return self.storage_path / self.META_DB_NAME

    @property
    def checkpoint_db_path(self) -> Path:
        return self.storage_path / self.CHECKPOINT_DB_NAME

    # ---------- 便捷属性 ----------
    @property
    def allowed_extensions(self) -> list[str]:
        return [e.strip().lower() for e in self.ALLOWED_FILE_TYPES.split(",") if e.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.LLM_API_KEY and self.LLM_API_KEY.startswith(("sk-", "sess-", "Bearer")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""

    return Settings()


settings = get_settings()
