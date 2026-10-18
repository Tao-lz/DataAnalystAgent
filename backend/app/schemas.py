"""HTTP 接口的请求 / 响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    id: str
    name: str
    filename: str
    rows: int = 0
    cols: int = 0
    size_bytes: int = 0
    sheet: str | None = None
    encoding: str | None = None
    revision: int = 0
    quality_score: float | None = None
    created_at: str | None = None
    updated_at: str | None = None


class UploadResponse(BaseModel):
    dataset: dict[str, Any]
    profile: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class PreviewResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    total_rows: int
    offset: int
    limit: int


class ChatRequest(BaseModel):
    dataset_id: str = Field(description="数据集 id")
    message: str = Field(min_length=1, description="自然语言分析需求")
    conversation_id: str | None = Field(default=None, description="留空则新建会话")


class ChatResponse(BaseModel):
    conversation_id: str
    dataset_id: str
    intent: str | None = None
    plan: list[str] = Field(default_factory=list)
    route_reason: str | None = None
    answer: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    tool_rounds: int = 0
    status: str = "completed"
    report_id: str | None = None
    created_at: str | None = None


class ConversationInfo(BaseModel):
    id: str
    dataset_id: str
    title: str | None = None
    turns: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class ReportRequest(BaseModel):
    dataset_id: str
    conversation_id: str | None = None
    focus: str | None = Field(
        default=None,
        description="可选的分析侧重点，例如「重点关注各地区的销售差异」",
    )


class ToolInfo(BaseModel):
    name: str
    group: str
    group_label: str
    description: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_model: str
    llm_base_url: str
    llm_configured: bool
    tool_count: int
    checkpoint: dict[str, Any]
    datasets: int
