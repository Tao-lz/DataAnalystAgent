"""Agent 状态定义。

``messages`` 用 LangGraph 官方的 ``add_messages`` reducer 追加；
``evidence`` / ``artifacts`` 是"每轮重置"的普通字段——每轮开始时清空，
这样前端拿到的就是本轮分析产生的图表与证据，而不是整个会话的堆积。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# 分析意图（由路由节点判定）
Intent = str  # chat | overview | cleaning | statistics | visualization | report

VALID_INTENTS = ("chat", "overview", "cleaning", "statistics", "visualization", "report")


class Evidence(TypedDict, total=False):
    """一条工具执行留下的证据。"""

    tool: str
    summary: str
    data: Any
    run_id: str


class Artifact(TypedDict, total=False):
    """一个图表产物。"""

    id: str
    kind: str
    title: str
    description: str
    url: str


class AgentState(TypedDict, total=False):
    """LangGraph 状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    dataset_id: str
    question: str
    intent: str
    plan: list[str]
    route_reason: str
    evidence: list[Evidence]
    artifacts: list[Artifact]
    tool_rounds: int
    status: str
    answer: str
    report_id: str | None
    error: str | None
