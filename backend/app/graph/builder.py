"""组装 LangGraph 工作流。

流程概览：

    START -> understand（意图识别 + 规划）
        ├── intent=chat ──────────────> chat -> END
        └── intent≠chat ──────────────> agent
                                          ├── 有 tool_calls 且轮次未超限 -> tools -> agent（循环）
                                          ├── 有 tool_calls 但轮次超限 ──> wrap_up -> finalize -> END
                                          ├── 无 tool_calls 且 intent=report -> report -> finalize -> END
                                          └── 无 tool_calls ─────────────> finalize -> END
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .checkpointer import get_checkpointer, reset_checkpointer
from .nodes import (
    agent_node,
    chat_node,
    finalize_node,
    report_node,
    route_after_agent,
    route_after_understand,
    tools_node,
    understand_node,
    wrap_up_node,
)
from .state import AgentState


def build_workflow() -> StateGraph:
    """搭建（未编译的）状态图。"""

    builder = StateGraph(AgentState)

    builder.add_node("understand", understand_node)
    builder.add_node("chat", chat_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("wrap_up", wrap_up_node)
    builder.add_node("report", report_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        route_after_understand,
        {"chat": "chat", "agent": "agent"},
    )
    builder.add_edge("chat", END)

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "wrap_up": "wrap_up", "report": "report", "finalize": "finalize"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("wrap_up", "finalize")
    builder.add_edge("report", "finalize")
    builder.add_edge("finalize", END)

    return builder


@lru_cache(maxsize=1)
def get_graph():
    """编译并缓存图实例（带 SQLite checkpointer）。"""

    return build_workflow().compile(checkpointer=get_checkpointer())


def reset_graph() -> None:
    """清空图与 checkpointer 缓存（测试 / 热重载用）。"""

    get_graph.cache_clear()
    reset_checkpointer()


def mermaid_diagram() -> str:
    """导出 Mermaid 流程图源码，方便写文档。"""

    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001
        return f"<!-- 导出失败：{exc} -->"
