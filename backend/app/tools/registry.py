"""工具注册中心。

按能力分组，并支持"路由感知的工具绑定"：不同分析意图只把相关工具挂给大模型，
既降低选错工具的概率，也减少每次请求的 token 消耗。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .cleaning import CLEANING_TOOLS
from .profiling import PROFILING_TOOLS
from .query import QUERY_TOOLS
from .statistics import STATISTICS_TOOLS
from .visualize import VISUALIZATION_TOOLS

# 能力分组
TOOL_GROUPS: dict[str, list[BaseTool]] = {
    "profiling": list(PROFILING_TOOLS),
    "cleaning": list(CLEANING_TOOLS),
    "analytics": list(STATISTICS_TOOLS) + list(QUERY_TOOLS),
    "visualization": list(VISUALIZATION_TOOLS),
}

GROUP_LABELS = {
    "profiling": "数据解析与质量诊断",
    "cleaning": "数据清洗",
    "analytics": "统计计算与查询",
    "visualization": "可视化",
}

# 意图 -> 可用工具分组（由 LangGraph 的条件路由决定）
INTENT_TOOL_GROUPS: dict[str, list[str]] = {
    "chat": [],
    "overview": ["profiling"],
    "cleaning": ["profiling", "cleaning"],
    "statistics": ["profiling", "analytics"],
    "visualization": ["profiling", "analytics", "visualization"],
    "report": ["profiling", "cleaning", "analytics", "visualization"],
}

ALL_TOOLS: list[BaseTool] = [tool for group in TOOL_GROUPS.values() for tool in group]
TOOLS_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in ALL_TOOLS}
TOOL_NAMES: list[str] = sorted(TOOLS_BY_NAME)


def tools_for_intent(intent: str) -> list[BaseTool]:
    """取某个分析意图下应该暴露给模型的工具集合。"""

    groups = INTENT_TOOL_GROUPS.get(intent, INTENT_TOOL_GROUPS["report"])
    return [tool for group in groups for tool in TOOL_GROUPS.get(group, [])]


def build_tools(intent: str = "report") -> list[BaseTool]:
    return tools_for_intent(intent)


def tool_catalog() -> list[dict[str, str]]:
    """给前端/README 用的工具清单。"""

    items: list[dict[str, str]] = []
    for group, tools in TOOL_GROUPS.items():
        for tool in tools:
            first_line = (tool.description or "").strip().splitlines()[0]
            items.append(
                {
                    "name": tool.name,
                    "group": group,
                    "group_label": GROUP_LABELS.get(group, group),
                    "description": first_line,
                }
            )
    return items
