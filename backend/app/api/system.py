"""系统接口：健康检查、工具清单、工作流结构。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .. import __version__
from ..config import get_settings
from ..core.store import get_store
from ..graph.builder import mermaid_diagram
from ..graph.runner import checkpoint_stats
from ..tools.registry import INTENT_TOOL_GROUPS, TOOL_NAMES, tool_catalog

router = APIRouter()


@router.get("/health", response_model=None)
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": __version__,
        "llm_model": settings.LLM_MODEL,
        "llm_base_url": settings.LLM_BASE_URL,
        "llm_configured": settings.llm_configured,
        "tool_count": len(TOOL_NAMES),
        "checkpoint": checkpoint_stats(),
        "datasets": len(get_store().list_datasets()),
    }


@router.get("/tools", response_model=None)
def tools() -> dict[str, Any]:
    """列出全部工具，以及各分析意图下会暴露给模型的分组。"""

    return {
        "count": len(TOOL_NAMES),
        "intent_groups": INTENT_TOOL_GROUPS,
        "items": tool_catalog(),
    }


@router.get("/graph", response_model=None)
def graph() -> dict[str, str]:
    """导出 Mermaid 流程图源码（可直接贴到 Markdown 里渲染）。"""

    return {"mermaid": mermaid_diagram()}
