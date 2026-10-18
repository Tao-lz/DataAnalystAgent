"""报告接口：一键生成、查询、下载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from ..core.store import get_store
from ..graph.runner import run_agent
from ..schemas import ReportRequest

router = APIRouter()


@router.post("/", response_model=None)
def create_report(payload: ReportRequest) -> dict[str, Any]:
    """触发一次完整的报告生成流程（会真实执行统计与绘图工具）。"""

    message = "请对这个数据集做一次全面分析，并生成一份完整的结构化分析报告。"
    if payload.focus:
        message += f"侧重：{payload.focus}"
    try:
        result = run_agent(payload.dataset_id, message, payload.conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    report_id = result.get("report_id")
    if not report_id:
        raise HTTPException(
            status_code=502,
            detail=f"报告生成失败，Agent 返回状态：{result.get('status')}。{result.get('answer', '')[:200]}",
        )
    stored = get_store().get_report(report_id) or {}
    return {
        "report_id": report_id,
        "conversation_id": result["conversation_id"],
        "title": stored.get("title"),
        "markdown": stored.get("markdown"),
        "html": stored.get("html"),
        "payload": stored.get("payload") or {},
        "answer": result.get("answer"),
        "artifacts": result.get("artifacts") or [],
    }


@router.get("/", response_model=None)
def list_reports(dataset_id: str | None = Query(default=None)) -> dict[str, Any]:
    return {"items": get_store().list_reports(dataset_id)}


@router.get("/{report_id}", response_model=None)
def get_report(report_id: str) -> dict[str, Any]:
    row = get_store().get_report(report_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"报告 {report_id} 不存在。")
    return {
        "report_id": row["id"],
        "dataset_id": row["dataset_id"],
        "conversation_id": row["conversation_id"],
        "title": row["title"],
        "generated_at": row["generated_at"],
        "markdown": row["markdown"],
        "html": row["html"],
        "payload": row["payload"],
    }


@router.get("/{report_id}/download", response_model=None)
def download_report(
    report_id: str,
    fmt: str = Query(default="html", pattern="^(html|md)$"),
    download: bool = Query(
        default=False,
        description="false=inline（供页面 iframe 内嵌预览），true=attachment（触发浏览器下载）",
    ),
):
    """获取报告文件。

    默认返回 ``inline``：带 ``Content-Disposition: attachment`` 的响应浏览器不会在
    iframe 里渲染，报告预览会变成一片空白，所以只有明确要求下载时才加 attachment。
    """

    row = get_store().get_report(report_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"报告 {report_id} 不存在。")

    if fmt == "md":
        path = row.get("markdown_path")
        filename = f"{report_id}.md" if download else None
        if path and Path(path).exists():
            return FileResponse(path, media_type="text/markdown", filename=filename)
        return PlainTextResponse(row.get("markdown") or "", media_type="text/markdown")

    path = row.get("html_path")
    if path and Path(path).exists():
        return FileResponse(
            path,
            media_type="text/html",
            filename=f"{report_id}.html" if download else None,
        )
    return HTMLResponse(row.get("html") or "")
