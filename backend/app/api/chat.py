"""对话接口：非流式、SSE 流式、历史读取、会话管理。"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..core.store import get_store
from ..graph.runner import (
    conversation_history,
    delete_conversation,
    run_agent,
    stream_agent,
)
from ..schemas import ChatRequest

router = APIRouter()


@router.post("/", response_model=None)
def chat(payload: ChatRequest) -> dict[str, Any]:
    """执行一轮分析（同步返回，适合脚本调用）。"""

    try:
        return run_agent(payload.dataset_id, payload.message, payload.conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/stream", response_model=None)
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """执行一轮分析并流式推送每个节点的进展（Server-Sent Events）。

    事件类型：``start`` / ``step``（含 ``node``、``tool_calls``、``results``）/ ``final`` / ``error``。
    """

    def generator() -> Iterator[str]:
        try:
            for event in stream_agent(payload.dataset_id, payload.message, payload.conversation_id):
                yield _sse(event)
        except ValueError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/history/{conversation_id}", response_model=None)
def history(conversation_id: str) -> dict[str, Any]:
    """从 checkpoint 中读出完整多轮上下文（服务重启后依然可读）。"""

    return {
        "conversation_id": conversation_id,
        "messages": conversation_history(conversation_id),
    }


@router.get("/conversations", response_model=None)
def list_conversations(
    dataset_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """列出会话（可按数据集过滤）。"""

    return {"items": get_store().list_conversations(dataset_id)}


@router.delete("/conversations/{conversation_id}", response_model=None)
def remove_conversation(conversation_id: str) -> dict[str, str]:
    """删除会话：同时清掉 checkpoint 与元数据。"""

    delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}
