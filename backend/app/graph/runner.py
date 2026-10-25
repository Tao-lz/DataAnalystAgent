"""Agent 调用入口：把一次 HTTP 请求翻译成一次图执行。"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..core.store import get_store, now_iso
from ..tools.context import ToolContext, reset_context, set_context
from .builder import get_graph

NODE_LABELS = {
    "understand": "意图识别与规划",
    "chat": "直接回答",
    "agent": "Agent 推理",
    "tools": "工具调用",
    "wrap_up": "收尾归纳",
    "report": "生成结构化报告",
    "finalize": "汇总答复",
}


def new_conversation_id() -> str:
    return f"C{uuid.uuid4().hex[:12]}"


def _clean_title(text: str | None, limit: int = 24) -> str | None:
    """用首条提问给会话起个标题，方便在「历史会话」下拉框里辨认。"""

    if not text:
        return None
    compact = " ".join(str(text).split())
    if not compact:
        return None
    return compact[:limit] + ("…" if len(compact) > limit else "")


def _config(conversation_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": 80,
    }


def prepare_context(
    dataset_id: str,
    conversation_id: str | None,
    title: str | None = None,
) -> tuple[ToolContext, str]:
    """建立或复用会话，返回绑定了数据集与线程 id 的上下文。"""

    store = get_store()
    if not store.get_dataset(dataset_id):
        raise ValueError(f"数据集 {dataset_id} 不存在，请先上传文件。")
    conv_id = conversation_id or new_conversation_id()
    # 标题只在首次建会话时写入（ON CONFLICT 不会覆盖），后续消息不会改写它
    store.ensure_conversation(conv_id, dataset_id, _clean_title(title))
    return ToolContext(dataset_id=dataset_id, conversation_id=conv_id), conv_id


def _response(state: dict[str, Any], ctx: ToolContext, conversation_id: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "dataset_id": ctx.dataset_id,
        "intent": state.get("intent"),
        "plan": list(state.get("plan") or []),
        "route_reason": state.get("route_reason"),
        "answer": str(state.get("answer") or ""),
        "evidence": list(state.get("evidence") or ctx.evidence),
        "artifacts": list(state.get("artifacts") or ctx.artifacts),
        "tool_rounds": int(state.get("tool_rounds") or 0),
        "status": state.get("status") or "completed",
        "report_id": state.get("report_id"),
        "created_at": now_iso(),
    }


def run_agent(dataset_id: str, message: str, conversation_id: str | None = None) -> dict[str, Any]:
    """执行一轮完整分析，返回最终答复与证据。"""

    ctx, conv_id = prepare_context(dataset_id, conversation_id, title=message)
    store = get_store()
    token = set_context(ctx)
    try:
        state = get_graph().invoke(
            {
                "messages": [HumanMessage(content=message)],
                "dataset_id": dataset_id,
                "question": message,
            },
            _config(conv_id),
        )
    finally:
        reset_context(token)
    store.touch_conversation(conv_id)
    return _response(state, ctx, conv_id)


def _step_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """把节点的状态更新压缩成前端可展示的一步。"""

    event: dict[str, Any] = {"type": "step", "node": node, "label": NODE_LABELS.get(node, node)}

    if node == "understand":
        event.update(
            {
                "intent": update.get("intent"),
                "plan": update.get("plan"),
                "reason": update.get("route_reason"),
            }
        )
    elif node == "agent":
        messages = update.get("messages") or []
        last = messages[-1] if messages else None
        calls = list(getattr(last, "tool_calls", None) or [])
        if calls:
            event["tool_calls"] = [
                {"name": str(c.get("name")), "args": c.get("args") or {}} for c in calls
            ]
        else:
            content = getattr(last, "content", "") if last is not None else ""
            event["text"] = content if isinstance(content, str) else str(content)
    elif node == "tools":
        results = []
        for msg in update.get("messages") or []:
            if not isinstance(msg, ToolMessage):
                continue
            item: dict[str, Any] = {"tool": getattr(msg, "name", None)}
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            try:
                payload = json.loads(raw)
                item["ok"] = bool(payload.get("ok"))
                item["summary"] = payload.get("summary") or payload.get("error")
                chart = (payload.get("data") or {}).get("chart")
                if chart:
                    item["chart"] = chart
            except (ValueError, TypeError):
                item["ok"] = True
                item["summary"] = raw[:200]
            results.append(item)
        event["results"] = results
    elif node == "report":
        event["report_id"] = update.get("report_id")
    elif node in ("chat", "wrap_up", "finalize"):
        event["text"] = str(update.get("answer") or "")[:400]

    return event


def stream_agent(
    dataset_id: str,
    message: str,
    conversation_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """流式执行：逐个节点产出事件，最后给出一份完整结果。

    注意：工具是通过 ContextVar 拿到当前数据集的，而 Starlette 在流式响应里会用
    线程池逐次 ``next()`` 这个同步生成器——每次恢复执行都发生在**拷贝出来的新上下文**，
    生成器开头的 ``set_context`` 会失效。所以这里每恢复一次就重新绑定一次上下文。
    """

    ctx, conv_id = prepare_context(dataset_id, conversation_id, title=message)
    store = get_store()
    final_state: dict[str, Any] = {}
    token = set_context(ctx)

    def inner() -> Iterator[dict[str, Any]]:
        yield {"type": "start", "conversation_id": conv_id, "dataset_id": dataset_id}
        for chunk in get_graph().stream(
            {
                "messages": [HumanMessage(content=message)],
                "dataset_id": dataset_id,
                "question": message,
            },
            _config(conv_id),
            stream_mode="updates",
        ):
            for node, update in (chunk or {}).items():
                if isinstance(update, dict):
                    final_state.update(update)
                    yield _step_event(node, update)
        store.touch_conversation(conv_id)
        yield {"type": "final", **_response(final_state, ctx, conv_id)}

    iterator = inner()
    try:
        while True:
            set_context(ctx)
            try:
                event = next(iterator)
            except StopIteration:
                break
            yield event
    finally:
        try:
            reset_context(token)
        except (ValueError, LookupError):
            # token 可能属于另一个 Context，忽略即可
            pass


def conversation_history(conversation_id: str) -> list[dict[str, Any]]:
    """从 checkpoint 里读出某个会话的全部消息。"""

    try:
        snapshot = get_graph().get_state(_config(conversation_id))
    except Exception:  # noqa: BLE001
        return []
    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages") or []

    history: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            calls = [
                {"name": str(c.get("name")), "args": c.get("args") or {}}
                for c in (getattr(msg, "tool_calls", None) or [])
            ]
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            history.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif isinstance(msg, ToolMessage):
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            try:
                payload = json.loads(raw)
                summary = payload.get("summary") or payload.get("error") or ""
            except (ValueError, TypeError):
                summary = raw[:200]
            history.append({"role": "tool", "tool": getattr(msg, "name", None), "content": summary})
    return history


def delete_conversation(conversation_id: str) -> None:
    get_graph().checkpointer.delete_thread(conversation_id)
    get_store().delete_conversation(conversation_id)


def checkpoint_stats() -> dict[str, Any]:
    checkpointer = get_graph().checkpointer
    return {
        "backend": type(checkpointer).__name__,
        "db": str(getattr(checkpointer, "db_path", "")),
        **checkpointer.stats(),
    }
