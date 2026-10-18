"""SQLite Checkpoint 持久化测试（多轮上下文、跨实例恢复、删除）。"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ._harness import expect, expect_eq
from .fixtures import temp_dir


class ToyState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    counter: int
    note: str


def _bump(state: ToyState) -> dict:
    return {"counter": int(state.get("counter") or 0) + 1}


def _note(state: ToyState) -> dict:
    return {"note": f"轮次 {state.get('counter')}"}


def _toy_graph(saver):
    builder = StateGraph(ToyState)
    builder.add_node("bump", _bump)
    builder.add_node("note", _note)
    builder.add_edge(START, "bump")
    builder.add_edge("bump", "note")
    builder.add_edge("note", END)
    return builder.compile(checkpointer=saver)


def test_multiturn_context_is_persisted() -> None:
    from backend.app.graph.checkpointer import SqliteCheckpointSaver

    with temp_dir("cp-multi") as tmp:
        db = tmp / "cp.db"
        config = {"configurable": {"thread_id": "t-multi"}}

        saver = SqliteCheckpointSaver(db)
        graph = _toy_graph(saver)
        graph.invoke({"messages": [("user", "第一轮")]}, config)
        out = graph.invoke({"messages": [("user", "第二轮")]}, config)
        expect_eq(out["counter"], 2, "第二轮应基于第一轮的 counter")
        expect_eq(len(out["messages"]), 2, "两轮消息都应保留")
        stats = saver.stats()
        expect(stats["checkpoints"] > 0, "应有 checkpoint 落库")
        expect(stats["writes"] > 0, "应有 writes 落库")


def test_state_survives_process_restart() -> None:
    """丢到对象重新 new 一个 saver，模拟服务重启。"""

    from backend.app.graph.checkpointer import SqliteCheckpointSaver

    with temp_dir("cp-restart") as tmp:
        db = tmp / "cp.db"
        config = {"configurable": {"thread_id": "t-restart"}}

        saver1 = SqliteCheckpointSaver(db)
        graph1 = _toy_graph(saver1)
        graph1.invoke({"messages": [("user", "重启前")]}, config)
        stats_before = saver1.stats()
        del graph1, saver1

        saver2 = SqliteCheckpointSaver(db)
        expect_eq(saver2.stats(), stats_before, "重新加载后数据量应一致")
        snapshot = _toy_graph(saver2).get_state(config)
        expect(bool(snapshot.values), "应能恢复出状态")
        expect_eq(snapshot.values["counter"], 1, "counter 应被恢复")
        expect_eq(
            [m.content for m in snapshot.values["messages"]],
            ["重启前"],
            "历史消息应被恢复",
        )

        out = _toy_graph(saver2).invoke({"messages": [("user", "重启后")]}, config)
        expect_eq(out["counter"], 2, "重启后应继续累加而不是从零开始")


def test_list_history_and_delete_thread() -> None:
    from backend.app.graph.checkpointer import SqliteCheckpointSaver

    with temp_dir("cp-del") as tmp:
        db = tmp / "cp.db"
        config = {"configurable": {"thread_id": "t-del"}}
        saver = SqliteCheckpointSaver(db)
        graph = _toy_graph(saver)
        graph.invoke({"messages": [("user", "A")]}, config)
        graph.invoke({"messages": [("user", "B")]}, config)

        history = list(saver.list(config))
        expect(len(history) >= 2, f"应能列出多轮 checkpoint，实际 {len(history)}")
        expect_eq(len(saver.threads()), 1, "应只有一个线程")

        saver.delete_thread("t-del")
        expect_eq(saver.stats(), {"checkpoints": 0, "blobs": 0, "writes": 0}, "删除后应为空")
        expect_eq(len(saver.threads()), 0, "线程列表应为空")


def test_workflow_compiles_with_checkpointer() -> None:
    from backend.app.graph.builder import build_workflow

    workflow = build_workflow()
    for node in ("understand", "chat", "agent", "tools", "wrap_up", "report", "finalize"):
        expect(node in workflow.nodes, f"缺少节点 {node}")


def test_route_functions() -> None:
    from backend.app.graph.nodes import route_after_agent, route_after_understand

    expect_eq(route_after_understand({"intent": "chat"}), "chat")
    expect_eq(route_after_understand({"intent": "statistics"}), "agent")

    from langchain_core.messages import AIMessage

    plain = AIMessage(content="结论")
    expect_eq(route_after_agent({"messages": [plain], "intent": "statistics"}), "finalize")
    expect_eq(route_after_agent({"messages": [plain], "intent": "report"}), "report")

    with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "dataset_overview", "args": {}, "id": "call-1"}],
    )
    expect_eq(
        route_after_agent({"messages": [with_tools], "intent": "statistics", "tool_rounds": 0}),
        "tools",
    )
    expect_eq(
        route_after_agent({"messages": [with_tools], "intent": "statistics", "tool_rounds": 999}),
        "wrap_up",
        "轮次耗尽应走收尾节点",
    )
