"""LangGraph 各节点实现。

节点分工：
- understand    意图识别 + 生成执行计划（结构化输出）
- chat          闲聊 / 能力咨询，不触发工具
- agent         ReAct 主体：绑定工具，决定调用什么
- tools         执行工具，回收证据与图表产物
- wrap_up       工具轮次耗尽后的强制收尾
- report        把画像 + 证据 + 图表整合成结构化报告
- finalize      收口：确定最终答复文本与状态
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..config import get_settings
from ..core.llm import get_llm
from ..core.store import get_store
from ..report.composer import build_report
from ..tools.context import get_context
from ..tools.registry import TOOLS_BY_NAME, build_tools
from .prompts import (
    ROUTER_SYSTEM,
    agent_system,
    chat_system,
    report_system,
    report_user,
    router_user,
    wrap_up_system,
)
from .state import VALID_INTENTS, AgentState


# --------------------------------------------------------------------------- #
# 结构化输出模型
# --------------------------------------------------------------------------- #
class RouteDecision(BaseModel):
    """意图识别结果。"""

    intent: Literal["chat", "overview", "cleaning", "statistics", "visualization", "report"] = Field(
        description="本轮分析意图"
    )
    plan: list[str] = Field(default_factory=list, description="2-5 步执行计划")
    reason: str = Field(default="", description="判定理由，一句话")


class ReportNarrative(BaseModel):
    """报告正文。"""

    title: str = Field(description="报告标题")
    overview: str = Field(description="数据概览，2-4 句")
    quality: str = Field(description="数据质量评价，2-4 句，含具体数字")
    findings: list[str] = Field(default_factory=list, description="3-6 条关键发现，每条含具体数字")
    recommendations: list[str] = Field(default_factory=list, description="2-4 条可执行建议")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _text_of(message: Any) -> str:
    """兼容 str / 内容块列表两种 message.content 形态。"""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts)
    return "" if content is None else str(content)


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("没有找到 JSON 对象")
    return json.loads(text[start : end + 1])


def _trim_history(messages: list[Any], window: int) -> list[Any]:
    """按条数截断历史，并且只在"用户提问"处切开，避免切断 tool_calls 配对。"""

    if len(messages) <= window:
        return list(messages)
    start = len(messages) - window
    while start > 0 and not isinstance(messages[start], HumanMessage):
        start -= 1
    return list(messages[start:])


def _structured(messages: list[Any], schema: type[BaseModel], default: BaseModel) -> BaseModel:
    """结构化输出。

    主路径走 ``function_calling``（DeepSeek 不支持 ``response_format=json_schema``，
    直接调用 with_structured_output 会返回 400）；主路径失败时退化为
    "要求模型只输出 JSON"的纯文本兜底。
    """

    method = get_settings().LLM_STRUCTURED_METHOD
    try:
        llm = get_llm(temperature=0)
        return llm.with_structured_output(schema, method=method).invoke(messages)
    except TypeError:  # 老版本 langchain 没有 method 参数
        try:
            return get_llm(temperature=0).with_structured_output(schema).invoke(messages)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    try:
        llm = get_llm(temperature=0)
        fallback_messages = [
            *messages,
            HumanMessage(content="请只输出一个 JSON 对象，字段与要求一致，不要任何解释文字。"),
        ]
        raw = llm.invoke(fallback_messages)
        data = _extract_json(_text_of(raw))
        return schema(**{**default.model_dump(), **data})
    except Exception:  # noqa: BLE001
        return default


DEFAULT_PLANS: dict[str, list[str]] = {
    "overview": [
        "调用 dataset_overview 掌握整体规模、字段清单与质量评分",
        "针对重点字段说明语义类型、缺失情况与取值范围",
        "给出这个数据集适合做哪些分析的建议",
    ],
    "cleaning": [
        "用 detect_missing_values / detect_outliers 定位具体问题",
        "选择合适的清洗动作并执行，记录改动前后差异",
        "复查清洗后的数据质量是否改善",
    ],
    "statistics": [
        "先确认字段名与类型",
        "计算总体指标与分组对比",
        "对异常或反直觉的结果做二次验证（如筛选明细、检查离群点）",
    ],
    "visualization": [
        "确认要画的字段与维度",
        "选择合适的图表类型并生成图表",
        "生成后计算关键极值，用文字点出图表里的重点",
    ],
    "report": [
        "调用 dataset_overview 与 detect_missing_values 摸清数据现状",
        "计算关键指标的整体统计与分组对比",
        "绘制 2-3 张关键图表（趋势、对比、分布）",
        "汇总为结构化分析报告",
    ],
}


def _merge_plan(intent: str, plan: list[str]) -> list[str]:
    plan = [p.strip() for p in (plan or []) if str(p).strip()]
    if not plan:
        plan = list(DEFAULT_PLANS.get(intent, DEFAULT_PLANS["statistics"]))
    if intent in ("report", "visualization") and not any(
        kw in "".join(plan) for kw in ("图", "plot_", "可视")
    ):
        plan.append("绘制关键图表（折线图/柱状图/直方图）")
    return plan[:6]


# --------------------------------------------------------------------------- #
# 节点 1：意图识别 + 规划
# --------------------------------------------------------------------------- #
def understand_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    ctx = get_context()
    question = state.get("question") or ""
    brief = ctx.brief

    default = RouteDecision(
        intent="statistics",
        plan=list(DEFAULT_PLANS["statistics"]),
        reason="默认按统计分析处理",
    )
    try:
        decision = _structured(
            [SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=router_user(question, brief))],
            RouteDecision,
            default,
        )
        intent = decision.intent if decision.intent in VALID_INTENTS else "statistics"
        plan = _merge_plan(intent, list(decision.plan))
        reason = decision.reason or ""
    except Exception as exc:  # noqa: BLE001
        intent, plan, reason = "statistics", _merge_plan("statistics", []), f"路由失败，回退到综合分析：{exc}"

    return {
        "intent": intent,
        "plan": plan,
        "route_reason": reason,
        # 每轮重置"本轮产物"
        "evidence": [],
        "artifacts": [],
        "tool_rounds": 0,
        "answer": "",
        "report_id": None,
        "error": None,
        "status": "planned",
    }


# --------------------------------------------------------------------------- #
# 节点 2：闲聊
# --------------------------------------------------------------------------- #
def chat_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    ctx = get_context()
    messages = [
        SystemMessage(content=chat_system(ctx.brief)),
        *_trim_history(state.get("messages") or [], get_settings().HISTORY_WINDOW),
    ]
    ai = get_llm().invoke(messages)
    text = _text_of(ai)
    return {"messages": [ai], "answer": text, "status": "answered"}


# --------------------------------------------------------------------------- #
# 节点 3：ReAct 主体
# --------------------------------------------------------------------------- #
def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    ctx = get_context()
    intent = str(state.get("intent") or "statistics")
    tools = build_tools(intent)

    llm = get_llm().bind_tools(tools) if tools else get_llm()
    messages = [
        SystemMessage(content=agent_system(ctx.brief, intent, list(state.get("plan") or []))),
        *_trim_history(state.get("messages") or [], get_settings().HISTORY_WINDOW),
    ]
    ai = llm.invoke(messages)

    update: dict[str, Any] = {
        "messages": [ai],
        "tool_rounds": int(state.get("tool_rounds") or 0),
        "status": "acting" if getattr(ai, "tool_calls", None) else "answered",
    }
    if not getattr(ai, "tool_calls", None):
        update["answer"] = _text_of(ai)
    return update


# --------------------------------------------------------------------------- #
# 节点 4：工具执行
# --------------------------------------------------------------------------- #
def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    ctx = get_context()
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    calls = list(getattr(last, "tool_calls", None) or [])

    outputs: list[ToolMessage] = []
    for call in calls:
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        call_id = str(call.get("id") or uuid.uuid4().hex)
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            content = json.dumps(
                {"ok": False, "tool": name, "error": f"没有名为 {name} 的工具，请从可用工具中选择。"},
                ensure_ascii=False,
            )
        else:
            try:
                result = tool.invoke(args)
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                content = json.dumps(
                    {
                        "ok": False,
                        "tool": name,
                        "error": f"参数不正确或执行失败：{type(exc).__name__}: {exc}",
                        "hint": "请检查字段名与参数类型后重试。",
                    },
                    ensure_ascii=False,
                )
        outputs.append(ToolMessage(content=content, tool_call_id=call_id, name=name or None))

    return {
        "messages": outputs,
        "tool_rounds": int(state.get("tool_rounds") or 0) + 1,
        "evidence": list(ctx.evidence),
        "artifacts": list(ctx.artifacts),
        "status": "tool_done",
    }


# --------------------------------------------------------------------------- #
# 节点 5：强制收尾
# --------------------------------------------------------------------------- #
def wrap_up_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    ctx = get_context()
    messages = [
        SystemMessage(content=wrap_up_system(ctx.brief)),
        *_trim_history(state.get("messages") or [], get_settings().HISTORY_WINDOW),
    ]
    ai = get_llm().invoke(messages)
    return {"messages": [ai], "answer": _text_of(ai), "status": "wrapped"}


# --------------------------------------------------------------------------- #
# 节点 6：结构化报告
# --------------------------------------------------------------------------- #
def report_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    ctx = get_context()
    settings = get_settings()
    store = get_store()

    brief = ctx.brief
    meta = ctx.meta
    profile = meta.get("profile") or {}
    evidence = list(state.get("evidence") or ctx.evidence)
    artifacts = list(state.get("artifacts") or ctx.artifacts)

    default = ReportNarrative(
        title=f"{ctx.name} 数据分析报告",
        overview=f"本报告基于数据集「{ctx.name}」，共 {brief.get('rows')} 行 × {brief.get('cols')} 列。",
        quality=f"数据质量评分 {profile.get('quality', {}).get('score', '-')} / 100。",
        findings=[e.get("summary", "") for e in evidence[:5] if e.get("summary")],
        recommendations=["结合业务口径复核关键结论，并对缺失与异常数据做进一步确认。"],
    )
    narrative = _structured(
        [
            SystemMessage(content=report_system(brief)),
            HumanMessage(content=report_user(brief, evidence, artifacts)),
        ],
        ReportNarrative,
        default,
    )
    if not narrative.findings:
        narrative.findings = default.findings
    if not narrative.recommendations:
        narrative.recommendations = default.recommendations

    payload = build_report(
        dataset=meta,
        profile=profile,
        narrative=narrative.model_dump(),
        evidence=evidence,
        artifacts=artifacts,
    )

    report_id = f"R{uuid.uuid4().hex[:12]}"
    md_path = settings.report_path / f"{report_id}.md"
    html_path = settings.report_path / f"{report_id}.html"
    md_path.write_text(payload["markdown"], encoding="utf-8")
    html_path.write_text(payload["html"], encoding="utf-8")

    store.save_report(
        report_id=report_id,
        dataset_id=ctx.dataset_id,
        conversation_id=ctx.conversation_id,
        title=payload["title"],
        markdown=payload["markdown"],
        html=payload["html"],
        markdown_path=md_path,
        html_path=html_path,
        payload=payload,
    )

    findings_preview = "；".join(narrative.findings[:2]) if narrative.findings else ""
    answer = (
        f"已生成结构化分析报告《{payload['title']}》，共 5 个章节、"
        f"{len(artifacts)} 张图表、{len(narrative.findings)} 条关键发现 "
        f"（report_id: {report_id}）。"
    )
    if findings_preview:
        answer += f"\n\n核心结论：{findings_preview}"

    return {
        "report_id": report_id,
        "answer": answer,
        "status": "reported",
        "artifacts": artifacts,
        "evidence": evidence,
    }


# --------------------------------------------------------------------------- #
# 节点 7：收口
# --------------------------------------------------------------------------- #
def finalize_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    answer = str(state.get("answer") or "").strip()
    if not answer:
        for message in reversed(state.get("messages") or []):
            if isinstance(message, AIMessage):
                text = _text_of(message).strip()
                if text:
                    answer = text
                    break
    if not answer:
        answer = "本轮没有产生文字结论。可以换个更具体的问法，例如「各地区的销售额对比」。"
    return {"answer": answer, "status": "completed"}


# --------------------------------------------------------------------------- #
# 条件路由
# --------------------------------------------------------------------------- #
def route_after_understand(state: AgentState) -> str:
    return "chat" if state.get("intent") == "chat" else "agent"


def route_after_agent(state: AgentState) -> str:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        limit = get_settings().MAX_TOOL_ROUNDS
        return "tools" if int(state.get("tool_rounds") or 0) < limit else "wrap_up"
    if state.get("intent") == "report":
        return "report"
    return "finalize"
