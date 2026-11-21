"""模型评测入口：用固定用例集衡量 Agent 的意图、工具选择与数字准确性。

用法：
    python evals/run_evals.py                     # 跑全部 20 条
    python evals/run_evals.py --only viz          # 只跑可视化
    python evals/run_evals.py --limit 3           # 先跑 3 条试水
    python evals/run_evals.py --repeat 3          # 每条跑 3 次，降低随机影响
    python evals/run_evals.py --fresh             # 清空评测专用存储后重跑

评测使用独立的 storage（evals/.storage），不会污染日常使用的数据。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
CASES_FILE = EVAL_DIR / "cases.jsonl"
REPORT_DIR = EVAL_DIR / "reports"


def _preparse_storage() -> Path:
    """必须在导入 backend 之前确定存储目录。"""

    for i, arg in enumerate(sys.argv):
        if arg == "--storage" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
        if arg.startswith("--storage="):
            return Path(arg.split("=", 1)[1])
    return EVAL_DIR / ".storage"


STORAGE = _preparse_storage().resolve()
import os  # noqa: E402

os.environ["STORAGE_DIR"] = str(STORAGE)
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

from backend.app.config import get_settings  # noqa: E402
from backend.app.core.store import get_store  # noqa: E402
from backend.app.dataset.loader import load_dataframe  # noqa: E402
from backend.app.dataset.profiler import profile_dataframe  # noqa: E402
from backend.app.graph.builder import get_graph  # noqa: E402
from backend.app.graph.runner import conversation_history, run_agent  # noqa: E402

DATASET_ID = "Deval00000001"
DATASET_NAME = "评测样本_电商订单"
DEFAULT_DATASET = ROOT / "data" / "samples" / "电商订单_示例数据.csv"

NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万|亿|千)?")
UNITS = {"万": 1e4, "亿": 1e8, "千": 1e3}


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def ensure_dataset(source: Path) -> dict[str, Any]:
    store = get_store()
    row = store.get_dataset(DATASET_ID)
    if row and Path(row["working_path"]).exists():
        print(f"复用评测数据集：{DATASET_ID}")
        return row

    df, _ = load_dataframe(source)
    settings = get_settings()
    working = settings.upload_path / f"{DATASET_ID}_v0.csv"
    df.to_csv(working, index=False, encoding="utf-8-sig")
    profile = profile_dataframe(df, name=DATASET_NAME)
    store.save_dataset(
        dataset_id=DATASET_ID,
        name=DATASET_NAME,
        filename=source.name,
        original_path=source,
        working_path=working,
        sheet=None,
        encoding="utf-8",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        size_bytes=source.stat().st_size,
        profile=profile,
    )
    print(f"已建立评测数据集：{DATASET_ID}（{df.shape[0]} 行 × {df.shape[1]} 列）")
    return store.get_dataset(DATASET_ID) or {}


def baseline_reset() -> None:
    """每条用例开跑前把工作副本复位，避免上一条的清洗影响下一条。"""

    store = get_store()
    row = store.get_dataset(DATASET_ID)
    if not row:
        return
    v0 = get_settings().upload_path / f"{DATASET_ID}_v0.csv"
    store.update_working_copy(
        DATASET_ID,
        working_path=v0,
        rows=int((row.get("profile") or {}).get("rows") or row.get("rows") or 0),
        cols=int((row.get("profile") or {}).get("cols") or row.get("cols") or 0),
        profile=row.get("profile") or {},
    )


# --------------------------------------------------------------------------- #
# 执行与打分
# --------------------------------------------------------------------------- #
def run_once(case: dict[str, Any]) -> dict[str, Any]:
    baseline_reset()
    started = time.perf_counter()
    result = run_agent(DATASET_ID, case["question"])
    elapsed = time.perf_counter() - started

    history = conversation_history(result["conversation_id"])
    calls: list[dict[str, Any]] = []
    for message in history:
        if message.get("role") == "assistant":
            calls.extend(message.get("tool_calls") or [])

    return {
        "conversation_id": result["conversation_id"],
        "intent": result.get("intent"),
        "answer": result.get("answer") or "",
        "tool_calls": calls,
        "rounded_rounds": result.get("tool_rounds") or 0,
        "report_id": result.get("report_id"),
        "latency": round(elapsed, 2),
        "tokens": _token_usage(result["conversation_id"]),
        "status": result.get("status"),
    }


def _token_usage(conversation_id: str) -> int:
    try:
        snapshot = get_graph().get_state({"configurable": {"thread_id": conversation_id}})
        messages = (getattr(snapshot, "values", None) or {}).get("messages") or []
        total = 0
        for message in messages:
            if isinstance(message, AIMessage):
                usage = getattr(message, "usage_metadata", None) or {}
                total += int(usage.get("total_tokens") or 0)
        return total
    except Exception:  # noqa: BLE001
        return 0


def extract_numbers(text: str) -> list[float]:
    # 去掉千分位逗号，否则 2,260,964.83 会被拆成三个数
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", str(text))
    values: list[float] = []
    for match in NUM_RE.finditer(cleaned):
        value = float(match.group(1)) * UNITS.get(match.group(2) or "", 1.0)
        values.append(value)
    return values


def number_hit(expected: float, candidates: list[float], rel_tol: float = 0.01) -> bool:
    for value in candidates:
        if expected == 0:
            if abs(value) < 1e-9:
                return True
        elif abs(value - expected) / abs(expected) <= rel_tol:
            return True
    return False


def _arg_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (list, tuple)):
        return str(expected).strip() in [str(x).strip() for x in actual]
    if isinstance(expected, (list, tuple)):
        return str(actual).strip() in [str(x).strip() for x in expected]
    return str(actual).strip() == str(expected).strip()


def check_args(case: dict[str, Any], calls: list[dict[str, Any]]) -> bool:
    expected = case.get("expected_args") or {}
    for tool, params in expected.items():
        matched = False
        for call in calls:
            if call.get("name") != tool:
                continue
            args = call.get("args") or {}
            if all(_arg_matches(args.get(key), value) for key, value in params.items()):
                matched = True
                break
        if not matched:
            return False
    return True


def resolve_allowed(case: dict[str, Any], called: set[str]) -> set[str]:
    """允许集合：优先用 allowed_group（该意图的整组工具），其次 allowed_tools，否则不限制。"""

    group = case.get("allowed_group")
    if group:
        from backend.app.tools.registry import tools_for_intent

        return {tool.name for tool in tools_for_intent(group)}
    allowed = case.get("allowed_tools")
    return set(allowed) if allowed is not None else called


def score(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    called = [str(c.get("name")) for c in run["tool_calls"]]
    called_set = set(called)
    required = set(case.get("required_tools") or [])
    allowed_set = resolve_allowed(case, called_set)

    missing = sorted(required - called_set)
    recall = 1.0 if not required else (len(required) - len(missing)) / len(required)
    precision = 1.0 if not called else len([c for c in called if c in allowed_set]) / len(called)

    intent_ok = run["intent"] in (case.get("intents") or [])
    no_tools_ok = (len(called) == 0) if case.get("no_tools") else True
    args_ok = check_args(case, run["tool_calls"])

    candidates = extract_numbers(run["answer"])
    expected_numbers = [float(n) for n in (case.get("numbers") or [])]
    number_detail = [
        {"expected": n, "hit": number_hit(n, candidates)} for n in expected_numbers
    ]
    numbers_ok = all(item["hit"] for item in number_detail)

    keywords = case.get("expect_keywords") or []
    keywords_ok = (not keywords) or any(k in run["answer"] for k in keywords)
    report_ok = bool(run.get("report_id")) if case.get("expect_report") else True

    passed = all(
        [intent_ok, not missing, args_ok, numbers_ok, keywords_ok, report_ok, no_tools_ok]
    )
    return {
        "passed": passed,
        "intent_ok": intent_ok,
        "tool_recall": recall,
        "tool_precision": precision,
        "args_ok": args_ok,
        "numbers_ok": numbers_ok,
        "number_detail": number_detail,
        "keywords_ok": keywords_ok,
        "report_ok": report_ok,
        "no_tools_ok": no_tools_ok,
        "missing_tools": missing,
        "called_tools": called,
    }


# --------------------------------------------------------------------------- #
# 用例集与汇总
# --------------------------------------------------------------------------- #
def load_cases(only: str | None, limit: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in CASES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    if only:
        key = only.lower()
        cases = [
            c for c in cases
            if str(c.get("tag", "")).lower().startswith(key)
            or str(c.get("id", "")).lower().startswith(key)
        ]
    if limit:
        cases = cases[:limit]
    return cases


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return round(statistics.mean([r["score"][key] for r in results]), 4) if results else 0.0

    number_cases = [r for r in results if r["score"]["number_detail"]]
    arg_cases = [r for r in results if r["case"].get("expected_args")]
    return {
        "cases": len(results),
        "pass_rate": mean("passed"),
        "intent_accuracy": mean("intent_ok"),
        "tool_recall": mean("tool_recall"),
        "tool_precision": mean("tool_precision"),
        "args_accuracy": round(
            statistics.mean([r["score"]["args_ok"] for r in arg_cases]), 4
        ) if arg_cases else None,
        "number_hit_rate": round(
            statistics.mean(
                [
                    (sum(1 for d in r["score"]["number_detail"] if d["hit"]) / len(r["score"]["number_detail"]))
                    for r in number_cases
                ]
            ), 4
        ) if number_cases else None,
        "avg_rounds": round(statistics.mean([r["run"]["rounded_rounds"] for r in results]), 2) if results else 0,
        "avg_latency": round(statistics.mean([r["run"]["latency"] for r in results]), 2) if results else 0,
        "avg_tokens": int(statistics.mean([r["run"]["tokens"] for r in results])) if results else 0,
        "max_tool_calls": max(
            (len(r["run"]["tool_calls"]) for r in results), default=0
        ),
    }


def render_markdown(meta: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    lines = [
        "# 模型评测报告",
        "",
        f"- 时间：{meta['time']}",
        f"- 模型：{meta['model']}",
        f"- 数据集：{meta['dataset']}",
        f"- 用例数：{summary['cases']}（每条重复 {meta['repeat']} 次）",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 用例通过率 | {pct(summary['pass_rate'])} |",
        f"| 意图准确率 | {pct(summary['intent_accuracy'])} |",
        f"| 工具召回率 | {pct(summary['tool_recall'])} |",
        f"| 工具精确率 | {pct(summary['tool_precision'])} |",
        f"| 参数准确率 | {pct(summary['args_accuracy'])} |",
        f"| 关键数字命中率 | {pct(summary['number_hit_rate'])} |",
        f"| 平均工具轮次 | {summary['avg_rounds']} |",
        f"| 平均延迟 | {summary['avg_latency']} s |",
        f"| 平均 token | {summary['avg_tokens']} |",
        "",
        "## 逐条结果",
        "",
        "| 用例 | 意图 | 调用次数 | 实际工具 | 结果 | 未命中 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        case, run, score_ = item["case"], item["run"], item["score"]
        tools = ", ".join(score_["called_tools"]) or "（未调用）"
        reasons = []
        if not score_["intent_ok"]:
            reasons.append(f"意图={run['intent']}")
        if score_["missing_tools"]:
            reasons.append("缺工具:" + ",".join(score_["missing_tools"]))
        if not score_["args_ok"]:
            reasons.append("参数不对")
        if not score_["numbers_ok"]:
            missed = [str(d["expected"]) for d in score_["number_detail"] if not d["hit"]]
            reasons.append("缺数字:" + ",".join(missed))
        if not score_["keywords_ok"]:
            reasons.append("未表达不确定性")
        if not score_["report_ok"]:
            reasons.append("未生成报告")
        verdict = "✅ 通过" if score_["passed"] else "❌ 失败"
        lines.append(
            f"| `{case['id']}` | {run['intent']} | {len(run['tool_calls'])} | {tools} | {verdict} | {'；'.join(reasons) or '-'} |"
        )
    lines += ["", "## 失败用例的回答片段", ""]
    failures = [r for r in results if not r["score"]["passed"]]
    if not failures:
        lines.append("无。")
    for item in failures:
        lines.append(f"### `{item['case']['id']}`")
        lines.append("")
        lines.append(f"> {item['case']['question']}")
        lines.append("")
        lines.append("```text")
        lines.append((item["run"]["answer"] or "")[:800])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="DataAnalysisAgent 模型评测")
    parser.add_argument("--only", help="只跑某个 tag 或 id 前缀，例如 viz / stat / report")
    parser.add_argument("--limit", type=int, help="只跑前 N 条")
    parser.add_argument("--repeat", type=int, default=1, help="每条用例重复次数（默认 1）")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="评测用 CSV/Excel")
    parser.add_argument("--storage", default=str(EVAL_DIR / ".storage"), help="评测专用存储目录")
    parser.add_argument("--fresh", action="store_true", help="清空评测存储后重跑")
    args = parser.parse_args()

    if args.fresh and STORAGE.exists():
        shutil.rmtree(STORAGE, ignore_errors=True)
        print(f"已清空评测存储：{STORAGE}")

    settings = get_settings()
    print("=" * 74)
    print(f"模型：{settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
    print(f"评测存储：{STORAGE}")
    print("=" * 74)

    ensure_dataset(Path(args.dataset))
    cases = load_cases(args.only, args.limit)
    if not cases:
        print("没有匹配的用例。")
        return 1

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] {case['id']}  {case['question']}")
        runs: list[dict[str, Any]] = []
        scores: list[dict[str, Any]] = []
        for attempt in range(max(1, args.repeat)):
            run = run_once(case)
            score_ = score(case, run)
            runs.append(run)
            scores.append(score_)
            mark = "✓" if score_["passed"] else "✗"
            print(
                f"    {mark} intent={run['intent']:<14} tools={','.join(score_['called_tools']) or '-':<46}"
                f" {run['latency']:5.1f}s"
            )
            if score_["missing_tools"]:
                print(f"      缺少工具：{score_['missing_tools']}")
            if not score_["numbers_ok"]:
                missed = [d["expected"] for d in score_["number_detail"] if not d["hit"]]
                print(f"      未命中数字：{missed}")

        passed_count = sum(1 for s in scores if s["passed"])
        merged = {
            key: round(statistics.mean([s[key] for s in scores]), 4)
            for key in ("tool_recall", "tool_precision")
        }
        merged["passed"] = passed_count * 2 >= len(scores)
        merged["intent_ok"] = statistics.mean([s["intent_ok"] for s in scores]) >= 0.5
        merged["args_ok"] = all(s["args_ok"] for s in scores)
        merged["numbers_ok"] = all(s["numbers_ok"] for s in scores)
        merged["keywords_ok"] = all(s["keywords_ok"] for s in scores)
        merged["report_ok"] = all(s["report_ok"] for s in scores)
        merged["no_tools_ok"] = all(s["no_tools_ok"] for s in scores)
        merged["number_detail"] = [
            {"expected": d["expected"], "hit": all(s["numbers_ok"] for s in scores)}
            for d in scores[0]["number_detail"]
        ]
        merged["missing_tools"] = scores[0]["missing_tools"]
        merged["called_tools"] = sorted({t for s in scores for t in s["called_tools"]})

        results.append(
            {
                "case": case,
                "run": runs[0],
                "runs": runs,
                "score": merged,
                "pass_count": passed_count,
            }
        )

    summary = summarize(results)
    meta = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": settings.LLM_MODEL,
        "dataset": Path(args.dataset).name,
        "repeat": args.repeat,
        "storage": str(STORAGE),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORT_DIR / f"eval-{stamp}.json"
    md_path = REPORT_DIR / f"eval-{stamp}.md"
    json_path.write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(meta, summary, results), encoding="utf-8")

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    print("\n" + "=" * 74)
    print(f"用例通过率      {pct(summary['pass_rate'])}  ({sum(1 for r in results if r['score']['passed'])}/{summary['cases']})")
    print(f"意图准确率      {pct(summary['intent_accuracy'])}")
    print(f"工具召回率      {pct(summary['tool_recall'])}")
    print(f"工具精确率      {pct(summary['tool_precision'])}")
    print(f"参数准确率      {pct(summary['args_accuracy'])}")
    print(f"关键数字命中率  {pct(summary['number_hit_rate'])}")
    print(f"平均轮次 / 延迟 / token   {summary['avg_rounds']} / {summary['avg_latency']}s / {summary['avg_tokens']}")
    print(f"单条最多工具调用  {summary['max_tool_calls']} 次")
    print("=" * 74)
    print(f"详细报告：{md_path}")
    print(f"原始数据：{json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
