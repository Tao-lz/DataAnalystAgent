"""工具公共工具函数：统一返回信封、列名解析、数值与表格渲染。"""

from __future__ import annotations

import json
import math
from functools import wraps
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..dataset.profiler import to_jsonable, to_native
from .context import ToolError, get_context


# --------------------------------------------------------------------------- #
# 返回信封
# --------------------------------------------------------------------------- #
def ok(tool: str, summary: str, data: Any = None, **extra: Any) -> str:
    """成功结果：``summary`` 是一句话结论，``data`` 是结构化细节。"""

    ctx = get_context()
    clean = to_native(data)
    ctx.add_evidence(tool, summary, clean)
    payload = {"ok": True, "tool": tool, "summary": summary, "data": clean}
    payload.update({k: to_native(v) for k, v in extra.items()})
    return json.dumps(payload, ensure_ascii=False)


def fail(tool: str, message: str, **extra: Any) -> str:
    payload = {"ok": False, "tool": tool, "error": message}
    payload.update({k: to_native(v) for k, v in extra.items()})
    return json.dumps(payload, ensure_ascii=False)


def guard(tool: str):
    """把异常转成可读的失败信封，避免整个 Agent 崩掉。"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ToolError as exc:
                return fail(tool, str(exc))
            except Exception as exc:  # noqa: BLE001
                return fail(tool, f"{type(exc).__name__}: {exc}")

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# 列名解析
# --------------------------------------------------------------------------- #
def resolve_column(df: pd.DataFrame, name: str) -> str:
    """宽松匹配列名：精确 -> 忽略大小写/空格 -> 唯一子串。"""

    columns = [str(c) for c in df.columns]
    if name in columns:
        return name
    stripped = name.strip()
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}
    key = stripped.lower().replace(" ", "").replace("_", "")
    if key in normalized:
        return normalized[key]
    hits = [c for c in columns if key and key in c.lower().replace(" ", "").replace("_", "")]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ToolError(f"列名 '{name}' 有歧义，候选：{hits[:8]}")
    raise ToolError(f"找不到列 '{name}'。可用列：{columns[:30]}")


def resolve_columns(df: pd.DataFrame, names: Sequence[str] | None) -> list[str]:
    if not names:
        return []
    return [resolve_column(df, n) for n in names]


def numeric_columns(df: pd.DataFrame, names: Sequence[str] | None = None) -> list[str]:
    cols = resolve_columns(df, names) if names else [str(c) for c in df.columns]
    return [c for c in cols if ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])]


def require_numeric(df: pd.DataFrame, column: str) -> str:
    col = resolve_column(df, column)
    if not (ptypes.is_numeric_dtype(df[col]) and not ptypes.is_bool_dtype(df[col])):
        raise ToolError(f"列 '{col}' 不是数值列（当前类型 {df[col].dtype}），无法做数值计算。")
    return col


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return "-"
    if f != 0 and (abs(f) >= 1e6 or abs(f) < 1e-3):
        return f"{f:.{digits}g}"
    if float(f).is_integer():
        return f"{int(f):,}"
    return f"{f:,.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    """把 DataFrame 渲染成 Markdown 表格（不依赖 tabulate）。"""

    frame = df.reset_index() if index else df
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float):
                cells.append(fmt_num(v))
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("-")
            else:
                text = str(v)
                cells.append(text.replace("|", "/")[:60])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def top_table(series: pd.Series, top_n: int = 10, label: str = "取值") -> str:
    vc = series.astype(str).value_counts().head(top_n)
    frame = pd.DataFrame(
        {
            label: vc.index,
            "数量": vc.to_numpy(),
            "占比": [f"{v / max(int(series.notna().sum()), 1):.2%}" for v in vc.to_numpy()],
        }
    )
    return markdown_table(frame)


def tmp_artifact_path(ctx_suffix: str, ext: str = ".png") -> Any:
    from .context import get_context

    ctx = get_context()
    return ctx.settings.artifact_path / f"{ctx.run_id}-{ctx_suffix}{ext}"


def safe_filename(text: str, max_len: int = 40) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    cleaned = "".join("_" if ch in bad else ch for ch in str(text)).strip()
    return (cleaned or "chart")[:max_len]
