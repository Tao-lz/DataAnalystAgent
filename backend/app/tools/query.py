"""数据筛选与排序工具。"""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from langchain_core.tools import tool

from ..dataset.profiler import is_numeric, to_jsonable
from .base import fail, fmt_num, guard, markdown_table, ok, resolve_column, resolve_columns
from .context import ToolError, get_context

OPS = (
    "==", "!=", ">", ">=", "<", "<=",
    "contains", "startswith", "endswith",
    "in", "not_in", "between", "is_null", "not_null",
)


def _coerce(value: Any, series: pd.Series) -> Any:
    """把字符串参数转成与列同类型的值。"""

    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if is_numeric(series):
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return value
    return value


def _mask_for(series: pd.Series, op: str, value: Any) -> pd.Series:
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ToolError("between 需要 value=[下界, 上界] 两个值。")
        low, high = _coerce(value[0], series), _coerce(value[1], series)
        return series.between(low, high)
    if op in ("in", "not_in"):
        if not isinstance(value, (list, tuple)):
            value = [v.strip() for v in str(value).split(",")]
        values = [_coerce(v, series) for v in value]
        mask = series.isin(values)
        return mask if op == "in" else ~mask

    scalar = _coerce(value, series)
    if op == "==":
        return series == scalar
    if op == "!=":
        return series != scalar
    if op in (">", ">=", "<", "<="):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            left: pd.Series = numeric
            scalar = _coerce(value, pd.Series([1.0]))
        else:
            left = series.astype(str)
        if op == ">":
            return left > scalar
        if op == ">=":
            return left >= scalar
        if op == "<":
            return left < scalar
        return left <= scalar
    if op == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    if op == "startswith":
        return series.astype(str).str.startswith(str(value), na=False)
    if op == "endswith":
        return series.astype(str).str.endswith(str(value), na=False)
    raise ToolError(f"不支持的操作符 {op}，可用：{OPS}")


@tool
@guard("filter_rows")
def filter_rows(
    filters: Annotated[
        list[dict],
        "筛选条件列表，每项形如 {'column': '地区', 'op': '==', 'value': '华东'}；"
        "多个条件之间是「且」的关系；op 支持 ==、!=、>、>=、<、<=、contains、startswith、"
        "endswith、in、not_in、between、is_null、not_null",
    ],
    columns: Annotated[list[str] | None, "只返回这些字段；留空表示返回全部字段"] = None,
    sort_by: Annotated[str | None, "排序字段"] = None,
    ascending: Annotated[bool, "是否升序"] = False,
    limit: Annotated[int, "最多返回多少行（也用于『取前 N 名』这类需求）"] = 20,
) -> str:
    """按条件筛选数据行，可排序并只取前 N 行。这是回答"某类具体记录长什么样""销售额最高的前 10 笔订单"的主力工具。

    适用场景：需要下钻到明细、按条件圈定子集、或者想找极值记录时。
    """

    ctx = get_context()
    df = ctx.df
    if not filters:
        return fail("filter_rows", "请至少给出一个筛选条件。")

    mask = pd.Series(True, index=df.index)
    applied: list[dict[str, Any]] = []
    for cond in filters:
        if not isinstance(cond, dict) or "column" not in cond or "op" not in cond:
            return fail("filter_rows", "每个条件必须是 {'column':..., 'op':..., 'value':...} 形式。")
        col = resolve_column(df, str(cond["column"]))
        op = str(cond["op"]).strip()
        value = cond.get("value")
        try:
            cond_mask = _mask_for(df[col], op, value)
        except ToolError as exc:
            return fail("filter_rows", str(exc))
        mask &= cond_mask.fillna(False)
        applied.append({"column": col, "op": op, "value": to_jsonable(value)})

    result = df[mask]
    total_matched = int(result.shape[0])
    if total_matched == 0:
        return ok(
            "filter_rows",
            f"按 {len(applied)} 个条件筛选后没有匹配到任何记录，建议放宽条件。",
            {"filters": applied, "matched": 0},
        )

    if sort_by:
        sort_col = resolve_column(df, sort_by)
        result = result.sort_values(sort_col, ascending=ascending, kind="stable")

    selected = resolve_columns(df, columns)
    view = result[selected] if selected else result
    shown = view.head(max(limit, 1))

    display = shown.copy()
    for col in display.columns:
        display[col] = display[col].map(
            lambda v: fmt_num(v) if isinstance(v, (int, float)) else ("" if pd.isna(v) else str(v)[:40])
        )

    cond_text = " 且 ".join(
        f"{c['column']} {c['op']} {c['value']}" for c in applied
    )
    summary = (
        f"筛选条件【{cond_text}】命中 {total_matched} 条记录（占全表 {total_matched / max(len(df), 1):.1%}）"
    )
    if sort_by:
        summary += f"，已按 {sort_by} {'升序' if ascending else '降序'}排列"
    summary += f"，展示前 {len(shown)} 条。"

    return ok(
        "filter_rows",
        summary,
        {
            "filters": applied,
            "matched": total_matched,
            "total_rows": int(len(df)),
            "returned": int(len(shown)),
            "sort_by": sort_by,
            "order": "asc" if ascending else "desc",
            "rows": to_jsonable(shown.to_dict(orient="records")),
        },
        table=markdown_table(display),
    )


QUERY_TOOLS = [filter_rows]
