"""统计计算工具：描述统计、分组聚合、频次分布、透视表。"""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from langchain_core.tools import tool

from ..dataset.profiler import is_numeric, to_jsonable
from .base import (
    fail,
    fmt_num,
    guard,
    markdown_table,
    numeric_columns,
    ok,
    require_numeric,
    resolve_column,
    resolve_columns,
    top_table,
)
from .context import ToolError, get_context

AGG_CHOICES = ("sum", "mean", "count", "median", "max", "min", "std", "nunique")


@tool
@guard("compute_statistics")
def compute_statistics(
    columns: Annotated[list[str] | None, "要统计的数值字段；留空表示所有数值字段"] = None,
    group_by: Annotated[list[str] | None, "分组字段，例如 ['地区'] 或 ['地区','商品类别']；留空表示整体统计"] = None,
    metrics: Annotated[list[str] | None, "聚合方式，可选 sum/mean/count/median/max/min/std/nunique；默认 ['sum','mean','count']"] = None,
    top_n: Annotated[int, "分组统计时返回前多少个分组（按第一个指标降序）"] = 20,
    sort_by: Annotated[str | None, "按哪个字段排序，留空则按第一个统计指标降序"] = None,
    ascending: Annotated[bool, "是否升序排列"] = False,
) -> str:
    """计算统计指标。不分组时输出各数值字段的描述统计（求和/均值/中位数/最大最小/标准差）；给定 group_by 时做分组聚合，例如"各地区销售额"。

    适用场景：用户问"总共多少""平均值是多少""哪个地区/品类最高"等需要具体数字的问题。
    """

    ctx = get_context()
    df = ctx.df
    metrics_list = [m for m in (metrics or ["sum", "mean", "count"]) if m in AGG_CHOICES]
    if not metrics_list:
        return fail("compute_statistics", f"metrics 只能是 {AGG_CHOICES} 中的值。")

    target_cols = resolve_columns(df, columns)
    value_cols = [c for c in (target_cols or [str(c) for c in df.columns]) if is_numeric(df[c])]
    if not value_cols:
        return fail("compute_statistics", "数据集中没有可用于统计计算的数值字段。")
    if target_cols:
        bad = [c for c in target_cols if c not in value_cols]
        if bad:
            return fail("compute_statistics", f"字段 {bad} 不是数值列，无法参与统计计算。")

    # ---------------- 整体描述统计 ----------------
    if not group_by:
        desc = df[value_cols].agg(["count", "sum", "mean", "median", "min", "max", "std"]).T
        desc = desc.round(4)
        frame = desc.reset_index().rename(columns={"index": "字段"})
        for col in frame.columns:
            if col != "字段":
                frame[col] = frame[col].map(fmt_num)
        summary = f"已对 {len(value_cols)} 个数值字段完成描述统计。"
        if value_cols:
            biggest = df[value_cols].sum().idxmax()
            summary += f" 合计最大的字段是「{biggest}」（{fmt_num(df[biggest].sum())}）。"
        return ok(
            "compute_statistics",
            summary,
            {"columns": value_cols, "describe": desc.to_dict(orient="index")},
            table=markdown_table(frame),
        )

    # ---------------- 分组聚合 ----------------
    group_cols = resolve_columns(df, group_by)
    if not group_cols:
        return fail("compute_statistics", "group_by 里的字段名都没匹配上。")

    grouped = df.groupby(group_cols, dropna=False, observed=True)[value_cols].agg(metrics_list)
    grouped.columns = [f"{col}_{m}" for col, m in grouped.columns]
    grouped = grouped.reset_index()

    order_col = None
    if sort_by:
        order_col = resolve_column(grouped, sort_by)
    if order_col is None:
        candidates = [c for c in grouped.columns if c.endswith(f"_{metrics_list[0]}")]
        order_col = candidates[0] if candidates else grouped.columns[-1]
    grouped = grouped.sort_values(order_col, ascending=ascending, kind="stable")

    total_groups = len(grouped)
    shown = grouped.head(max(top_n, 1))

    display = shown.copy()
    for col in display.columns:
        if col not in group_cols:
            display[col] = display[col].map(fmt_num)

    top_row = shown.iloc[0]
    summary = (
        f"按 {'、'.join(group_cols)} 分组，共 {total_groups} 组；"
        f"按 {order_col} {'升序' if ascending else '降序'}排列，"
        f"第一组是「{' / '.join(str(top_row[c]) for c in group_cols)}」"
        f"（{order_col} = {fmt_num(top_row[order_col])}）。"
    )
    return ok(
        "compute_statistics",
        summary,
        {
            "group_by": group_cols,
            "metrics": metrics_list,
            "total_groups": total_groups,
            "order_by": order_col,
            "result": to_jsonable(shown.to_dict(orient="records")),
        },
        table=markdown_table(display),
    )


@tool
@guard("value_counts")
def value_counts(
    column: Annotated[str, "要统计的字段名，例如 '地区' 或 '商品类别'"],
    top_n: Annotated[int, "返回前几个高频取值"] = 15,
    normalize: Annotated[bool, "是否同时输出占比"] = True,
) -> str:
    """统计某个字段的频次分布（每个取值出现多少次、占多少比例）。

    适用场景：用户问"有哪些类别""各类占比多少""最集中的是哪个"。
    """

    ctx = get_context()
    df = ctx.df
    col = resolve_column(df, column)
    series = df[col]
    non_null = max(int(series.notna().sum()), 1)

    vc = series.astype(str).value_counts().head(top_n)
    frame = pd.DataFrame(
        {
            col: vc.index,
            "数量": vc.to_numpy(),
            "占比": [f"{v / non_null:.2%}" for v in vc.to_numpy()],
            "累计占比": [f"{vc.head(i + 1).sum() / non_null:.2%}" for i in range(len(vc))],
        }
    )
    total_unique = int(series.nunique(dropna=True))
    top1 = vc.index[0]
    summary = (
        f"字段「{col}」共 {total_unique} 个不同取值，总量 {non_null}；"
        f"最高频的是「{top1}」（{int(vc.iloc[0])} 次，占 {int(vc.iloc[0]) / non_null:.1%}）。"
    )
    return ok(
        "value_counts",
        summary,
        {
            "column": col,
            "total_unique": total_unique,
            "top_values": [
                {"value": str(k), "count": int(v), "ratio": round(int(v) / non_null, 4)}
                for k, v in vc.items()
            ],
        },
        table=markdown_table(frame),
    )


@tool
@guard("pivot_table")
def pivot_table(
    index: Annotated[str, "行维度字段，例如 '地区'"],
    values: Annotated[str, "要聚合的数值字段，例如 '销售额'"],
    columns: Annotated[str | None, "列维度字段，例如 '销售渠道'；留空则只做行透视"] = None,
    aggfunc: Annotated[str, "聚合函数：sum/mean/count/median/max/min"] = "sum",
    top_n: Annotated[int, "行列维度各取前多少个高频取值"] = 12,
) -> str:
    """生成交叉透视表，例如"各地区 × 各渠道的销售额"。适合回答双重维度对比的问题。

    适用场景：用户同时关心两个维度（地区+渠道、品类+月份）的对比关系时。
    """

    ctx = get_context()
    df = ctx.df
    if aggfunc not in ("sum", "mean", "count", "median", "max", "min"):
        return fail("pivot_table", "aggfunc 只能是 sum/mean/count/median/max/min。")

    idx = resolve_column(df, index)
    val = resolve_column(df, values)
    if aggfunc != "count":
        val = require_numeric(df, val)

    work = df.copy()
    top_index = work[idx].astype(str).value_counts().head(top_n).index.tolist()
    work = work[work[idx].astype(str).isin(top_index)]

    if columns:
        col_dim = resolve_column(df, columns)
        top_cols = work[col_dim].astype(str).value_counts().head(top_n).index.tolist()
        work = work[work[col_dim].astype(str).isin(top_cols)]
        table = pd.pivot_table(
            work, index=idx, columns=col_dim, values=val, aggfunc=aggfunc, observed=True
        ).round(2)
        table = table.fillna(0)
        flat = table.reset_index()
        flat.columns = [" / ".join(str(p) for p in c if p not in ("", None)) if isinstance(c, tuple) else str(c) for c in flat.columns]
        summary = (
            f"已生成「{idx} × {col_dim}」的 {val} {aggfunc} 透视表，"
            f"覆盖 {table.shape[0]} 行 × {table.shape[1]} 列。"
        )
        data: dict[str, Any] = {
            "index": idx,
            "columns": col_dim,
            "values": val,
            "aggfunc": aggfunc,
            "shape": [int(table.shape[0]), int(table.shape[1])],
            "table": to_jsonable(table.to_dict()),
        }
    else:
        table = (
            work.groupby(idx, observed=True)[val].agg(aggfunc).sort_values(ascending=False).round(2)
        )
        flat = table.reset_index()
        flat.columns = [idx, f"{val}_{aggfunc}"]
        best = table.index[0]
        summary = f"已生成按「{idx}」的 {val} {aggfunc} 透视，最高的是「{best}」（{fmt_num(table.iloc[0])}）。"
        data = {
            "index": idx,
            "values": val,
            "aggfunc": aggfunc,
            "table": to_jsonable(table.to_dict()),
        }

    display = flat.copy()
    for col in display.columns:
        if col != idx:
            display[col] = display[col].map(lambda v: fmt_num(v) if isinstance(v, (int, float)) else v)

    return ok("pivot_table", summary, data, table=markdown_table(display))


STATISTICS_TOOLS = [compute_statistics, value_counts, pivot_table]
