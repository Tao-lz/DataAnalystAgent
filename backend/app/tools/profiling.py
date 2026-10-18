"""数据概览与质量诊断工具。

这一组是"看病"工具：先看清数据集长什么样、哪里有问题，再决定怎么治。
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pandas.api import types as ptypes

from ..dataset.profiler import (
    PLACEHOLDER_TOKENS,
    detect_placeholder_missing,
    is_datetime,
    is_numeric,
    profile_dataframe,
    to_jsonable,
)
from .base import fail, fmt_num, guard, markdown_table, numeric_columns, ok, require_numeric, resolve_column, top_table
from .context import ToolError, get_context


@tool
@guard("dataset_overview")
def dataset_overview() -> str:
    """查看当前数据集的整体概览：行列规模、字段清单（名称/语义类型/缺失数/唯一值数）、数据质量评分和问题清单。

    这是所有分析的起点，建议在回答任何问题之前先调用一次，用来确认字段名和数据类型。
    """

    ctx = get_context()
    df = ctx.df
    profile = profile_dataframe(df, name=ctx.name)
    quality = profile["quality"]

    rows = [
        {
            "字段": c["name"],
            "类型": c["semantic_type"],
            "缺失": c["missing"],
            "缺失率": f"{c['missing_ratio']:.1%}",
            "唯一值": c["unique"],
            "示例": ", ".join(str(v) for v in c["sample_values"][:3]),
        }
        for c in profile["columns"]
    ]

    summary = (
        f"数据集「{profile['name']}」共 {profile['rows']} 行 × {profile['cols']} 列，"
        f"质量评分 {quality['score']}/100，发现 {len(quality['issues'])} 类数据问题，"
        f"其中数值字段 {len(profile['column_groups']['numeric'])} 个、"
        f"日期字段 {len(profile['column_groups']['datetime'])} 个。"
    )
    return ok(
        "dataset_overview",
        summary,
        {
            "rows": profile["rows"],
            "cols": profile["cols"],
            "memory_mb": profile["memory_mb"],
            "column_groups": profile["column_groups"],
            "quality": quality,
            "columns": [
                {
                    "name": c["name"],
                    "dtype": c["dtype"],
                    "semantic_type": c["semantic_type"],
                    "missing": c["missing"],
                    "missing_ratio": c["missing_ratio"],
                    "unique": c["unique"],
                    "sample_values": c["sample_values"],
                }
                for c in profile["columns"]
            ],
        },
        table=markdown_table(pd.DataFrame(rows)),
    )


@tool
@guard("detect_missing_values")
def detect_missing_values(
    threshold: Annotated[float, "只报告缺失率大于等于该比例的字段，0~1，默认 0 表示全部字段"] = 0.0,
) -> str:
    """检测缺失值：逐字段给出缺失数量与占比，统计受影响行数，并识别 '?'、'N/A'、'未知'、'-' 这类没有被当成缺失的伪空值。

    适用场景：用户提到"缺失""空值""数据质量"，或准备清洗数据之前。
    """

    ctx = get_context()
    df = ctx.df
    n = max(len(df), 1)

    rows: list[dict[str, Any]] = []
    placeholders: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        ph = detect_placeholder_missing(series)
        if missing / n >= threshold or ph:
            rows.append(
                {
                    "字段": str(col),
                    "缺失数": missing,
                    "缺失率": f"{missing / n:.2%}",
                    "伪空值": ph,
                    "有效值": int(series.notna().sum()),
                }
            )
        if ph:
            tokens = series.dropna().astype(str).str.strip()
            bad = tokens[tokens.str.lower().isin(PLACEHOLDER_TOKENS)].value_counts().head(5)
            placeholders.append(
                {"column": str(col), "count": ph, "tokens": {str(k): int(v) for k, v in bad.items()}}
            )

    total_missing = int(df.isna().sum().sum())
    affected_rows = int(df.isna().any(axis=1).sum())
    complete_rows = int((~df.isna().any(axis=1)).sum())

    if not rows:
        summary = f"数据很干净：{len(df)} 行数据中没有任何缺失值。"
    else:
        summary = (
            f"共 {total_missing} 个缺失单元格，分布在 {affected_rows} 行（占 {affected_rows / n:.1%}），"
            f"完整行 {complete_rows} 行；涉及 {len(rows)} 个字段，"
            f"其中 {sum(1 for r in rows if r['缺失数'] > 0)} 个字段存在真实缺失、"
            f"{len(placeholders)} 个字段含伪空值。"
        )

    return ok(
        "detect_missing_values",
        summary,
        {
            "total_missing": total_missing,
            "affected_rows": affected_rows,
            "complete_rows": complete_rows,
            "missing_columns": rows,
            "placeholder_columns": placeholders,
        },
        table=markdown_table(pd.DataFrame(rows)) if rows else "无缺失",
    )


@tool
@guard("analyze_column")
def analyze_column(
    column: Annotated[str, "要分析的字段名，例如 '销售额'"],
    top_n: Annotated[int, "类别型字段展示的高频取值个数"] = 10,
) -> str:
    """深入分析单个字段：数值列给出均值/中位数/四分位/标准差/偏度/离群点，类别列给出取值分布，日期列给出时间跨度与月度趋势。

    适用场景：用户问"某个字段的分布/取值范围/有没有异常"时。
    """

    ctx = get_context()
    df = ctx.df
    col = resolve_column(df, column)
    series = df[col]
    n = max(len(df), 1)
    non_null = int(series.notna().sum())

    data: dict[str, Any] = {
        "column": col,
        "dtype": str(series.dtype),
        "non_null": non_null,
        "missing": int(series.isna().sum()),
        "missing_ratio": round(int(series.isna().sum()) / n, 4),
        "unique": int(series.nunique(dropna=True)),
    }
    table = ""
    summary = ""

    if is_numeric(series):
        s = series.dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        out_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr) if iqr else pd.Series(False, index=s.index)
        data.update(
            {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "sum": float(s.sum()),
                "q1": q1,
                "q3": q3,
                "skew": float(s.skew()) if len(s) > 2 else 0.0,
                "kurtosis": float(s.kurt()) if len(s) > 3 else 0.0,
                "zeros": int((s == 0).sum()),
                "negatives": int((s < 0).sum()),
                "outlier_count": int(out_mask.sum()),
                "outlier_ratio": round(int(out_mask.sum()) / max(non_null, 1), 4),
            }
        )
        summary = (
            f"字段「{col}」为数值型，共 {non_null} 个有效值：均值 {fmt_num(data['mean'])}、"
            f"中位数 {fmt_num(data['median'])}、范围 {fmt_num(data['min'])} ~ {fmt_num(data['max'])}、"
            f"标准差 {fmt_num(data['std'])}；偏度 {data['skew']:.2f}，"
            f"IQR 规则下发现 {data['outlier_count']} 个离群点（{data['outlier_ratio']:.1%}）。"
        )
        stats = pd.DataFrame(
            [{"统计量": k, "数值": fmt_num(v)} for k, v in data.items() if isinstance(v, (int, float)) and k not in ("non_null", "missing", "unique")]
        )
        table = markdown_table(stats)

    elif is_datetime(series) or df[col].astype(str).str.contains(r"[-/年]", na=False).mean() > 0.8:
        s = pd.to_datetime(series, errors="coerce", format="mixed")
        valid = s.dropna()
        if valid.empty:
            return fail("analyze_column", f"字段「{col}」无法解析成日期。")
        monthly = valid.dt.to_period("M").value_counts().sort_index()
        best = valid.dt.to_period("M").value_counts().idxmax()
        data.update(
            {
                "date_min": to_jsonable(valid.min()),
                "date_max": to_jsonable(valid.max()),
                "span_days": int((valid.max() - valid.min()).days),
                "monthly_counts": {str(k): int(v) for k, v in monthly.items()},
                "peak_month": str(best),
            }
        )
        summary = (
            f"字段「{col}」为日期型，覆盖 {data['date_min']} ~ {data['date_max']}"
            f"（跨度 {data['span_days']} 天），共 {len(valid)} 条有效记录；"
            f"记录最多的是 {best}（{int(monthly.max())} 条）。"
        )
        table = markdown_table(
            pd.DataFrame({"月份": [str(k) for k in monthly.index], "记录数": monthly.to_numpy()})
        )

    else:
        vc = series.astype(str).value_counts().head(top_n)
        data["top_values"] = [
            {"value": str(k), "count": int(v), "ratio": round(int(v) / max(non_null, 1), 4)}
            for k, v in vc.items()
        ]
        data["cardinality_ratio"] = round(int(series.nunique(dropna=True)) / max(non_null, 1), 4)
        top = vc.index[0]
        summary = (
            f"字段「{col}」为类别型，共 {int(series.nunique(dropna=True))} 个不同取值；"
            f"最高频取值是「{top}」（{int(vc.iloc[0])} 次，占 {int(vc.iloc[0]) / max(non_null, 1):.1%}）。"
        )
        table = top_table(series, top_n, label=col)

    return ok("analyze_column", summary, data, table=table)


@tool
@guard("detect_outliers")
def detect_outliers(
    column: Annotated[str, "要检测的数值字段；留空表示扫描所有数值字段"] = "",
    method: Annotated[str, "检测方法：'iqr'（四分位距，默认）或 'zscore'（标准差）"] = "iqr",
    threshold: Annotated[float, "iqr 方法为 1.5（可放宽到 3.0）；zscore 方法为 3.0"] = 1.5,
) -> str:
    """检测数值字段中的离群点（异常值），给出数量、占比和极值示例。

    适用场景：用户问"有没有异常值/极端值"，或做清洗决策之前。
    """

    ctx = get_context()
    df = ctx.df
    if method not in ("iqr", "zscore"):
        return fail("detect_outliers", f"不支持的方法 {method}，请用 'iqr' 或 'zscore'。")

    targets = [resolve_column(df, column)] if column else numeric_columns(df)
    if not targets:
        return fail("detect_outliers", "数据集中没有可用于离群检测的数值字段。")

    results = []
    for col in targets:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - threshold * iqr, q3 + threshold * iqr
        else:
            mean, std = s.mean(), s.std()
            if not std:
                continue
            low, high = mean - threshold * std, mean + threshold * std
        mask = (s < low) | (s > high)
        count = int(mask.sum())
        if count == 0:
            continue
        results.append(
            {
                "column": col,
                "method": method,
                "lower_bound": float(low),
                "upper_bound": float(high),
                "outlier_count": count,
                "outlier_ratio": round(count / max(len(s), 1), 4),
                "min_outlier": float(s[mask].min()),
                "max_outlier": float(s[mask].max()),
                "examples": [float(v) for v in s[mask].head(5)],
            }
        )

    results.sort(key=lambda r: r["outlier_ratio"], reverse=True)
    if not results:
        summary = f"按 {method} 方法（阈值 {threshold}）未发现明显离群点。"
    else:
        top = results[0]
        summary = (
            f"发现 {len(results)} 个字段存在离群点，最严重的是「{top['column']}」："
            f"{top['outlier_count']} 个异常值（占 {top['outlier_ratio']:.1%}），"
            f"正常区间约为 [{fmt_num(top['lower_bound'])}, {fmt_num(top['upper_bound'])}]。"
        )

    table = markdown_table(
        pd.DataFrame(
            [
                {
                    "字段": r["column"],
                    "离群数": r["outlier_count"],
                    "占比": f"{r['outlier_ratio']:.2%}",
                    "下界": fmt_num(r["lower_bound"]),
                    "上界": fmt_num(r["upper_bound"]),
                    "最大异常值": fmt_num(r["max_outlier"]),
                }
                for r in results
            ]
        )
    ) if results else "无离群点"

    return ok(
        "detect_outliers",
        summary,
        {"method": method, "threshold": threshold, "results": results},
        table=table,
    )


@tool
@guard("correlation_analysis")
def correlation_analysis(
    method: Annotated[str, "相关系数类型：'pearson'（线性，默认）、'spearman'（秩相关，抗异常）或 'kendall'"] = "pearson",
    top_n: Annotated[int, "展示相关性最强的字段对数量"] = 10,
    columns: Annotated[list[str] | None, "参与计算的数值字段；留空表示全部数值字段"] = None,
) -> str:
    """计算数值字段之间的相关性，输出相关性矩阵和最强相关字段对（自动忽略弱相关，标注正负方向）。

    适用场景：用户问"哪些指标相关""什么因素影响销售额"这类问题时。注意：相关不等于因果。
    """

    ctx = get_context()
    df = ctx.df
    if method not in ("pearson", "spearman", "kendall"):
        return fail("correlation_analysis", f"不支持的方法 {method}。")

    cols = numeric_columns(df, columns)
    if len(cols) < 2:
        return fail("correlation_analysis", f"至少需要 2 个数值字段才能算相关性，当前只有 {len(cols)} 个。")

    corr = df[cols].corr(method=method, numeric_only=True).round(3)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.isna(value):
                continue
            pairs.append({"column_a": a, "column_b": b, "corr": float(value), "abs_corr": abs(float(value))})
    pairs.sort(key=lambda p: p["abs_corr"], reverse=True)

    strong = [p for p in pairs if p["abs_corr"] >= 0.5]
    summary = (
        f"基于 {method} 方法计算了 {len(cols)} 个数值字段的相关性，"
        f"其中绝对值 ≥0.5 的强相关字段对 {len(strong)} 组。"
    )
    if pairs:
        top = pairs[0]
        direction = "正相关" if top["corr"] > 0 else "负相关"
        summary += f" 最强的一组是「{top['column_a']}」与「{top['column_b']}」（r={top['corr']:.3f}，{direction}）。"

    matrix_table = markdown_table(corr.reset_index().rename(columns={"index": "字段"}))
    pair_table = markdown_table(
        pd.DataFrame(
            [
                {
                    "字段A": p["column_a"],
                    "字段B": p["column_b"],
                    "相关系数": f"{p['corr']:.3f}",
                    "方向": "正" if p["corr"] > 0 else "负",
                    "强度": "强" if p["abs_corr"] >= 0.7 else ("中等" if p["abs_corr"] >= 0.4 else "弱"),
                }
                for p in pairs[:top_n]
            ]
        )
    )
    return ok(
        "correlation_analysis",
        summary,
        {
            "method": method,
            "columns": cols,
            "matrix": {c: {k: to_jsonable(v) for k, v in corr.loc[c].items()} for c in cols},
            "top_pairs": pairs[:top_n],
            "strong_pairs": strong,
        },
        table=matrix_table + "\n\n" + pair_table,
    )


PROFILING_TOOLS = [
    dataset_overview,
    detect_missing_values,
    analyze_column,
    detect_outliers,
    correlation_analysis,
]
