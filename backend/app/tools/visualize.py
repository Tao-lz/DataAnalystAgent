"""可视化工具集：折线图、柱状图、直方图、热力图、散点图、箱线图。

所有图表都落盘成 PNG 作为"分析产物"（artifact），并把结论以 JSON 形式回给大模型，
这样模型既能引用具体数字，也能在最终报告里引用图片。
"""

from __future__ import annotations

from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from ..dataset.profiler import is_datetime, is_numeric, to_jsonable  # noqa: E402
from ..config import get_settings  # noqa: E402
from .base import (  # noqa: E402
    fail,
    fmt_num,
    guard,
    numeric_columns,
    ok,
    require_numeric,
    resolve_column,
    safe_filename,
)
from .context import ToolError, get_context  # noqa: E402

AGG_SET = ("sum", "mean", "count", "median", "max", "min")
_STYLED = False


def _style() -> None:
    """配置中文字体与画布风格（只做一次）。"""

    global _STYLED
    if _STYLED:
        return
    s = get_settings()
    try:
        plt.style.use(s.CHART_THEME)
    except Exception:  # noqa: BLE001 - 主题不存在时用默认的
        pass
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = s.CHART_DPI
    plt.rcParams["figure.figsize"] = (9.0, 4.8)
    _STYLED = True


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _finish(
    fig,
    *,
    tool_name: str,
    kind: str,
    title: str,
    description: str,
    summary: str,
    data: dict[str, Any] | None = None,
) -> str:
    ctx = get_context()
    filename = f"{ctx.run_id}-{kind}-{safe_filename(title)}.png"
    path = ctx.settings.artifact_path / filename
    fig.savefig(path, dpi=ctx.settings.CHART_DPI, bbox_inches="tight")
    plt.close(fig)

    info = ctx.add_artifact(kind=kind, title=title, description=description, file_path=path)
    payload = dict(data or {})
    payload["chart"] = {
        "artifact_id": info["id"],
        "title": title,
        "url": info["url"],
        "description": description,
    }
    return ok(tool_name, summary, payload, markdown=f"![{title}]({info['url']})")


def _time_keys(series: pd.Series) -> tuple[pd.Series, str]:
    """把日期列压成合适的粒度标签，返回 ``(标签, 粒度)``。"""

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    valid = parsed.dropna()
    if valid.empty:
        return series.astype(str), ""
    span = (valid.max() - valid.min()).days
    if span > 730:
        return parsed.dt.to_period("Y").dt.strftime("%Y"), "年"
    if span > 120:
        return parsed.dt.to_period("M").dt.strftime("%Y-%m"), "月"
    if span > 45:
        return parsed.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d"), "周"
    return parsed.dt.strftime("%Y-%m-%d"), "日"


def _looks_like_date(series: pd.Series) -> bool:
    if is_datetime(series):
        return True
    values = series.dropna().astype(str).head(50)
    if len(values) < 3:
        return False
    if not any(any(h in v for h in ("-", "/", "年")) for v in values):
        return False
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return float(parsed.notna().mean()) >= 0.9


def _line_like_frame(
    df: pd.DataFrame,
    xcol: str,
    ycol: str | None,
    huecol: str | None,
    agg: str,
) -> tuple[pd.DataFrame, str, str]:
    """把原始数据整理成"x 为索引、每个 hue 一列"的宽表。"""

    if agg not in AGG_SET:
        raise ToolError(f"聚合方式 {agg} 不支持，可用：{AGG_SET}")

    needed = [c for c in dict.fromkeys([xcol, ycol, huecol]) if c]
    work = df[needed].copy()

    if _looks_like_date(work[xcol]):
        keys, grain = _time_keys(work[xcol])
        work["_x"] = keys
    else:
        work["_x"] = work[xcol].astype(str)
        grain = ""
    work = work[work["_x"].notna()]

    if ycol is None:
        y_label = "记录数"
        if huecol:
            wide = work.groupby(["_x", huecol], observed=True).size().unstack(huecol)
        else:
            wide = work.groupby("_x", observed=True).size().to_frame(y_label)
    else:
        work["_y"] = pd.to_numeric(work[ycol], errors="coerce") if not is_numeric(work[ycol]) else work[ycol]
        work = work.dropna(subset=["_y"])
        y_label = f"{ycol}({agg})" if huecol else ycol
        if huecol:
            wide = work.pivot_table(
                index="_x", columns=huecol, values="_y", aggfunc=agg, observed=True
            )
        else:
            wide = work.groupby("_x", observed=True)["_y"].agg(agg).to_frame(ycol)

    if grain:
        wide = wide.sort_index()
    else:
        order = wide.sum(axis=1).sort_values(ascending=False)
        wide = wide.loc[order.index]
    return wide.fillna(0), y_label, grain


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """不依赖 scipy 的高斯核密度估计。"""

    if values.size == 0:
        return np.zeros_like(grid)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if std == 0:
        return np.zeros_like(grid)
    bw = 1.06 * std * values.size ** (-0.2)
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (values.size * bw * np.sqrt(2 * np.pi))


def _boxplot(ax, datasets, labels):
    """matplotlib 3.9 起把 boxplot 的 labels 参数改名为 tick_labels，这里做兼容。"""

    try:
        return ax.boxplot(datasets, tick_labels=labels, patch_artist=True)
    except TypeError:  # pragma: no cover - 兼容 matplotlib < 3.9
        return ax.boxplot(datasets, labels=labels, patch_artist=True)


def _trend_label(values: np.ndarray) -> str:
    if values.size < 2:
        return "数据点不足"
    x = np.arange(values.size, dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    scale = float(np.nanmax(np.abs(values))) or 1.0
    if abs(slope) / scale < 1e-3:
        return "整体平稳"
    return "整体上升" if slope > 0 else "整体下降"


# --------------------------------------------------------------------------- #
# 折线图
# --------------------------------------------------------------------------- #
@tool
@guard("plot_line")
def plot_line(
    x: Annotated[str, "横轴字段，通常是日期或有序维度，例如 '订单日期'"],
    y: Annotated[str | None, "纵轴数值字段，例如 '销售额'；留空表示统计记录条数"] = None,
    hue: Annotated[str | None, "分组字段（每条线一个取值），例如 '地区'；取值过多时会自动只保留前 8 个"] = None,
    agg: Annotated[str, "同一横轴取值下的聚合方式：sum/mean/count/median/max/min"] = "sum",
    title: Annotated[str | None, "图标题；留空则自动生成"] = None,
) -> str:
    """绘制折线图，展示随时间的趋势变化（例如"每月销售额走势"）。

    适用场景：横轴是日期、需要看趋势或对比多个分组的走势时。日期会自动按月/周/日聚合。
    """

    ctx = get_context()
    df = ctx.df
    xcol = resolve_column(df, x)
    ycol = require_numeric(df, y) if y else None
    huecol = resolve_column(df, hue) if hue else None

    if huecol and df[huecol].nunique(dropna=True) > 8:
        keep = df[huecol].astype(str).value_counts().head(8).index
        df = df[df[huecol].astype(str).isin(keep)]

    wide, y_label, grain = _line_like_frame(df, xcol, ycol, huecol, agg)
    if wide.empty:
        return fail("plot_line", "整理后没有可绘制的数据点，请检查字段是否包含大量缺失值。")
    if not grain and len(wide) > 20:
        wide = wide.head(20)

    grain_text = f"（按{grain}聚合）" if grain else ""
    chart_title = title or f"{xcol}{grain_text}上的{y_label}趋势"

    fig, ax = plt.subplots()
    x_labels = [str(i) for i in wide.index]
    for i, col in enumerate(wide.columns):
        ax.plot(
            x_labels, wide[col].to_numpy(), marker="o", markersize=4,
            linewidth=1.8, color=_PALETTE[i % len(_PALETTE)], label=str(col),
        )
    ax.set_title(chart_title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    if len(x_labels) > 8:
        step = max(len(x_labels) // 12, 1)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha="right")
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if len(wide.columns) > 1:
        ax.legend(title=hue, fontsize=8, ncol=min(len(wide.columns), 4))

    total = wide.sum(axis=1)
    peak_key, trough_key = total.idxmax(), total.idxmin()
    trend = _trend_label(total.to_numpy())
    peak_series = wide.loc[peak_key].idxmax() if len(wide.columns) > 1 else wide.columns[0]
    summary = (
        f"折线图已生成：{chart_title}。{total.index[0]} 到 {total.index[-1]} 共 {len(wide)} 个数据点，{trend}；"
        f"最高点在「{peak_key}」（合计 {fmt_num(total.loc[peak_key])}），最低点在「{trough_key}」"
        f"（合计 {fmt_num(total.loc[trough_key])}）"
    )
    summary += f"，峰值主要来自「{peak_series}」。" if len(wide.columns) > 1 else "。"

    return _finish(
        fig,
        tool_name="plot_line",
        kind="line",
        title=chart_title,
        description=f"{xcol} × {y_label} 折线图",
        summary=summary,
        data={
            "x": xcol,
            "y": y_label,
            "group": huecol,
            "grain": grain or "原始取值",
            "points": int(len(wide)),
            "series": [str(c) for c in wide.columns],
            "trend": trend,
            "peak": {"x": str(peak_key), "value": float(total.loc[peak_key])},
            "trough": {"x": str(trough_key), "value": float(total.loc[trough_key])},
            "values": {str(k): to_jsonable(v) for k, v in wide.to_dict(orient="index").items()},
        },
    )


# --------------------------------------------------------------------------- #
# 柱状图
# --------------------------------------------------------------------------- #
@tool
@guard("plot_bar")
def plot_bar(
    x: Annotated[str, "分类维度字段，例如 '地区' 或 '商品类别'"],
    y: Annotated[str | None, "纵轴数值字段；留空表示统计记录条数"] = None,
    hue: Annotated[str | None, "分组字段，用于并列对比，例如 '销售渠道'"] = None,
    agg: Annotated[str, "聚合方式：sum/mean/count/median/max/min"] = "sum",
    top_n: Annotated[int, "只展示前 N 个分类（按合计值降序）"] = 12,
    horizontal: Annotated[bool, "分类名称较长时建议设为 True（横向条形图）"] = False,
    title: Annotated[str | None, "图标题"] = None,
) -> str:
    """绘制柱状图/条形图，对分类维度做汇总对比（例如"各地区销售额"）。

    适用场景：用户问"哪个类别最高/最低""不同渠道的对比"这类分类比较问题时。
    """

    ctx = get_context()
    df = ctx.df
    xcol = resolve_column(df, x)
    ycol = require_numeric(df, y) if y else None
    huecol = resolve_column(df, hue) if hue else None

    if huecol and df[huecol].nunique(dropna=True) > 8:
        keep = df[huecol].astype(str).value_counts().head(8).index
        df = df[df[huecol].astype(str).isin(keep)]

    work = df[[c for c in dict.fromkeys([xcol, ycol, huecol]) if c]].copy()
    work["_x"] = work[xcol].astype(str)
    work = work[work[xcol].notna()]

    if ycol is None:
        y_label = "记录数"
        if huecol:
            wide = work.groupby(["_x", huecol], observed=True).size().unstack(huecol)
        else:
            wide = work.groupby("_x", observed=True).size().to_frame(y_label)
    else:
        if agg not in AGG_SET:
            return fail("plot_bar", f"聚合方式 {agg} 不支持，可用：{AGG_SET}")
        work["_y"] = pd.to_numeric(work[ycol], errors="coerce") if not is_numeric(work[ycol]) else work[ycol]
        y_label = f"{ycol}({agg})" if huecol else ycol
        if huecol:
            wide = work.pivot_table(index="_x", columns=huecol, values="_y", aggfunc=agg, observed=True)
        else:
            wide = work.groupby("_x", observed=True)["_y"].agg(agg).to_frame(ycol)

    wide = wide.fillna(0)
    wide = wide.loc[wide.sum(axis=1).sort_values(ascending=False).index]
    truncated = len(wide) > top_n
    wide = wide.head(max(top_n, 1))
    if wide.empty:
        return fail("plot_bar", "没有可绘制的数据，请检查字段取值是否为空。")

    chart_title = title or f"{xcol} 的 {y_label} 对比"
    fig, ax = plt.subplots(figsize=(9.0, 5.2 if horizontal else 4.8))

    positions = np.arange(len(wide))
    width = 0.8 / max(len(wide.columns), 1)
    for i, col in enumerate(wide.columns):
        offset = (i - (len(wide.columns) - 1) / 2) * width
        label = str(col) if len(wide.columns) > 1 else None
        bars = (
            ax.barh(positions + offset, wide[col].to_numpy(), height=width * 0.92, label=label, color=_PALETTE[i % len(_PALETTE)])
            if horizontal
            else ax.bar(positions + offset, wide[col].to_numpy(), width=width * 0.92, label=label, color=_PALETTE[i % len(_PALETTE)])
        )
        if len(wide.columns) == 1:
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    labels = [str(i) for i in wide.index]
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(y_label)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45 if max(len(s) for s in labels) > 4 else 0, ha="right" if max(len(s) for s in labels) > 4 else "center")
        ax.set_ylabel(y_label)
    ax.set_title(chart_title + ("（仅展示前 %d 项）" % top_n if truncated else ""))
    ax.grid(alpha=0.3, axis="x" if horizontal else "y")
    if len(wide.columns) > 1:
        ax.legend(title=hue, fontsize=8)

    total = wide.sum(axis=1)
    top_key, bottom_key = total.idxmax(), total.idxmin()
    total_all = float(total.sum())
    share = float(total.loc[top_key]) / total_all if total_all else 0.0
    summary = (
        f"柱状图已生成：{chart_title}。最高的分类是「{top_key}」（{fmt_num(total.loc[top_key])}，"
        f"占展示总量的 {share:.1%}），最低的是「{bottom_key}」（{fmt_num(total.loc[bottom_key])}）"
    )
    summary += f"；已截取前 {top_n} 项。" if truncated else "。"

    return _finish(
        fig,
        tool_name="plot_bar",
        kind="bar",
        title=chart_title,
        description=f"{xcol} × {y_label} 柱状图",
        summary=summary,
        data={
            "x": xcol,
            "y": y_label,
            "group": huecol,
            "categories": [str(i) for i in wide.index],
            "truncated": truncated,
            "top": {"category": str(top_key), "value": float(total.loc[top_key])},
            "bottom": {"category": str(bottom_key), "value": float(total.loc[bottom_key])},
            "values": {str(k): to_jsonable(v) for k, v in wide.to_dict(orient="index").items()},
        },
    )


# --------------------------------------------------------------------------- #
# 直方图
# --------------------------------------------------------------------------- #
@tool
@guard("plot_histogram")
def plot_histogram(
    column: Annotated[str, "要分析分布的数值字段，例如 '销售额'"],
    hue: Annotated[str | None, "可选分组字段，会分面绘制多组分布"] = None,
    bins: Annotated[int, "分箱数量，默认 30"] = 30,
    kde: Annotated[bool, "是否叠加核密度曲线"] = True,
    title: Annotated[str | None, "图标题"] = None,
) -> str:
    """绘制直方图，展示单个数值字段的分布形态（集中区间、偏态、长尾）。

    适用场景：用户问"销售额怎么分布""是否接近正态分布""集中在什么区间"。
    """

    ctx = get_context()
    df = ctx.df
    col = require_numeric(df, column)
    huecol = resolve_column(df, hue) if hue else None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return fail("plot_histogram", f"字段「{col}」没有有效的数值。")

    chart_title = title or f"{col} 的分布"
    fig, ax = plt.subplots()
    detail: dict[str, Any] = {}

    if huecol and df[huecol].nunique(dropna=True) <= 6:
        groups = df[[col, huecol]].dropna()
        for i, (name, sub) in enumerate(groups.groupby(huecol, observed=True)):
            values = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()
            ax.hist(values, bins=bins, alpha=0.5, label=str(name), color=_PALETTE[i % len(_PALETTE)], density=True)
            detail[str(name)] = {"count": int(values.size), "mean": float(np.mean(values)) if values.size else None}
        ax.legend(fontsize=8)
        main_values = series.to_numpy()
    else:
        values = series.to_numpy()
        ax.hist(values, bins=bins, color=_PALETTE[0], alpha=0.85, density=True, edgecolor="white", linewidth=0.5)
        main_values = values
        if kde:
            grid = np.linspace(float(values.min()), float(values.max()), 200)
            ax.plot(grid, _kde(values[:5000], grid), color=_PALETTE[3], linewidth=1.8, label="核密度 curve")
            ax.legend(fontsize=8)

    mean, median = float(np.mean(main_values)), float(np.median(main_values))
    ax.axvline(mean, color=_PALETTE[2], linestyle="--", linewidth=1.4, label=f"均值 {fmt_num(mean)}")
    ax.axvline(median, color=_PALETTE[1], linestyle=":", linewidth=1.4, label=f"中位数 {fmt_num(median)}")
    ax.set_title(chart_title)
    ax.set_xlabel(col)
    ax.set_ylabel("密度")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    q1, q3 = float(np.percentile(main_values, 25)), float(np.percentile(main_values, 75))
    skew = float(pd.Series(main_values).skew())
    shape = "接近对称" if abs(skew) < 0.5 else ("右偏（长尾在高值侧）" if skew > 0 else "左偏（长尾在低值侧）")
    summary = (
        f"直方图已生成：{col} 共 {main_values.size} 个有效值，均值 {fmt_num(mean)}、中位数 {fmt_num(median)}，"
        f"中间 50% 的数据落在 {fmt_num(q1)} ~ {fmt_num(q3)} 区间；分布{shape}（偏度 {skew:.2f}）。"
    )

    return _finish(
        fig,
        tool_name="plot_histogram",
        kind="histogram",
        title=chart_title,
        description=f"{col} 分布直方图",
        summary=summary,
        data={
            "column": col,
            "bins": bins,
            "count": int(main_values.size),
            "mean": mean,
            "median": median,
            "q1": q1,
            "q3": q3,
            "skew": skew,
            "shape": shape,
            "min": float(main_values.min()),
            "max": float(main_values.max()),
            "groups": detail,
        },
    )


# --------------------------------------------------------------------------- #
# 热力图
# --------------------------------------------------------------------------- #
@tool
@guard("plot_heatmap")
def plot_heatmap(
    columns: Annotated[list[str] | None, "参与计算的数值字段；留空表示全部数值字段"] = None,
    method: Annotated[str, "相关系数类型：pearson/spearman/kendall"] = "pearson",
    annot: Annotated[bool, "是否在每个格子里标注数值"] = True,
    title: Annotated[str | None, "图标题"] = None,
) -> str:
    """绘制相关性热力图，用颜色直观展示数值字段之间的相关强度与方向。

    适用场景：字段较多、需要一眼看出"哪些指标彼此关联"时；配合 correlation_analysis 使用效果更好。
    """

    ctx = get_context()
    df = ctx.df
    cols = numeric_columns(df, columns)
    if len(cols) < 2:
        return fail("plot_heatmap", f"至少需要 2 个数值字段，当前只有 {len(cols)} 个。")
    if method not in ("pearson", "spearman", "kendall"):
        return fail("plot_heatmap", "method 只能是 pearson/spearman/kendall。")

    corr = df[cols].corr(method=method, numeric_only=True).round(2)
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(6.0, n * 0.85), max(5.0, n * 0.75)))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    if annot:
        for i in range(n):
            for j in range(n):
                value = float(corr.iloc[i, j])
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else "black",
                )
    chart_title = title or f"数值字段相关性热力图（{method}）"
    ax.set_title(chart_title)
    fig.colorbar(im, ax=ax, shrink=0.85, label="相关系数")

    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append((abs(float(v)), float(v), a, b))
    pairs.sort(reverse=True)
    if pairs:
        _, v, a, b = pairs[0]
        direction = "正相关" if v > 0 else "负相关"
        summary = (
            f"热力图已生成：{len(cols)} 个数值字段中，相关性最强的是「{a}」与「{b}」"
            f"（r={v:.2f}，{direction}）；绝对值 ≥0.5 的字段对共 {sum(1 for p in pairs if p[0] >= 0.5)} 组。"
        )
    else:
        summary = f"热力图已生成：{len(cols)} 个数值字段之间没有可计算的相关性。"

    return _finish(
        fig,
        tool_name="plot_heatmap",
        kind="heatmap",
        title=chart_title,
        description=f"{len(cols)} 个数值字段的 {method} 相关热力图",
        summary=summary,
        data={
            "columns": cols,
            "method": method,
            "top_pairs": [
                {"a": p[2], "b": p[3], "corr": p[1]} for p in pairs[:10]
            ],
        },
    )


# --------------------------------------------------------------------------- #
# 散点图
# --------------------------------------------------------------------------- #
@tool
@guard("plot_scatter")
def plot_scatter(
    x: Annotated[str, "横轴数值字段，例如 '单价'"],
    y: Annotated[str, "纵轴数值字段，例如 '销售额'"],
    hue: Annotated[str | None, "分类着色字段"] = None,
    trendline: Annotated[bool, "是否叠加线性趋势线并给出拟合优度"] = True,
    sample: Annotated[int, "最多绘制多少个点（避免点太密）"] = 3000,
    title: Annotated[str | None, "图标题"] = None,
) -> str:
    """绘制散点图，观察两个数值字段之间的关系（是否线性、有无聚集与离群点）。

    适用场景：用户问"价格和销量有关系吗""两个指标是不是一起涨"。
    """

    ctx = get_context()
    df = ctx.df
    xcol = require_numeric(df, x)
    ycol = require_numeric(df, y)
    huecol = resolve_column(df, hue) if hue else None

    work = df[[c for c in dict.fromkeys([xcol, ycol, huecol]) if c]].copy()
    work[xcol] = pd.to_numeric(work[xcol], errors="coerce")
    work[ycol] = pd.to_numeric(work[ycol], errors="coerce")
    work = work.dropna(subset=[xcol, ycol])
    if work.empty:
        return fail("plot_scatter", "这两个字段没有成对的有效数值，无法绘制散点图。")
    sampled = work if len(work) <= sample else work.sample(sample, random_state=42)

    chart_title = title or f"{xcol} 与 {ycol} 的关系"
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    if huecol and sampled[huecol].nunique(dropna=True) <= 8:
        for i, (name, sub) in enumerate(sampled.groupby(huecol, observed=True)):
            ax.scatter(sub[xcol], sub[ycol], s=18, alpha=0.6, color=_PALETTE[i % len(_PALETTE)], label=str(name))
        ax.legend(title=hue, fontsize=8)
    else:
        ax.scatter(sampled[xcol], sampled[ycol], s=18, alpha=0.55, color=_PALETTE[0])

    r_value = float(work[xcol].corr(work[ycol]))
    fit_text = ""
    if trendline and len(work) > 2:
        xs = work[xcol].to_numpy(dtype=float)
        ys = work[ycol].to_numpy(dtype=float)
        k, b = np.polyfit(xs, ys, 1)
        grid = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(grid, k * grid + b, color=_PALETTE[3], linewidth=2.0, label="线性趋势")
        pred = k * xs + b
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        fit_text = f"线性拟合 {ycol} ≈ {k:.4f}×{xcol} + {b:.2f}，R²={r2:.3f}。"
        ax.legend(fontsize=8)

    ax.set_title(chart_title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(alpha=0.3)

    strength = "强" if abs(r_value) >= 0.7 else ("中等" if abs(r_value) >= 0.4 else "弱")
    direction = "正" if r_value > 0 else "负"
    summary = (
        f"散点图已生成：{xcol} 与 {ycol} 的皮尔逊相关系数 r={r_value:.3f}（{strength}{direction}相关），"
        f"共 {len(work)} 个有效样本点"
    )
    summary += f"，绘制时抽样 {len(sampled)} 个。" if len(sampled) < len(work) else "。"
    if fit_text:
        summary += " " + fit_text

    return _finish(
        fig,
        tool_name="plot_scatter",
        kind="scatter",
        title=chart_title,
        description=f"{xcol} × {ycol} 散点图",
        summary=summary,
        data={
            "x": xcol,
            "y": ycol,
            "group": huecol,
            "samples": int(len(work)),
            "plotted": int(len(sampled)),
            "corr": r_value,
            "strength": f"{strength}{direction}相关",
            "fit": fit_text,
        },
    )


# --------------------------------------------------------------------------- #
# 箱线图
# --------------------------------------------------------------------------- #
@tool
@guard("plot_box")
def plot_box(
    y: Annotated[str | None, "要展示分布的数值字段；留空表示所有数值字段"] = None,
    x: Annotated[str | None, "分组维度字段，例如 '地区'，用于对比不同组的分布差异"] = None,
    title: Annotated[str | None, "图标题"] = None,
) -> str:
    """绘制箱线图，对比不同分组下数值的分布差异（中位数、四分位、离群点）。

    适用场景：用户问"各地区的销售额分布差异大吗""哪一组的波动最大"。
    """

    ctx = get_context()
    df = ctx.df
    ycol = require_numeric(df, y) if y else None
    xcol = resolve_column(df, x) if x else None

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    if xcol and ycol:
        work = df[[xcol, ycol]].dropna()
        work[ycol] = pd.to_numeric(work[ycol], errors="coerce")
        work = work.dropna(subset=[ycol])
        top = work[xcol].astype(str).value_counts().head(10).index.tolist()
        work = work[work[xcol].astype(str).isin(top)]
        groups = [work.loc[work[xcol].astype(str) == g, ycol].to_numpy() for g in top]
        if not groups:
            return fail("plot_box", "分组后没有数据。")
        bp = _boxplot(ax, groups, top)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(_PALETTE[i % len(_PALETTE)])
            box.set_alpha(0.7)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        chart_title = title or f"不同{xcol}下的{ycol}分布"
        means = {g: float(np.mean(vals)) if vals.size else None for g, vals in zip(top, groups)}
        best = max(means, key=lambda k: means[k] if means[k] is not None else -np.inf)
        summary = (
            f"箱线图已生成：{len(top)} 个 {xcol} 分组的 {ycol} 分布对比完成；"
            f"均值最高的是「{best}」（{fmt_num(means[best])}）。"
        )
        data = {
            "group": xcol,
            "value": ycol,
            "groups": [
                {
                    "name": g,
                    "count": int(vals.size),
                    "median": float(np.median(vals)) if vals.size else None,
                    "mean": float(np.mean(vals)) if vals.size else None,
                    "q1": float(np.percentile(vals, 25)) if vals.size else None,
                    "q3": float(np.percentile(vals, 75)) if vals.size else None,
                }
                for g, vals in zip(top, groups)
            ],
            "highest_mean_group": best,
        }
    else:
        cols = [ycol] if ycol else numeric_columns(df)
        if not cols:
            return fail("plot_box", "没有可用于绘制的数值字段。")
        cols = cols[:12]
        values = [pd.to_numeric(df[c], errors="coerce").dropna().to_numpy() for c in cols]
        keep = [(c, v) for c, v in zip(cols, values) if v.size]
        if not keep:
            return fail("plot_box", "这些字段都没有有效数值。")
        cols = [c for c, _ in keep]
        bp = _boxplot(ax, [v for _, v in keep], cols)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(_PALETTE[i % len(_PALETTE)])
            box.set_alpha(0.7)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        chart_title = title or "各数值字段的分布对比"
        spread = {
            c: float(np.percentile(v, 75) - np.percentile(v, 25)) for c, v in keep
        }
        widest = max(spread, key=spread.get)
        summary = (
            f"箱线图已生成：对比了 {len(cols)} 个数值字段的分布；"
            f"四分位距最大（波动最剧烈）的是「{widest}」（IQR={fmt_num(spread[widest])}）。"
        )
        data = {
            "columns": cols,
            "iqr": spread,
            "widest": widest,
        }

    ax.set_title(chart_title)
    ax.grid(alpha=0.3, axis="y")
    return _finish(
        fig,
        tool_name="plot_box",
        kind="box",
        title=chart_title,
        description="箱线图分布对比",
        summary=summary,
        data=data,
    )


VISUALIZATION_TOOLS = [
    plot_line,
    plot_bar,
    plot_histogram,
    plot_heatmap,
    plot_scatter,
    plot_box,
]

# 导入本模块即完成画布风格与中文字体的一键配置
_style()
