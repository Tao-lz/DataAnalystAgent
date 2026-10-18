"""数据清洗工具。

每一次清洗都会落成新的工作副本（revision+1），原始文件始终保留，
所以 Agent 可以放心操作，也能一键回滚。
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pandas.api import types as ptypes

from ..dataset.profiler import PLACEHOLDER_TOKENS, is_numeric, to_jsonable
from .base import fail, fmt_num, guard, markdown_table, ok, require_numeric, resolve_column, resolve_columns
from .context import ToolError, get_context


# --------------------------------------------------------------------------- #
# 内部辅助
# --------------------------------------------------------------------------- #
def _snapshot(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def _diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    return (
        f"行 {before['rows']} → {after['rows']}，"
        f"列 {before['cols']} → {after['cols']}，"
        f"缺失值 {before['missing']} → {after['missing']}，"
        f"重复行 {before['duplicates']} → {after['duplicates']}"
    )


def _apply(df: pd.DataFrame, note: str) -> dict[str, Any]:
    """落盘 + 返回前后对比。"""

    ctx = get_context()
    before = _snapshot(ctx.df)
    ctx.persist(df, note=note)
    after = _snapshot(df)
    return {"before": before, "after": after, "diff_text": _diff(before, after), "revision": int(ctx.meta.get("revision") or 0)}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
@tool
@guard("fill_missing")
def fill_missing(
    columns: Annotated[list[str], "要填充缺失值的字段列表，例如 ['销售额', '客户等级']"],
    strategy: Annotated[
        str,
        "填充策略：'mean' 均值、'median' 中位数、'mode' 众数、'ffill' 用前一个有效值、"
        "'bfill' 用后一个有效值、'zero' 填 0、'constant' 填指定值（需同时给 value）",
    ] = "median",
    value: Annotated[str | None, "strategy='constant' 时使用的填充值，例如 '未知' 或 '0'"] = None,
) -> str:
    """用统计值或指定值填充缺失值，只作用于指定字段，其余字段保持不变。

    适用场景：缺失比例不高（建议 <30%）、且删行会明显损失样本量时。数值列建议 median，类别列建议 mode。
    """

    ctx = get_context()
    df = ctx.df.copy()
    cols = resolve_columns(df, columns)
    if not cols:
        return fail("fill_missing", "请指定要填充的字段名。")

    allowed = ("mean", "median", "mode", "ffill", "bfill", "zero", "constant")
    if strategy not in allowed:
        return fail("fill_missing", f"不支持的策略 {strategy}，可用：{'/'.join(allowed)}。")
    if strategy == "constant" and value is None:
        return fail("fill_missing", "strategy='constant' 时必须提供 value 参数。")

    detail = []
    for col in cols:
        series = df[col]
        filled = int(series.isna().sum())
        if filled == 0:
            detail.append({"column": col, "filled": 0, "note": "本来就没有缺失值"})
            continue

        if strategy == "mean":
            if not is_numeric(series):
                detail.append({"column": col, "filled": 0, "note": "非数值列不能用均值填充，已跳过"})
                continue
            df[col] = series.fillna(series.mean())
            note = f"均值 {fmt_num(series.mean())}"
        elif strategy == "median":
            if not is_numeric(series):
                detail.append({"column": col, "filled": 0, "note": "非数值列不能用中位数填充，已跳过"})
                continue
            df[col] = series.fillna(series.median())
            note = f"中位数 {fmt_num(series.median())}"
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            if modes.empty:
                detail.append({"column": col, "filled": 0, "note": "没有可用于填充的众数"})
                continue
            df[col] = series.fillna(modes.iloc[0])
            note = f"众数 {modes.iloc[0]}"
        elif strategy == "ffill":
            df[col] = series.ffill()
            note = "前向填充"
        elif strategy == "bfill":
            df[col] = series.bfill()
            note = "后向填充"
        elif strategy == "zero":
            if not is_numeric(series):
                detail.append({"column": col, "filled": 0, "note": "非数值列不能填 0，已跳过"})
                continue
            df[col] = series.fillna(0)
            note = "填充 0"
        elif strategy == "constant":
            if is_numeric(series):
                try:
                    df[col] = series.fillna(float(value))
                except ValueError:
                    df[col] = series.astype("object").fillna(value)
            else:
                df[col] = series.astype("object").fillna(value)
            note = f"填充常量 {value}"
        detail.append({"column": col, "filled": filled, "note": note})

    total_filled = sum(d["filled"] for d in detail)
    if total_filled == 0:
        return ok(
            "fill_missing",
            f"没有执行任何填充：{'; '.join(d['note'] for d in detail) or '目标字段都没有缺失值'}。",
            {"details": detail},
        )

    changed_desc = ", ".join(f"{d['column']}({d['filled']} 个)" for d in detail if d["filled"])
    outcome = _apply(df, note=f"填充缺失值：{changed_desc}")
    summary = f"已填充 {total_filled} 个缺失值（字段：{', '.join(d['column'] for d in detail if d['filled'])}）；{outcome['diff_text']}。"
    return ok(
        "fill_missing",
        summary,
        {"strategy": strategy, "details": detail, **outcome},
        table=markdown_table(pd.DataFrame(detail)),
    )


@tool
@guard("drop_missing_rows")
def drop_missing_rows(
    columns: Annotated[list[str] | None, "只在这些字段上判断缺失；留空表示检查所有字段"] = None,
    how: Annotated[str, "'any' 任一字段缺失就删（默认），'all' 全部字段都缺失才删"] = "any",
) -> str:
    """删除包含缺失值的行。适合缺失比例很低、或关键字段缺失就必须作废的记录。

    适用场景：关键字段（如主键、金额）缺失且无法合理填充时。
    """

    ctx = get_context()
    df = ctx.df
    cols = resolve_columns(df, columns)
    subset = cols or list(df.columns)
    if how not in ("any", "all"):
        return fail("drop_missing_rows", "how 只能是 'any' 或 'all'。")

    removed = int(df[subset].isna().any(axis=1).sum()) if how == "any" else int(df[subset].isna().all(axis=1).sum())
    if removed == 0:
        return ok("drop_missing_rows", "目标字段没有缺失行，未做删除。", {"removed": 0})

    new_df = df.dropna(subset=subset, how=how).reset_index(drop=True)
    outcome = _apply(new_df, note=f"删除缺失行 {removed} 行（{how}，字段：{', '.join(subset[:10])}）")
    summary = (
        f"已删除 {removed} 行含缺失值的记录（判断字段：{', '.join(subset[:10])}{'…' if len(subset) > 10 else ''}）；"
        f"{outcome['diff_text']}。"
    )
    return ok("drop_missing_rows", summary, outcome)


@tool
@guard("drop_duplicates")
def drop_duplicates(
    subset: Annotated[list[str] | None, "按这些字段判断重复；留空表示整行完全相同才算重复"] = None,
    keep: Annotated[str, "保留策略：'first'（默认）或 'last'"] = "first",
) -> str:
    """删除重复记录，支持按指定字段组合判重（例如同一订单号+商品只保留一条）。

    适用场景：发现完全重复行，或用业务主键判重时。
    """

    ctx = get_context()
    df = ctx.df
    cols = resolve_columns(df, subset)
    if keep not in ("first", "last"):
        return fail("drop_duplicates", "keep 只能是 'first' 或 'last'。")

    dup_mask = df.duplicated(subset=cols or None, keep=keep)
    removed = int(dup_mask.sum())
    if removed == 0:
        return ok("drop_duplicates", "没有发现重复记录，未做删除。", {"removed": 0})

    new_df = df[~dup_mask].reset_index(drop=True)
    key = "整行" if not cols else ", ".join(cols)
    outcome = _apply(new_df, note=f"按 {key} 去重，删除 {removed} 行")
    summary = f"按 {key}{'（整行相同）' if not cols else ''} 判定，删除 {removed} 条重复记录；{outcome['diff_text']}。"
    return ok("drop_duplicates", summary, {**outcome, "judge_columns": cols or "整行"})


@tool
@guard("drop_columns")
def drop_columns(
    columns: Annotated[list[str], "要删除的字段列表"],
) -> str:
    """删除整列字段。适合处理全空列、常量列、与业务分析无关的列。

    适用场景：画面像体检里报了"整列全为空"或"取值恒定"时。
    """

    ctx = get_context()
    df = ctx.df
    cols = resolve_columns(df, columns)
    if not cols:
        return fail("drop_columns", "请指定要删除的字段名。")
    if len(cols) >= df.shape[1]:
        return fail("drop_columns", "不能删除全部字段。")

    new_df = df.drop(columns=cols).reset_index(drop=True)
    outcome = _apply(new_df, note=f"删除字段：{', '.join(cols)}")
    return ok("drop_columns", f"已删除 {len(cols)} 个字段（{', '.join(cols)}）；{outcome['diff_text']}。", outcome)


@tool
@guard("convert_types")
def convert_types(
    columns: Annotated[list[str], "要转换类型的字段列表"],
    target_type: Annotated[str, "目标类型：'int'、'float'、'datetime'、'category'、'string'、'bool'"],
) -> str:
    """转换字段类型，例如把文本日期列转成真正的日期类型（之后才能按时间分组），或把数字字符串转成数值。

    适用场景：字段被识别为字符串但实际是日期/数值时。
    """

    ctx = get_context()
    df = ctx.df.copy()
    cols = resolve_columns(df, columns)
    if not cols:
        return fail("convert_types", "请指定要转换的字段名。")

    allowed = {"int", "float", "datetime", "category", "string", "bool"}
    if target_type not in allowed:
        return fail("convert_types", f"不支持的类型 {target_type}，可用：{sorted(allowed)}。")

    detail = []
    for col in cols:
        original_dtype = str(df[col].dtype)
        try:
            if target_type == "int":
                numeric = pd.to_numeric(df[col], errors="coerce")
                failed = int(numeric.isna().sum() - df[col].isna().sum())
                df[col] = numeric.astype("Int64")
            elif target_type == "float":
                numeric = pd.to_numeric(df[col], errors="coerce")
                failed = int(numeric.isna().sum() - df[col].isna().sum())
                df[col] = numeric
            elif target_type == "datetime":
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                failed = int(parsed.isna().sum() - df[col].isna().sum())
                df[col] = parsed
            elif target_type == "category":
                df[col] = df[col].astype("category")
                failed = 0
            elif target_type == "bool":
                mapping = {
                    "true": True, "yes": True, "y": True, "1": True, "是": True, "真": True,
                    "false": False, "no": False, "n": False, "0": False, "否": False, "假": False,
                }
                if ptypes.is_bool_dtype(df[col]):
                    failed = 0
                elif is_numeric(df[col]):
                    df[col] = df[col].astype("boolean")
                    failed = 0
                else:
                    lowered = df[col].astype("object").where(df[col].notna())
                    mapped = lowered.map(lambda v: mapping.get(str(v).strip().lower()) if pd.notna(v) else None)
                    failed = int(mapped.isna().sum() - df[col].isna().sum())
                    df[col] = mapped.astype("boolean")
            else:  # string
                df[col] = df[col].astype("string")
                failed = 0
        except Exception as exc:  # noqa: BLE001
            detail.append({"column": col, "from": original_dtype, "to": target_type, "failed": None, "note": f"失败：{exc}"})
            continue
        detail.append(
            {
                "column": col,
                "from": original_dtype,
                "to": str(df[col].dtype),
                "failed": failed,
                "note": f"{failed} 个值无法转换，已置为缺失" if failed else "全部转换成功",
            }
        )

    outcome = _apply(df, note=f"类型转换：{target_type}（{', '.join(cols)}）")
    summary = f"已对 {len(cols)} 个字段执行 {target_type} 类型转换；{outcome['diff_text']}。"
    warnings = [d for d in detail if d.get("failed")]
    if warnings:
        summary += f" 注意：{len(warnings)} 个字段有无法转换的值，已置为缺失。"
    return ok("convert_types", summary, {"target_type": target_type, "details": detail, **outcome},
              table=markdown_table(pd.DataFrame(detail)))


@tool
@guard("normalize_text")
def normalize_text(
    columns: Annotated[list[str] | None, "要规整的文本字段；留空表示所有文本字段"] = None,
    strip: Annotated[bool, "是否去掉首尾空白字符"] = True,
    case: Annotated[str, "大小写统一：'keep'（默认）、'lower' 或 'upper'"] = "keep",
    replace_placeholders: Annotated[bool, "是否把 '?'、'N/A'、'未知'、'-' 等伪空值统一转成真正的缺失值"] = True,
) -> str:
    """规整文本字段：去首尾空格、统一大小写、把伪空值（?、N/A、未知、-）转成真正的缺失值。

    适用场景：同一类别因为空格/大小写/占位符被拆成多组，导致统计口径不一致时。
    """

    ctx = get_context()
    df = ctx.df.copy()
    cols = resolve_columns(df, columns) if columns else [
        str(c) for c in df.columns if not is_numeric(df[c]) and not ptypes.is_datetime64_any_dtype(df[c])
    ]
    if not cols:
        return fail("normalize_text", "没有找到可规整的文本字段。")

    detail = []
    for col in cols:
        series = df[col]
        changed = 0
        ph_replaced = 0
        work = series.astype("object").where(series.notna())
        valid = work.notna()
        if strip:
            stripped = work.map(lambda v: v.strip() if isinstance(v, str) else v)
            changed += int((stripped[valid] != work[valid]).sum())
            work = stripped
        if case == "lower":
            lowered = work.map(lambda v: v.lower() if isinstance(v, str) else v)
            changed += int((lowered[valid] != work[valid]).sum())
            work = lowered
        elif case == "upper":
            uppered = work.map(lambda v: v.upper() if isinstance(v, str) else v)
            changed += int((uppered[valid] != work[valid]).sum())
            work = uppered
        if replace_placeholders:
            mask = work.map(lambda v: isinstance(v, str) and v.strip().lower() in PLACEHOLDER_TOKENS)
            ph_replaced = int(mask.sum())
            work = work.where(~mask, None)
        df[col] = work
        detail.append({"column": col, "changed": changed, "placeholder_replaced": ph_replaced})

    outcome = _apply(df, note=f"文本规整：{len(cols)} 个字段")
    total_changed = sum(d["changed"] for d in detail)
    total_ph = sum(d["placeholder_replaced"] for d in detail)
    summary = (
        f"已规整 {len(cols)} 个文本字段，修正 {total_changed} 处空白/大小写差异，"
        f"并把 {total_ph} 处伪空值转为缺失值；{outcome['diff_text']}。"
    )
    return ok("normalize_text", summary, {"details": detail, **outcome},
              table=markdown_table(pd.DataFrame(detail)))


@tool
@guard("clip_outliers")
def clip_outliers(
    columns: Annotated[list[str], "要处理的数值字段列表"],
    method: Annotated[str, "'iqr'（默认）或 'zscore'"] = "iqr",
    threshold: Annotated[float, "iqr 默认 1.5；zscore 默认 3.0"] = 1.5,
    action: Annotated[str, "'clip' 截断到边界（默认）或 'remove' 删除所在行"] = "clip",
) -> str:
    """处理数值字段的离群点：截断到上下界（保留样本）或直接删除异常行。

    适用场景：确认异常值是录入错误或需要抑制极端值影响时。注意截断会改变分布，务必在报告中说明。
    """

    ctx = get_context()
    df = ctx.df.copy()
    cols = resolve_columns(df, columns)
    if not cols:
        return fail("clip_outliers", "请指定要处理的数值字段。")
    if action not in ("clip", "remove"):
        return fail("clip_outliers", "action 只能是 'clip' 或 'remove'。")

    numeric_cols = [require_numeric(df, c) for c in cols]
    bounds: dict[str, tuple[float, float]] = {}
    mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        s = df[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if not iqr:
                continue
            low, high = q1 - threshold * iqr, q3 + threshold * iqr
        elif method == "zscore":
            mean, std = s.mean(), s.std()
            if not std or pd.isna(std):
                continue
            low, high = mean - threshold * std, mean + threshold * std
        else:
            return fail("clip_outliers", "method 只能是 'iqr' 或 'zscore'。")
        bounds[col] = (float(low), float(high))
        mask |= (s < low) | (s > high)

    if not bounds:
        return ok("clip_outliers", "这些字段的分布过于集中，没有需要处理的离群点。", {"bounds": {}})

    affected = int(mask.sum())
    if action == "remove":
        new_df = df[~mask].reset_index(drop=True)
        note = f"删除离群行 {affected} 行"
    else:
        new_df = df.copy()
        for col, (low, high) in bounds.items():
            new_df[col] = new_df[col].clip(lower=low, upper=high)
        note = f"截断 {len(bounds)} 个字段的离群值"

    outcome = _apply(new_df, note=note)
    summary = (
        f"按 {method}（阈值 {threshold}）处理了 {len(bounds)} 个字段，"
        f"受影响记录 {affected} 条，采取{'删除异常行' if action == 'remove' else '截断到上下界'}；"
        f"{outcome['diff_text']}。"
    )
    return ok(
        "clip_outliers",
        summary,
        {
            "method": method,
            "action": action,
            "affected_rows": affected,
            "bounds": {k: {"lower": v[0], "upper": v[1]} for k, v in bounds.items()},
            **outcome,
        },
        table=markdown_table(
            pd.DataFrame(
                [{"字段": k, "下界": fmt_num(v[0]), "上界": fmt_num(v[1])} for k, v in bounds.items()]
            )
        ),
    )


@tool
@guard("restore_original_data")
def restore_original_data() -> str:
    """放弃所有清洗改动，把数据集恢复到最初上传时的状态。

    适用场景：清洗方向错了、或用户要求"重新来过"时。
    """

    ctx = get_context()
    before = _snapshot(ctx.df)
    df = ctx.reset_to_original()
    after = _snapshot(df)
    return ok(
        "restore_original_data",
        f"已恢复到原始数据；行 {before['rows']} → {after['rows']}，列 {before['cols']} → {after['cols']}。",
        {"before": before, "after": after},
    )


CLEANING_TOOLS = [
    fill_missing,
    drop_missing_rows,
    drop_duplicates,
    drop_columns,
    convert_types,
    normalize_text,
    clip_outliers,
    restore_original_data,
]
