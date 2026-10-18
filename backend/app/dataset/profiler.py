"""数据集画像：表结构识别、字段类型推断、数据质量体检。

产出既给前端表格用，也给大模型当"数据字典"用（严格控制 token 数量）。
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# 常见"伪空值"占位符
PLACEHOLDER_TOKENS = {
    "-", "--", "?", "??", "na", "n/a", "null", "none", "nan", "nil",
    "unknown", "missing", "#n/a", "#value!", "\\n",
    "未知", "无", "空", "缺失",
}

_DATE_HINT = ("-", "/", "年", "月", "日", ":", "T")
_ID_HINT = ("id", "编号", "序号", "代码", "code", "key", "uuid", "工号", "订单号")
_PCT_HINT = ("率", "占比", "比例", "百分比", "pct", "percent", "rate", "ratio")
_MONEY_HINT = (
    "金额", "销售额", "收入", "价格", "单价", "成本", "利润", "营收", "费用",
    "price", "amount", "revenue", "sales", "cost", "profit", "income", "gmv",
)
_BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "1", "0", "是", "否", "真", "假", "t", "f"}


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def to_jsonable(value: Any) -> Any:
    """把 numpy / pandas 类型递归转成原生 Python 类型，保证可 json.dumps。"""

    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        # bool / int 直接返回；注意 np.int64 不是 int 的子类，会走下面的分支
        return value
    if isinstance(value, float):
        # 关键：np.float64 是 float 的子类，必须显式转成原生 float，
        # 否则 round() 仍会返回 np.float64，导致 checkpointer 的 msgpack 序列化失败
        native = float(value)
        return None if (math.isnan(native) or math.isinf(native)) else round(native, 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.datetime64):
        return str(value)
    if value is pd.NaT:
        return None
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.Series):
        return {str(k): to_jsonable(v) for k, v in value.to_dict().items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def to_native(value: Any) -> Any:
    """规范化成一定能被 msgpack / json 序列化的原生类型。

    在 ``to_jsonable`` 基础上再做一次 JSON 往返，彻底消除 numpy / pandas 残留类型
    （LangGraph 的 checkpoint 用 msgpack 存储状态，混入 numpy 标量会直接报错）。
    """

    converted = to_jsonable(value)
    try:
        return json.loads(json.dumps(converted, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(converted)


def _name_has(name: str, hints: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


def is_numeric(series: pd.Series) -> bool:
    return bool(ptypes.is_numeric_dtype(series)) and not bool(ptypes.is_bool_dtype(series))


def is_datetime(series: pd.Series) -> bool:
    return bool(ptypes.is_datetime64_any_dtype(series)) or isinstance(series.dtype, pd.DatetimeTZDtype)


def is_boolean(series: pd.Series) -> bool:
    if ptypes.is_bool_dtype(series) and not is_numeric(series):
        return True
    if is_numeric(series) or is_datetime(series):
        return False
    uniq = series.dropna().unique()
    if len(uniq) > 2:
        return False
    return all(str(v).strip().lower() in _BOOL_TOKENS for v in uniq)


def _looks_like_datetime_strings(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(200)
    if len(values) < 3:
        return False
    if not any(any(h in v for h in _DATE_HINT) for v in values):
        return False
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(values, errors="coerce")
    return float(parsed.notna().mean()) >= 0.9


def detect_placeholder_missing(series: pd.Series) -> int:
    """统计"看起来是缺失"的占位符数量（?、N/A、未知、- 等）。"""

    if is_numeric(series) or is_datetime(series):
        return 0
    values = series.dropna().astype(str).str.strip().str.lower()
    if values.empty:
        return 0
    return int(values.isin(PLACEHOLDER_TOKENS).sum())


# --------------------------------------------------------------------------- #
# 字段类型推断
# --------------------------------------------------------------------------- #
def _is_datetime_like_number(series: pd.Series) -> bool:
    s = series.dropna()
    if not ptypes.is_integer_dtype(s) or s.empty:
        return False
    lens = s.astype("int64").astype(str).str.len()
    return bool(lens.isin({8, 10, 13}).mean() > 0.9)


def infer_semantic_type(series: pd.Series, name: str, total_rows: int) -> tuple[str, str]:
    """返回 ``(语义类型, 判断依据)``。"""

    non_null = int(series.notna().sum())
    uniq = int(series.nunique(dropna=True))

    if non_null == 0:
        return "empty", "整列缺失"
    if uniq <= 1:
        return "constant", f"只有 {uniq} 个取值"
    if is_datetime(series):
        return "datetime", "原生日期时间类型"
    if is_boolean(series):
        return "boolean", "布尔类型，或仅两个真假取值"

    if is_numeric(series):
        if _name_has(name, _PCT_HINT):
            return "percentage", "列名疑似占比，且为数值"
        if _name_has(name, _MONEY_HINT):
            return "currency", "列名疑似金额，且为数值"
        if _is_datetime_like_number(series):
            return "datetime_numeric", "整数形如 YYYYMMDD 或秒/毫秒时间戳"
        if uniq == total_rows and ptypes.is_integer_dtype(series):
            return "identifier", "取值唯一且为整数（主键特征）"
        return ("integer" if ptypes.is_integer_dtype(series) else "float"), "数值类型"

    if _looks_like_datetime_strings(series):
        return "datetime_string", "字符串可高比例解析为日期"

    unique_ratio = uniq / max(non_null, 1)
    if _name_has(name, _ID_HINT):
        return "identifier", f"列名含 ID 类关键词，基数 {uniq}"
    if uniq == non_null and unique_ratio > 0.9 and uniq == total_rows:
        return "identifier", "取值唯一（主键特征）"
    if uniq <= 50 and unique_ratio <= 0.5:
        return "category", f"只有 {uniq} 个枚举取值"
    if unique_ratio > 0.95:
        return "text", f"几乎取值唯一（{uniq}/{non_null}）"
    return "category", f"字符串枚举，基数 {uniq}"


# --------------------------------------------------------------------------- #
# 单列画像
# --------------------------------------------------------------------------- #
def _iqr_outliers(series: pd.Series) -> tuple[int, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return 0, 0.0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((series < low) | (series > high)).sum())
    return count, round(count / max(int(series.notna().sum()), 1), 4)


def profile_column(series: pd.Series, name: str, total_rows: int) -> dict[str, Any]:
    semantic, reason = infer_semantic_type(series, name, total_rows)
    non_null = int(series.notna().sum())
    missing = int(series.isna().sum())
    uniq = int(series.nunique(dropna=True))

    info: dict[str, Any] = {
        "name": name,
        "dtype": str(series.dtype),
        "semantic_type": semantic,
        "semantic_reason": reason,
        "non_null": non_null,
        "missing": missing,
        "missing_ratio": round(missing / max(total_rows, 1), 4),
        "unique": uniq,
        "unique_ratio": round(uniq / max(non_null, 1), 4),
        "is_unique": bool(uniq == non_null and missing == 0 and uniq > 1),
        "placeholder_missing": detect_placeholder_missing(series),
        "sample_values": [to_jsonable(v) for v in series.dropna().unique()[:5]],
    }

    if is_numeric(series):
        s = series.dropna()
        if not s.empty:
            info.update(
                {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "sum": float(s.sum()),
                    "zeros": int((s == 0).sum()),
                    "negatives": int((s < 0).sum()),
                    "skew": float(s.skew()) if len(s) > 2 else 0.0,
                    "kurtosis": float(s.kurt()) if len(s) > 3 else 0.0,
                    "q1": float(s.quantile(0.25)),
                    "median": float(s.quantile(0.50)),
                    "q3": float(s.quantile(0.75)),
                }
            )
            count, ratio = _iqr_outliers(s)
            info["outlier_count"] = count
            info["outlier_ratio"] = ratio
    elif is_datetime(series):
        s = series.dropna()
        if not s.empty:
            info["date_min"] = to_jsonable(s.min())
            info["date_max"] = to_jsonable(s.max())
            info["span_days"] = int((s.max() - s.min()).days)
    else:
        vc = series.astype(str).value_counts().head(8)
        info["top_values"] = [
            {"value": str(k), "count": int(v), "ratio": round(int(v) / max(non_null, 1), 4)}
            for k, v in vc.items()
        ]
        lengths = series.dropna().astype(str).str.len()
        if not lengths.empty:
            info["avg_length"] = round(float(lengths.mean()), 2)
            info["max_length"] = int(lengths.max())

    return to_jsonable(info)


# --------------------------------------------------------------------------- #
# 整表质量体检
# --------------------------------------------------------------------------- #
def _quality_report(df: pd.DataFrame, columns: list[dict[str, Any]]) -> dict[str, Any]:
    rows = max(int(df.shape[0]), 1)
    issues: list[dict[str, Any]] = []
    score = 100.0

    dup = int(df.duplicated().sum())
    if dup:
        ratio = dup / rows
        score -= min(15.0, ratio * 100 * 1.5)
        issues.append(
            {
                "level": "high" if ratio > 0.1 else "medium",
                "scope": "整表",
                "message": f"存在 {dup} 行完全重复（占 {ratio:.1%}）",
                "suggestion": "用 clean_data(action='drop_duplicates') 去重",
            }
        )

    avg_missing = float(np.mean([c["missing_ratio"] for c in columns])) if columns else 0.0
    if avg_missing > 0:
        score -= min(20.0, avg_missing * 100 * 0.8)

    empty_cols = [c["name"] for c in columns if c["semantic_type"] == "empty"]
    high_missing = [c for c in columns if c["missing_ratio"] >= 0.5 and c["semantic_type"] != "empty"]
    if empty_cols:
        score -= 5.0 * len(empty_cols)
        issues.append(
            {
                "level": "high",
                "scope": ", ".join(empty_cols),
                "message": "整列全为空，没有信息量",
                "suggestion": "用 clean_data(action='drop_columns') 删除",
            }
        )
    if high_missing:
        score -= 2.0 * len(high_missing)
        issues.append(
            {
                "level": "medium",
                "scope": ", ".join(c["name"] for c in high_missing),
                "message": "缺失率超过 50%，统计结果可能失真",
                "suggestion": "先判断缺失是否随机，再决定删除或填充",
            }
        )
    moderate = [c["name"] for c in columns if 0 < c["missing_ratio"] < 0.5]
    if moderate:
        issues.append(
            {
                "level": "low",
                "scope": ", ".join(moderate[:8]) + ("…" if len(moderate) > 8 else ""),
                "message": "存在少量缺失值",
                "suggestion": "数值列可用均值/中位数填充，类别列可用众数填充",
            }
        )

    const_cols = [c["name"] for c in columns if c["semantic_type"] == "constant"]
    if const_cols:
        score -= 2.0 * len(const_cols)
        issues.append(
            {
                "level": "low",
                "scope": ", ".join(const_cols),
                "message": "取值恒定，参与分析没有意义",
                "suggestion": "可考虑删除该列",
            }
        )

    placeholders = [c for c in columns if c.get("placeholder_missing")]
    if placeholders:
        total_ph = sum(c["placeholder_missing"] for c in placeholders)
        score -= min(10.0, total_ph / rows * 100)
        issues.append(
            {
                "level": "medium",
                "scope": ", ".join(c["name"] for c in placeholders[:8]),
                "message": f"共 {total_ph} 处伪空值（'?'、'N/A'、'未知'、'-' 等）未被识别为缺失",
                "suggestion": "用 clean_data(action='replace_missing_tokens') 统一转成缺失值",
            }
        )

    outlier_cols = [c for c in columns if c.get("outlier_ratio", 0) >= 0.05]
    if outlier_cols:
        score -= min(8.0, 2.0 * len(outlier_cols))
        issues.append(
            {
                "level": "low",
                "scope": ", ".join(c["name"] for c in outlier_cols[:8]),
                "message": "存在离群点（IQR 规则，占比 ≥5%）",
                "suggestion": "结合业务判断是异常还是长尾，必要时用 clean_data(action='clip_outliers')",
            }
        )

    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "duplicate_rows": dup,
        "duplicate_ratio": round(dup / rows, 4),
        "avg_missing_ratio": round(avg_missing, 4),
        "total_missing": int(df.isna().sum().sum()),
        "issues": issues,
    }


def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> dict[str, Any]:
    """完整画像：结构、字段类型、质量体检。"""

    rows, cols = df.shape
    total_rows = max(rows, 1)
    columns = [profile_column(df[c], str(c), total_rows) for c in df.columns]

    groups: dict[str, list[str]] = {
        "numeric": [], "datetime": [], "category": [], "text": [],
        "boolean": [], "identifier": [],
    }
    for c in columns:
        st = c["semantic_type"]
        if st in ("integer", "float", "percentage", "currency"):
            groups["numeric"].append(c["name"])
        elif st in ("datetime", "datetime_string", "datetime_numeric"):
            groups["datetime"].append(c["name"])
        elif st == "boolean":
            groups["boolean"].append(c["name"])
        elif st == "identifier":
            groups["identifier"].append(c["name"])
        elif st == "text":
            groups["text"].append(c["name"])
        else:
            groups["category"].append(c["name"])

    return to_native(
        {
            "name": name,
            "rows": int(rows),
            "cols": int(cols),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024 / 1024, 3),
            "column_groups": groups,
            "quality": _quality_report(df, columns),
            "columns": columns,
        }
    )


# --------------------------------------------------------------------------- #
# 给大模型看的轻量版
# --------------------------------------------------------------------------- #
def dataset_brief(
    df: pd.DataFrame,
    name: str = "dataset",
    *,
    max_columns: int = 40,
    sample_rows: int = 3,
) -> dict[str, Any]:
    """精简版数据字典：控制 prompt 体积，同时保留决策所需信息。"""

    total_rows = max(int(df.shape[0]), 1)
    columns: list[dict[str, Any]] = []
    for i, c in enumerate(df.columns):
        if i >= max_columns:
            break
        series = df[c]
        semantic, _ = infer_semantic_type(series, str(c), total_rows)
        item: dict[str, Any] = {
            "name": str(c),
            "type": semantic,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if is_numeric(series) and series.notna().any():
            item["range"] = [to_jsonable(series.min()), to_jsonable(series.max())]
        elif semantic in ("category", "boolean"):
            item["values"] = [to_jsonable(v) for v in series.dropna().unique()[:8]]
        columns.append(item)

    return to_native(
        {
            "name": name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": columns,
            "truncated_columns": max(0, int(df.shape[1]) - max_columns),
            "sample_rows": df.head(sample_rows).to_dict(orient="records"),
        }
    )


def profile_to_text(profile: dict[str, Any], max_columns: int = 40) -> str:
    """把画像渲染成可读文本（日志 / prompt 兜底）。"""

    lines = [
        f"数据集：{profile.get('name')}  形状：{profile.get('rows')} 行 × {profile.get('cols')} 列",
        f"质量评分：{profile.get('quality', {}).get('score')}",
        "字段清单：",
    ]
    for c in profile.get("columns", [])[:max_columns]:
        lines.append(
            f"  - {c['name']} | {c['semantic_type']} | 缺失 {c['missing']} | 唯一值 {c['unique']}"
        )
    return "\n".join(lines)
