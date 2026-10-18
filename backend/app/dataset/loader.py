"""数据集自动解析：编码嗅探、分隔符推断、Excel 多 Sheet 选择、表头清洗。

目标是把"用户随手丢进来的一个文件"变成一张干净可用的 DataFrame。
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
from charset_normalizer import from_bytes

from ..config import get_settings

CSV_EXTS = {".csv", ".txt", ".tsv"}
EXCEL_EXTS = {".xlsx", ".xlsm", ".xls", ".xlsb"}

_SNIFF_BYTES = 256 * 1024
_ENCODING_TRIAL = ("utf-8", "gb18030", "gbk", "big5", "utf-16", "latin-1")


class DatasetLoadError(RuntimeError):
    """文件无法解析成表格时抛出，消息面向最终用户。"""


# --------------------------------------------------------------------------- #
# 编码 / 分隔符
# --------------------------------------------------------------------------- #
def sniff_encoding(path: Path | str) -> str:
    """按 BOM -> 严格试解码 -> charset_normalizer 猜 的顺序判定编码。"""

    path = Path(path)
    raw = path.read_bytes()[:_SNIFF_BYTES]
    if not raw:
        return "utf-8"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"

    for enc in _ENCODING_TRIAL:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue

    best = from_bytes(raw).best()
    if best and best.encoding:
        return best.encoding
    return "latin-1"


def sniff_delimiter(text: str, ext: str = ".csv") -> str:
    """用 csv.Sniffer 推断分隔符，失败时按扩展名兜底。"""

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        pass
    for cand in ("\t", ";", "|", ","):
        if sample.count(cand) > 0:
            return cand
    return "\t" if ext == ".tsv" else ","


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def list_sheets(path: Path | str) -> list[dict[str, Any]]:
    """列出 Excel 的 Sheet 及大致行列数；非 Excel 返回空列表。"""

    path = Path(path)
    if path.suffix.lower() not in EXCEL_EXTS:
        return []
    if path.suffix.lower() == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise DatasetLoadError(
                "读取 .xls 需要额外安装 xlrd（pip install xlrd）；"
                "建议先把文件另存为 .xlsx 再上传。"
            ) from exc
        book = pd.ExcelFile(path)
        return [{"name": n, "rows": None, "cols": None} for n in book.sheet_names]

    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = [
            {
                "name": ws.title,
                "rows": int(ws.max_row or 0),
                "cols": int(ws.max_column or 0),
            }
            for ws in wb.worksheets
        ]
    finally:
        wb.close()
    return sheets


def _pick_sheet(sheets: list[dict[str, Any]]) -> str | None:
    if not sheets:
        return None
    if len(sheets) == 1:
        return sheets[0]["name"]
    return max(sheets, key=lambda s: (s.get("rows") or 0) * (s.get("cols") or 1))["name"]


# --------------------------------------------------------------------------- #
# 表头 / 列名清洗
# --------------------------------------------------------------------------- #
_MULTISPACE = re.compile(r"\s+")


def _clean_column_names(columns: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, raw in enumerate(columns):
        name = "" if raw is None else str(raw)
        name = name.replace("\n", " ").replace("\r", " ").replace("\ufeff", "")
        name = _MULTISPACE.sub(" ", name).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"列{idx + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """去掉全空行列、规范化列名、重置索引。"""

    if df.empty and df.shape[1] == 0:
        raise DatasetLoadError("文件里没有解析到任何表格数据。")

    df = df.copy()
    df.columns = _clean_column_names(list(df.columns))

    auto_named = [c for c in df.columns if c.startswith("列") and df[c].isna().all()]
    if auto_named:
        df = df.drop(columns=auto_named)

    if len(df.columns):
        df = df.dropna(how="all")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def load_dataframe(
    path: Path | str,
    *,
    sheet: str | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取表格文件，返回 ``(dataframe, meta)``。"""

    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"文件不存在：{path.name}")

    ext = path.suffix.lower()
    limit = max_rows if max_rows is not None else get_settings().MAX_ROWS
    meta: dict[str, Any] = {
        "filename": path.name,
        "extension": ext,
        "sheet": None,
        "encoding": None,
        "delimiter": None,
        "truncated": False,
    }

    try:
        if ext in EXCEL_EXTS:
            sheets = list_sheets(path)
            target = sheet or _pick_sheet(sheets)
            engine = "xlrd" if ext == ".xls" else "openpyxl"
            df = pd.read_excel(path, sheet_name=target or 0, engine=engine, nrows=limit + 1)
            meta["sheet"] = target
            meta["sheets"] = [s["name"] for s in sheets]
        else:
            enc = encoding or sniff_encoding(path)
            text_head = path.read_bytes()[:_SNIFF_BYTES].decode(enc, errors="replace")
            delim = delimiter or sniff_delimiter(text_head, ext)
            df = pd.read_csv(
                path,
                encoding=enc,
                sep=delim,
                nrows=limit + 1,
                skip_blank_lines=True,
                engine="python",
            )
            meta["encoding"] = enc
            meta["delimiter"] = delim
    except DatasetLoadError:
        raise
    except ImportError as exc:  # pragma: no cover
        raise DatasetLoadError(f"缺少解析该格式所需的依赖：{exc}") from exc
    except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
        raise DatasetLoadError(f"解析失败：{type(exc).__name__}: {exc}") from exc

    if isinstance(df, dict):
        df = next(iter(df.values()))

    if len(df) > limit:
        df = df.head(limit)
        meta["truncated"] = True

    df = normalize_dataframe(df)
    meta["rows"] = int(df.shape[0])
    meta["cols"] = int(df.shape[1])
    return df, meta


def frame_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """把 DataFrame 序列化成 UTF-8-SIG 的 CSV（Excel 可直接打开）。"""

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")
