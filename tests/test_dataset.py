"""数据集解析与画像测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._harness import expect, expect_eq, expect_in, expect_raises
from .fixtures import clean_frame, dirty_frame, temp_dir


def _write(path: Path, text: str, encoding: str) -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def test_sniff_encoding_utf8_bom() -> None:
    from backend.app.dataset.loader import sniff_encoding

    with temp_dir("bom") as tmp:
        path = tmp / "a.csv"
        _write(path, "列A,列B\n1,2\n", "utf-8-sig")
        expect_eq(sniff_encoding(path), "utf-8-sig", "BOM 文件应识别为 utf-8-sig")


def test_sniff_encoding_gbk() -> None:
    from backend.app.dataset.loader import sniff_encoding

    with temp_dir("gbk") as tmp:
        path = tmp / "b.csv"
        _write(path, "地区,销售额\n华东,100\n华南,200\n", "gb18030")
        enc = sniff_encoding(path)
        expect(enc in ("gb18030", "gbk"), f"GBK 文件识别为 {enc}")


def test_sniff_delimiter() -> None:
    from backend.app.dataset.loader import sniff_delimiter

    expect_eq(sniff_delimiter("a;b;c\n1;2;3\n"), ";", "分号分隔")
    expect_eq(sniff_delimiter("a\tb\tc\n1\t2\t3\n"), "\t", "制表符分隔")


def test_load_gbk_csv_keeps_chinese() -> None:
    from backend.app.dataset.loader import load_dataframe

    with temp_dir("gbkcsv") as tmp:
        path = tmp / "gbk.csv"
        _write(path, "地区,销售额\n华东,100\n华南,200\n", "gb18030")
        df, meta = load_dataframe(path)
        expect_eq(list(df.columns), ["地区", "销售额"], "列名应无乱码")
        expect_eq(int(df.shape[0]), 2)
        expect_eq(df.loc[0, "地区"], "华东")


def test_load_normalizes_columns() -> None:
    from backend.app.dataset.loader import load_dataframe

    with temp_dir("messy") as tmp:
        path = tmp / "messy.csv"
        path.write_text(
            " 地区 ,地区,,\n华东,华南,,\n西南,华北,,\n",
            encoding="utf-8",
        )
        df, _ = load_dataframe(path)
        names = list(df.columns)
        expect(len(names) == len(set(names)), f"列名应去重：{names}")
        expect(all(name.strip() == name for name in names), "列名应去掉首尾空格")
        expect("地区" in names, f"应保留地区列，实际 {names}")


def test_load_missing_file_raises() -> None:
    from backend.app.dataset.loader import DatasetLoadError, load_dataframe

    exc = expect_raises(DatasetLoadError, load_dataframe, Path("不存在的文件.csv"))
    expect("不存在" in str(exc), "错误信息应提示文件不存在")


def test_semantic_type_inference() -> None:
    from backend.app.dataset.profiler import infer_semantic_type

    n = 50
    frame = pd.DataFrame(
        {
            "订单号": [f"SO{i}" for i in range(n)],
            "订单日期": pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
            "销售额": np.linspace(10, 1000, n),
            "折扣率": np.tile([0.05, 0.1, 0.2, 0.3, 0.0], n // 5),
            "是否退货": ["是", "否"] * (n // 2),
            "地区": ["华东", "华北"] * (n // 2),
            "备注": [f"文本内容{i}" for i in range(n)],
            "常量列": ["X"] * n,
            "空列": [None] * n,
        }
    )
    got = {c: infer_semantic_type(frame[c], c, n)[0] for c in frame.columns}
    expect_eq(got["订单号"], "identifier", "唯一订单号应识别为主键")
    expect_in(got["订单日期"], ("datetime_string", "datetime"), "日期字符串应被识别为日期")
    expect_in(got["销售额"], ("float", "currency"), "销售额应为数值/金额")
    expect_eq(got["折扣率"], "percentage", "率类字段应识别为占比")
    expect_eq(got["是否退货"], "boolean", "是否类字段应识别为布尔")
    expect_eq(got["地区"], "category", "少量枚举应为类别")
    expect_eq(got["常量列"], "constant", "常量列应被识别")
    expect_eq(got["空列"], "empty", "全空列应被识别")


def test_quality_report_detects_issues() -> None:
    from backend.app.dataset.profiler import profile_dataframe

    profile = profile_dataframe(dirty_frame(), name="脏数据")
    quality = profile["quality"]
    expect(quality["duplicate_rows"] > 0, "应检出重复行")
    expect(quality["total_missing"] > 0, "应检出缺失值")
    expect(quality["score"] < 100, "有问题的数据评分应低于 100")
    scopes = " ".join(i.get("scope", "") for i in quality["issues"])
    expect("空列" in scopes, f"应报出整列全空：{quality['issues']}")
    expect(any(i.get("level") == "high" for i in quality["issues"]), "应存在 high 级问题")
    expect("numeric" in profile["column_groups"], "应有数值分组")
    expect("datetime" in profile["column_groups"], "应有日期分组")


def test_profile_is_json_serializable() -> None:
    import json

    from backend.app.dataset.profiler import profile_dataframe

    profile = profile_dataframe(clean_frame(), name="干净数据")
    text = json.dumps(profile, ensure_ascii=False)
    expect(len(text) > 100, "画像应可 JSON 序列化")
    expect(isinstance(profile["columns"][0]["name"], str), "字段名应为原生字符串")
    expect(isinstance(profile["quality"]["score"], float), "评分应为原生 float")


def test_dataset_brief_limits_columns() -> None:
    from backend.app.dataset.profiler import dataset_brief

    brief = dataset_brief(clean_frame(), name="测试", max_columns=3, sample_rows=2)
    expect_eq(len(brief["columns"]), 3, "应限制字段数量")
    expect(brief["truncated_columns"] > 0, "应报告被截断的字段数")
    expect_eq(len(brief["sample_rows"]), 2, "应返回指定行数的样例")
