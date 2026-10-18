"""23 个数据分析工具的离线测试（不需要大模型）。"""

from __future__ import annotations

import json

from ._harness import expect, expect_eq, expect_in
from .fixtures import clean_frame, dirty_frame, register_dataset, tool_context


def _call(name: str, args: dict | None = None) -> dict:
    """在当前上下文里调用工具，返回解析后的 JSON。"""

    from backend.app.tools.registry import TOOLS_BY_NAME

    tool = TOOLS_BY_NAME.get(name)
    expect(tool is not None, f"工具 {name} 未注册")
    raw = tool.invoke(args or {})
    return json.loads(raw)


def test_tool_inventory() -> None:
    from backend.app.tools.registry import (
        INTENT_TOOL_GROUPS,
        TOOLS_BY_NAME,
        TOOL_GROUPS,
        tools_for_intent,
    )

    expect_eq(len(TOOL_GROUPS), 4, "应有 4 个能力分组")
    expect(len(TOOLS_BY_NAME) >= 20, f"工具数量应不少于 20，实际 {len(TOOLS_BY_NAME)}")
    for name, tool in TOOLS_BY_NAME.items():
        expect(bool(tool.description), f"工具 {name} 缺少描述（会影响模型选择）")
        expect("." not in name, f"工具名 {name} 不应含点号")

    expect_eq(len(tools_for_intent("chat")), 0, "闲聊不应绑定任何工具")
    expect_eq(
        len(tools_for_intent("report")),
        len(TOOLS_BY_NAME),
        "报告意图应放开全部工具",
    )
    expect(
        len(tools_for_intent("statistics")) < len(tools_for_intent("report")),
        "统计意图应只绑定部分工具（路由感知绑定）",
    )
    expect_in("cleaning", INTENT_TOOL_GROUPS)


def test_profiling_tools() -> None:
    dataset_id = register_dataset(dirty_frame(), "脏数据_工具测试")
    with tool_context(dataset_id):
        overview = _call("dataset_overview")
        expect(overview["ok"], "dataset_overview 应成功")
        expect(overview["data"]["rows"] > 0, "应返回行数")
        expect(len(overview["data"]["columns"]) > 0, "应返回字段清单")

        missing = _call("detect_missing_values")
        expect(missing["ok"], "detect_missing_values 应成功")
        expect(missing["data"]["total_missing"] > 0, "应检出缺失值")
        expect(missing["data"]["affected_rows"] > 0, "应有受影响行")
        placeholders = missing["data"]["placeholder_columns"]
        expect(placeholders, "应识别出「未知 / N/A / - / ?」这类伪空值")
        expect(
            any(c["count"] > 0 for c in placeholders),
            f"伪空值计数应大于 0：{placeholders}",
        )

        numeric = _call("analyze_column", {"column": "销售额"})
        expect(numeric["ok"], "analyze_column(数值) 应成功")
        expect_in("mean", numeric["data"])
        expect(numeric["data"]["outlier_count"] > 0, "注入的极端值应被检出为离群点")

        category = _call("analyze_column", {"column": "地区"})
        expect(category["ok"], "analyze_column(类别) 应成功")
        expect(len(category["data"]["top_values"]) > 0, "应返回高频取值")

        corr = _call("correlation_analysis")
        expect(corr["ok"], "correlation_analysis 应成功")
        expect(len(corr["data"]["matrix"]) >= 2, "应返回相关性矩阵")

        outliers = _call("detect_outliers")
        expect(outliers["ok"], "detect_outliers 应成功")


def test_analyze_column_fuzzy_match() -> None:
    dataset_id = register_dataset(clean_frame(), "模糊列名")
    with tool_context(dataset_id):
        result = _call("analyze_column", {"column": " 销售额 "})
        expect(result["ok"], "带空格的列名应能被模糊匹配")
        expect_eq(result["data"]["column"], "销售额")

        bad = _call("analyze_column", {"column": "根本不存在的列"})
        expect(not bad["ok"], "不存在的列应返回 ok=false 而不是抛异常")
        expect("找不到列" in bad["error"], f"错误信息应给出提示：{bad.get('error')}")


def test_statistics_tools() -> None:
    dataset_id = register_dataset(clean_frame(), "统计测试")
    with tool_context(dataset_id):
        overall = _call("compute_statistics", {"metrics": ["sum", "mean"]})
        expect(overall["ok"], "整体统计应成功")
        expect_in("销售额", overall["data"]["describe"])

        grouped = _call(
            "compute_statistics",
            {"columns": ["销售额"], "group_by": ["地区"], "metrics": ["sum"]},
        )
        expect(grouped["ok"], "分组统计应成功")
        expect(grouped["data"]["total_groups"] == 4, "示例数据应分出 4 个地区")
        first = grouped["data"]["result"][0]
        expect("销售额_sum" in first, f"结果应含聚合列：{first}")
        values = [r["销售额_sum"] for r in grouped["data"]["result"]]
        expect_eq(values, sorted(values, reverse=True), "默认应按指标降序")

        counts = _call("value_counts", {"column": "销售渠道"})
        expect(counts["ok"], "value_counts 应成功")
        expect_eq(counts["data"]["total_unique"], 3, "示例数据有 3 个渠道")

        pivot = _call("pivot_table", {"index": "地区", "columns": "销售渠道", "values": "销售额"})
        expect(pivot["ok"], "pivot_table 应成功")
        expect_eq(pivot["data"]["shape"], [4, 3], "透视表形状应为 4x3")


def test_query_tool() -> None:
    dataset_id = register_dataset(clean_frame(), "筛选测试")
    with tool_context(dataset_id):
        result = _call(
            "filter_rows",
            {
                "filters": [
                    {"column": "地区", "op": "==", "value": "华东"},
                    {"column": "销售额", "op": ">", "value": 500},
                ],
                "sort_by": "销售额",
                "ascending": False,
                "limit": 5,
            },
        )
        expect(result["ok"], "filter_rows 应成功")
        expect(result["data"]["matched"] > 0, "应有命中记录")
        rows = result["data"]["rows"]
        expect(all(r["地区"] == "华东" for r in rows), "结果应全部是华东")
        expect(all(r["销售额"] > 500 for r in rows), "结果应全部大于 500")
        expect_eq(len(rows), min(5, result["data"]["matched"]), "应限制返回行数")
        expect_eq([r["销售额"] for r in rows], sorted((r["销售额"] for r in rows), reverse=True), "应按销售额降序")

        empty = _call("filter_rows", {"filters": [{"column": "地区", "op": "==", "value": "火星"}]})
        expect(empty["ok"] and empty["data"]["matched"] == 0, "无命中时应返回 0 而不是报错")


def test_cleaning_tools() -> None:
    dataset_id = register_dataset(dirty_frame(), "清洗测试")
    with tool_context(dataset_id):
        before = _call("dataset_overview")
        rows_before = before["data"]["rows"]
        missing_before = before["data"]["quality"]["total_missing"]
        dup_before = before["data"]["quality"]["duplicate_rows"]
        expect(dup_before > 0, "脏数据应含重复行")

        dropped = _call("drop_duplicates")
        expect(dropped["ok"], "去重应成功")
        expect(dropped["data"]["after"]["duplicates"] == 0, "去重后应无重复")
        expect(dropped["data"]["after"]["rows"] < rows_before, "行数应减少")

        cols = _call("drop_columns", {"columns": ["空列"]})
        expect(cols["ok"], "删除整列全空应成功")
        expect(
            cols["data"]["after"]["cols"] == cols["data"]["before"]["cols"] - 1,
            "列数应减 1",
        )

        normalized = _call("normalize_text", {"columns": ["备注"]})
        expect(normalized["ok"], "文本规整应成功")

        filled = _call("fill_missing", {"columns": ["满意度评分"], "strategy": "median"})
        expect(filled["ok"], "中位数填充应成功")
        expect(
            filled["data"]["after"]["missing"] < filled["data"]["before"]["missing"],
            "缺失值应减少",
        )

        filled_cat = _call("fill_missing", {"columns": ["地区"], "strategy": "mode"})
        expect(filled_cat["ok"], "众数填充类别列应成功")

        converted = _call("convert_types", {"columns": ["订单日期"], "target_type": "datetime"})
        expect(converted["ok"], "类型转换应成功")
        expect(
            any("datetime" in str(d["to"]) for d in converted["data"]["details"]),
            f"应转换成 datetime：{converted['data']['details']}",
        )

        bad_strategy = _call("fill_missing", {"columns": ["满意度评分"], "strategy": "不存在的策略"})
        expect(not bad_strategy["ok"], "非法策略应返回 ok=false")

        clipped = _call("clip_outliers", {"columns": ["销售额"], "action": "clip"})
        expect(clipped["ok"], "离群截断应成功")
        expect(clipped["data"]["affected_rows"] > 0, "应命中注入的极端值")

        restored = _call("restore_original_data")
        expect(restored["ok"], "恢复原始数据应成功")
        expect_eq(restored["data"]["after"]["rows"], rows_before, "恢复后行数应回到初始")
        expect_eq(restored["data"]["after"]["duplicates"], dup_before, "恢复后重复行应回来")
        expect_eq(restored["data"]["after"]["missing"], missing_before, "恢复后缺失值应回来")


def test_visualization_tools() -> None:
    dataset_id = register_dataset(clean_frame(), "可视化测试")
    charts = [
        ("plot_line", {"x": "订单日期", "y": "销售额"}),
        ("plot_bar", {"x": "地区", "y": "销售额"}),
        ("plot_histogram", {"column": "销售额"}),
        ("plot_heatmap", {}),
        ("plot_scatter", {"x": "单价", "y": "销售额"}),
        ("plot_box", {"x": "地区", "y": "销售额"}),
    ]
    with tool_context(dataset_id) as ctx:
        for name, args in charts:
            result = _call(name, args)
            expect(result["ok"], f"{name} 应成功：{result.get('error')}")
            chart = result["data"]["chart"]
            expect(chart["url"].startswith("/api/artifacts/"), f"{name} 应返回产物 URL")
            expect(bool(result.get("markdown")), f"{name} 应返回 Markdown 图片引用")
            expect(len(result["summary"]) > 10, f"{name} 应给出一句结论")

        expect_eq(len(ctx.artifacts), len(charts), "应登记全部图表产物")

        from backend.app.core.store import get_store
        from pathlib import Path

        for artifact in ctx.artifacts:
            row = get_store().get_artifact(artifact["id"])
            expect(row is not None, "产物应写入数据库")
            path = Path(row["file_path"])
            expect(path.exists(), f"图片文件应存在：{path}")
            expect(path.stat().st_size > 2000, "PNG 不应为空文件")


def test_visualization_invalid_column_returns_error() -> None:
    dataset_id = register_dataset(clean_frame(), "错误处理")
    with tool_context(dataset_id):
        result = _call("plot_bar", {"x": "不存在的维度", "y": "销售额"})
        expect(not result["ok"], "列名不存在时应返回 ok=false")
        expect(bool(result.get("error")), "应包含错误信息")


def test_evidence_and_state_are_native_types() -> None:
    """回归测试：numpy 标量混进状态会让 LangGraph 的 msgpack checkpoint 直接报错。"""

    import numpy as np

    dataset_id = register_dataset(clean_frame(), "原生类型回归")
    with tool_context(dataset_id) as ctx:
        _call("compute_statistics", {"group_by": ["地区"], "metrics": ["sum", "mean"]})
        _call("correlation_analysis")
        _call("plot_histogram", {"column": "销售额"})

        def check(value, path="evidence"):
            if isinstance(value, dict):
                for key, item in value.items():
                    check(item, f"{path}.{key}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    check(item, f"{path}[{idx}]")
            else:
                expect(
                    not isinstance(value, (np.generic, np.ndarray)),
                    f"{path} 含 numpy 类型：{type(value)}",
                )

        check(ctx.evidence)
        check(ctx.artifacts)
        # 能被 JSON 序列化，才说明是原生类型
        expect(len(json.dumps(ctx.evidence, ensure_ascii=False)) > 10, "证据应可 JSON 序列化")
