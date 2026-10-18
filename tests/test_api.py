"""HTTP 接口冒烟测试（不触发大模型调用）。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ._harness import expect, expect_eq

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "电商订单_示例数据.csv"


def _client() -> TestClient:
    from backend.app.main import app

    return TestClient(app)


def test_health_and_tools() -> None:
    with _client() as client:
        health = client.get("/api/health")
        expect_eq(health.status_code, 200)
        body = health.json()
        expect_eq(body["status"], "ok")
        expect(body["tool_count"] >= 20, "应报告工具数量")
        expect_eq(body["checkpoint"]["backend"], "SqliteCheckpointSaver")

        tools = client.get("/api/tools")
        expect_eq(tools.status_code, 200)
        expect(tools.json()["count"] >= 20, "应返回工具总数")

        graph = client.get("/api/graph")
        expect_eq(graph.status_code, 200)
        expect("understand" in graph.json()["mermaid"], "流程图应包含 understand 节点")


def test_upload_profile_preview_and_delete() -> None:
    expect(SAMPLE.exists(), f"示例数据缺失：{SAMPLE}")
    with _client() as client:
        with SAMPLE.open("rb") as fh:
            resp = client.post(
                "/api/datasets/upload",
                files={"file": (SAMPLE.name, fh, "text/csv")},
            )
        expect_eq(resp.status_code, 200, resp.text[:300])
        payload = resp.json()
        dataset = payload["dataset"]
        dataset_id = dataset["id"]
        expect(dataset["rows"] > 100, "应解析出行数")
        expect_eq(dataset["cols"], 16, "示例数据应解析出 16 列")
        expect(payload["profile"]["quality"]["score"] > 0, "应有质量评分")

        listing = client.get("/api/datasets/").json()
        expect(any(d["id"] == dataset_id for d in listing["items"]), "列表中应包含新数据集")

        profile = client.get(f"/api/datasets/{dataset_id}/profile").json()
        expect_eq(len(profile["columns"]), 16)

        preview = client.get(f"/api/datasets/{dataset_id}/preview", params={"limit": 5}).json()
        expect_eq(len(preview["rows"]), 5)
        expect_eq(preview["total_rows"], dataset["rows"])

        brief = client.get(f"/api/datasets/{dataset_id}/brief").json()
        expect(len(brief["columns"]) > 0, "brief 应包含字段")

        reset = client.post(f"/api/datasets/{dataset_id}/reset")
        expect_eq(reset.status_code, 200)
        expect_eq(reset.json()["rows"], dataset["rows"])

        deleted = client.delete(f"/api/datasets/{dataset_id}")
        expect_eq(deleted.status_code, 200)
        expect_eq(client.get(f"/api/datasets/{dataset_id}").status_code, 404)


def test_upload_rejects_bad_extension() -> None:
    with _client() as client:
        resp = client.post(
            "/api/datasets/upload",
            files={"file": ("恶意脚本.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        expect_eq(resp.status_code, 400, "不支持的格式应被拒绝")


def test_missing_dataset_returns_404() -> None:
    with _client() as client:
        expect_eq(client.get("/api/datasets/D不存在的").status_code, 404)
        expect_eq(client.get("/api/datasets/D不存在的/profile").status_code, 404)
        expect_eq(client.get("/api/reports/R不存在").status_code, 404)
        expect_eq(client.get("/api/artifacts/A不存在").status_code, 404)


def test_index_serves_frontend() -> None:
    with _client() as client:
        resp = client.get("/")
        expect_eq(resp.status_code, 200)
        expect("数据分析" in resp.text, "首页应返回前端页面")
        expect_eq(client.get("/static/app.js").status_code, 200)
        expect_eq(client.get("/static/style.css").status_code, 200)
