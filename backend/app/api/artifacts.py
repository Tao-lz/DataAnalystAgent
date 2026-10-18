"""图表产物接口：列表与图片下载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..core.store import get_store

router = APIRouter()


@router.get("/", response_model=None)
def list_artifacts(
    dataset_id: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
) -> dict[str, Any]:
    items = get_store().list_artifacts(dataset_id, conversation_id)
    for item in items:
        item["url"] = f"/api/artifacts/{item['id']}"
    return {"items": items}


@router.get("/{artifact_id}", response_class=FileResponse)
def get_artifact(artifact_id: str) -> FileResponse:
    """按 id 返回 PNG 图片。"""

    row = get_store().get_artifact(artifact_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"图表 {artifact_id} 不存在。")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="图表文件已丢失，请重新生成。")
    # 不传 filename：避免 Starlette 加上 Content-Disposition: attachment，
    # 那样 <img src> 无法直接渲染、新标签页也会变成下载
    return FileResponse(path, media_type="image/png")
