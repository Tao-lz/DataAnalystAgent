"""数据集接口：上传、解析、画像、预览、重置、删除。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..config import get_settings
from ..core.store import get_store
from ..dataset.loader import DatasetLoadError, list_sheets, load_dataframe
from ..dataset.profiler import profile_dataframe, to_jsonable
from ..tools.context import ToolContext, reset_context, set_context

router = APIRouter()

_CHUNK = 1024 * 1024


def _dataset_summary(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile") or {}
    return {
        "id": row["id"],
        "name": row["name"],
        "filename": row["filename"],
        "rows": row.get("rows") or 0,
        "cols": row.get("cols") or 0,
        "size_bytes": row.get("size_bytes") or 0,
        "sheet": row.get("sheet"),
        "encoding": row.get("encoding"),
        "revision": row.get("revision") or 0,
        "quality_score": (profile.get("quality") or {}).get("score"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


async def _save_upload(file: UploadFile, target: Path, limit: int) -> int:
    """边写边校验大小，避免超大文件打满磁盘。"""

    size = 0
    with target.open("wb") as out:
        while chunk := await file.read(_CHUNK):
            size += len(chunk)
            if size > limit:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"文件超过上限 {limit // 1024 // 1024} MB，请先抽样或拆分后上传。",
                )
            out.write(chunk)
    return size


@router.post("/upload", response_model=None)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV / Excel 数据文件"),
    name: str | None = Form(default=None, description="展示用名称，留空取文件名"),
    sheet: str | None = Form(default=None, description="Excel 指定 Sheet，留空自动选最大的"),
) -> dict[str, Any]:
    """上传数据集：自动嗅探编码与分隔符、解析表结构、生成数据画像。"""

    settings = get_settings()
    store = get_store()
    filename = Path(file.filename or "dataset.csv").name
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"暂不支持 {ext or '未知'} 格式，仅支持：{', '.join(settings.allowed_extensions)}",
        )

    dataset_id = f"D{uuid.uuid4().hex[:12]}"
    target = settings.upload_path / f"{dataset_id}{ext}"
    size = await _save_upload(file, target, settings.MAX_FILE_SIZE)

    try:
        df, meta = load_dataframe(target, sheet=sheet)
    except DatasetLoadError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    display_name = (name or Path(filename).stem).strip() or Path(filename).stem
    profile = profile_dataframe(df, name=display_name)

    # 工作副本：后续所有清洗都作用在副本上，原始文件永不改动
    working = settings.upload_path / f"{dataset_id}_v0.csv"
    df.to_csv(working, index=False, encoding="utf-8-sig")

    store.save_dataset(
        dataset_id=dataset_id,
        name=display_name,
        filename=filename,
        original_path=target,
        working_path=working,
        sheet=meta.get("sheet"),
        encoding=meta.get("encoding"),
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        size_bytes=size,
        profile=profile,
    )

    warnings: list[str] = []
    if meta.get("truncated"):
        warnings.append(f"数据行数超过上限，仅读取前 {settings.MAX_ROWS} 行。")
    if profile["quality"]["score"] < 80:
        warnings.append(
            f"数据质量评分 {profile['quality']['score']}，建议先处理缺失值与重复行。"
        )
    sheets = list_sheets(target)
    if len(sheets) > 1:
        warnings.append(
            f"该文件包含 {len(sheets)} 个 Sheet，已自动选择「{meta.get('sheet')}」；"
            "如需其他 Sheet，请重新上传时指定。"
        )

    row = store.get_dataset(dataset_id) or {}
    return {"dataset": _dataset_summary(row), "profile": profile, "warnings": warnings}


@router.get("/", response_model=None)
def list_datasets() -> dict[str, Any]:
    """列出已上传的数据集。"""

    return {"items": [_dataset_summary(r) for r in get_store().list_datasets()]}


@router.get("/{dataset_id}", response_model=None)
def get_dataset(dataset_id: str) -> dict[str, Any]:
    row = get_store().get_dataset(dataset_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在。")
    return {"dataset": _dataset_summary(row), "profile": row.get("profile") or {}}


@router.get("/{dataset_id}/profile", response_model=None)
def get_profile(dataset_id: str, refresh: bool = Query(default=False)) -> dict[str, Any]:
    """获取数据画像；``refresh=true`` 时按当前工作副本重新计算。"""

    store = get_store()
    row = store.get_dataset(dataset_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在。")
    if refresh or not row.get("profile"):
        ctx = ToolContext(dataset_id=dataset_id)
        token = set_context(ctx)
        try:
            profile = profile_dataframe(ctx.df, name=ctx.name)
        finally:
            reset_context(token)
        store.update_working_copy(
            dataset_id,
            working_path=Path(row["working_path"]),
            rows=int(profile["rows"]),
            cols=int(profile["cols"]),
            profile=profile,
        )
        return profile
    return row["profile"]


@router.get("/{dataset_id}/brief", response_model=None)
def get_brief(dataset_id: str) -> dict[str, Any]:
    """给大模型看的轻量数据字典。"""

    if not get_store().get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在。")
    ctx = ToolContext(dataset_id=dataset_id)
    token = set_context(ctx)
    try:
        return ctx.brief
    finally:
        reset_context(token)


@router.get("/{dataset_id}/preview", response_model=None)
def preview(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """分页预览当前（已清洗）数据。"""

    if not get_store().get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在。")
    ctx = ToolContext(dataset_id=dataset_id)
    token = set_context(ctx)
    try:
        df = ctx.df
        part = df.iloc[offset : offset + limit]
        records = [to_jsonable(row) for row in part.to_dict(orient="records")]
        return {
            "dataset_id": dataset_id,
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "rows": records,
            "total_rows": int(df.shape[0]),
            "offset": offset,
            "limit": limit,
        }
    finally:
        reset_context(token)


@router.post("/{dataset_id}/reset", response_model=None)
def reset_dataset(dataset_id: str) -> dict[str, Any]:
    """放弃所有清洗改动，恢复到最初上传的版本。"""

    if not get_store().get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在。")
    ctx = ToolContext(dataset_id=dataset_id)
    token = set_context(ctx)
    try:
        df = ctx.reset_to_original()
    finally:
        reset_context(token)
    row = get_store().get_dataset(dataset_id) or {}
    return {"dataset": _dataset_summary(row), "rows": int(df.shape[0]), "cols": int(df.shape[1])}


@router.delete("/{dataset_id}", response_model=None)
def delete_dataset(dataset_id: str) -> dict[str, str]:
    """删除数据集记录及其工作副本（原始上传文件保留）。"""

    store = get_store()
    row = store.get_dataset(dataset_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在。")

    for path in get_settings().upload_path.glob(f"{dataset_id}_v*.csv"):
        try:
            path.unlink()
        except OSError:
            pass
    store.delete_dataset(dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id}
