"""工具执行上下文。

工具函数本身不接收 dataset_id，而是从 ContextVar 里取当前请求的数据集，
这样同一个 LangGraph 图可以在并发请求之间安全复用。
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_settings
from ..core.store import Store, get_store
from ..dataset.loader import load_dataframe
from ..dataset.profiler import dataset_brief, profile_dataframe, to_native


class ToolError(RuntimeError):
    """工具执行失败，消息会原样回传给大模型，让它自己调整策略。"""


@dataclass
class ToolContext:
    """一次请求内的工具运行环境。"""

    dataset_id: str
    conversation_id: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    _df: pd.DataFrame | None = field(default=None, repr=False)
    _brief: dict[str, Any] | None = field(default=None, repr=False)

    # ---------------- 基础设施 ----------------
    @property
    def settings(self):
        return get_settings()

    @property
    def store(self) -> Store:
        return get_store()

    @property
    def meta(self) -> dict[str, Any]:
        row = self.store.get_dataset(self.dataset_id)
        if not row:
            raise ToolError(f"数据集 {self.dataset_id} 不存在，请先上传数据文件。")
        return row

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or "dataset")

    # ---------------- 数据读写 ----------------
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            working = Path(self.meta["working_path"])
            if not working.exists():
                raise ToolError("工作副本文件已丢失，请重新上传数据集。")
            df, _ = load_dataframe(working)
            self._df = df
        return self._df

    def invalidate(self) -> None:
        self._df = None
        self._brief = None

    @property
    def brief(self) -> dict[str, Any]:
        """给大模型看的轻量数据字典（按需计算并缓存）。"""

        if self._brief is None:
            self._brief = dataset_brief(self.df, self.name)
        return self._brief

    def persist(self, df: pd.DataFrame, note: str = "") -> Path:
        """把清洗后的数据落盘成新的工作副本，并刷新画像。"""

        meta = self.meta
        revision = int(meta.get("revision") or 0) + 1
        path = self.settings.upload_path / f"{self.dataset_id}_v{revision}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        profile = profile_dataframe(df, name=self.name)
        self.store.update_working_copy(
            self.dataset_id,
            working_path=path,
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            profile=profile,
        )
        self._df = df
        self._brief = None
        if note:
            self.add_evidence("数据清洗", note, {"revision": revision})
        return path

    def reset_to_original(self) -> pd.DataFrame:
        meta = self.meta
        df, _ = load_dataframe(meta["original_path"])
        self.persist(df, note="已将数据集恢复到最初上传的版本")
        return df

    # ---------------- 结果收集 ----------------
    def add_evidence(self, tool: str, summary: str, data: Any = None) -> None:
        self.evidence.append(
            {
                "tool": tool,
                "summary": summary,
                # 归一化成原生类型：checkpoint 用 msgpack 存储状态，混入 numpy 标量会报错
                "data": to_native(data),
                "run_id": self.run_id,
            }
        )

    def add_artifact(
        self,
        *,
        kind: str,
        title: str,
        description: str,
        file_path: Path,
    ) -> dict[str, Any]:
        artifact_id = f"{self.run_id}-{uuid.uuid4().hex[:8]}"
        self.store.save_artifact(
            artifact_id=artifact_id,
            dataset_id=self.dataset_id,
            conversation_id=self.conversation_id,
            kind=kind,
            title=title,
            description=description,
            file_path=file_path,
        )
        info = {
            "id": artifact_id,
            "kind": kind,
            "title": title,
            "description": description,
            "url": f"/api/artifacts/{artifact_id}",
        }
        self.artifacts.append(info)
        return info


_CURRENT: contextvars.ContextVar[ToolContext | None] = contextvars.ContextVar(
    "daa_tool_context", default=None
)


def set_context(ctx: ToolContext) -> contextvars.Token:
    return _CURRENT.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def get_context() -> ToolContext:
    ctx = _CURRENT.get()
    if ctx is None:
        raise ToolError("当前没有激活的数据集，请先上传并选择数据集。")
    return ctx
