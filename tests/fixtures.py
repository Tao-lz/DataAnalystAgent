"""测试夹具：构造干净 / 脏两种数据集，并注册到临时存储中。"""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


@contextmanager
def temp_dir(name: str = "case"):
    """临时目录：放在项目内（兼容受限沙箱），退出时自动清理。"""

    from backend.app.config import get_settings

    root = get_settings().storage_path / "_tmp"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}-{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    try:
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


def clean_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    regions = rng.choice(["华东", "华北", "华南", "西南"], size=rows)
    channels = rng.choice(["天猫", "京东", "线下"], size=rows)
    price = np.round(rng.uniform(50, 800, size=rows), 2)
    qty = rng.integers(1, 5, size=rows)
    return pd.DataFrame(
        {
            "订单号": [f"SO{100000 + i}" for i in range(rows)],
            "订单日期": dates.strftime("%Y-%m-%d"),
            "地区": regions,
            "销售渠道": channels,
            "数量": qty,
            "单价": price,
            "销售额": np.round(price * qty, 2),
            "是否退货": rng.choice(["否", "是"], size=rows, p=[0.9, 0.1]),
            "满意度评分": np.round(rng.normal(4.2, 0.4, size=rows).clip(1, 5), 1),
        }
    )


def dirty_frame(rows: int = 60) -> pd.DataFrame:
    """带缺失、重复行、伪空值、离群点、整列全空的脏数据。"""

    df = clean_frame(rows)
    rng = np.random.default_rng(11)
    df.loc[rng.choice(rows, size=6, replace=False), "地区"] = np.nan
    df.loc[rng.choice(rows, size=5, replace=False), "满意度评分"] = np.nan
    df["备注"] = rng.choice(["", "加急", "未知", "N/A", "-", "?"], size=rows)
    df.loc[rng.choice(rows, size=3, replace=False), "销售额"] = 99999.0
    df["空列"] = None
    return pd.concat([df, df.head(5)], ignore_index=True)


def register_dataset(df: pd.DataFrame, name: str = "测试数据集") -> str:
    """把 DataFrame 注册成数据集，返回 dataset_id。"""

    from backend.app.config import get_settings
    from backend.app.core.store import get_store
    from backend.app.dataset.profiler import profile_dataframe

    settings = get_settings()
    store = get_store()
    dataset_id = f"D{uuid.uuid4().hex[:10]}"
    working = settings.upload_path / f"{dataset_id}_v0.csv"
    df.to_csv(working, index=False, encoding="utf-8-sig")
    store.save_dataset(
        dataset_id=dataset_id,
        name=name,
        filename=f"{name}.csv",
        original_path=working,
        working_path=working,
        sheet=None,
        encoding="utf-8",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        size_bytes=int(working.stat().st_size),
        profile=profile_dataframe(df, name=name),
    )
    return dataset_id


class tool_context:
    """with tool_context(dataset_id) as ctx: ... """

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.token = None
        self.ctx = None

    def __enter__(self):
        from backend.app.tools.context import ToolContext, set_context

        self.ctx = ToolContext(dataset_id=self.dataset_id, conversation_id="test-conv")
        self.token = set_context(self.ctx)
        return self.ctx

    def __exit__(self, *exc) -> None:
        from backend.app.tools.context import reset_context

        if self.token is not None:
            reset_context(self.token)
