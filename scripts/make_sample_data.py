"""生成演示用的示例数据集（含缺失值、重复行、伪空值、离群点）。

用法：python scripts/make_sample_data.py
输出：data/samples/电商订单_示例数据.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "samples"


def build(rows: int = 600, seed: int = 20260801) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = ["华东", "华北", "华南", "西南", "华中", "东北", "西北"]
    channels = ["天猫", "京东", "抖音", "线下门店", "私域小程序"]
    levels = ["普通会员", "银卡会员", "金卡会员", "钻石会员"]
    categories = ["手机数码", "家用电器", "服饰鞋包", "食品生鲜", "美妆个护", "图书文娱"]
    products = {
        "手机数码": ["智能手机 Pro", "蓝牙耳机", "平板电脑", "智能手表"],
        "家用电器": ["变频空调", "滚筒洗衣机", "扫地机器人", "空气净化器"],
        "服饰鞋包": ["轻羽绒外套", "运动跑鞋", "通勤双肩包", "休闲卫衣"],
        "食品生鲜": ["有机牛奶", "冻干咖啡", "海鲜礼盒", "坚果组合"],
        "美妆个护": ["精华液", "防晒霜", "面霜", "洗发水套装"],
        "图书文娱": ["商业图书", "儿童绘本", "桌游套装", "蓝牙音箱"],
    }
    price_band = {
        "手机数码": (499, 7999),
        "家用电器": (299, 6999),
        "服饰鞋包": (99, 1299),
        "食品生鲜": (29, 499),
        "美妆个护": (59, 899),
        "图书文娱": (19, 699),
    }

    dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    order_dates = rng.choice(dates, size=rows)
    order_dates.sort()

    cat = rng.choice(categories, size=rows, p=[0.26, 0.2, 0.18, 0.16, 0.12, 0.08])
    unit_price = np.array(
        [round(rng.uniform(*price_band[c]), 2) for c in cat]
    )
    quantity = rng.integers(1, 9, size=rows)
    discount = np.round(rng.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3], size=rows), 2)
    revenue = np.round(unit_price * quantity * (1 - discount), 2)

    df = pd.DataFrame(
        {
            "订单号": [f"SO{202500000 + i}" for i in range(rows)],
            "订单日期": pd.Series(order_dates).dt.strftime("%Y-%m-%d"),
            "客户ID": [f"C{rng.integers(10000, 19999)}" for _ in range(rows)],
            "客户等级": rng.choice(levels, size=rows, p=[0.45, 0.28, 0.19, 0.08]),
            "地区": rng.choice(regions, size=rows),
            "销售渠道": rng.choice(channels, size=rows, p=[0.3, 0.26, 0.2, 0.14, 0.1]),
            "商品类别": cat,
            "商品名称": [rng.choice(products[c]) for c in cat],
            "数量": quantity,
            "单价": unit_price,
            "折扣率": discount,
            "销售额": revenue,
            "是否退货": rng.choice(["否", "是"], size=rows, p=[0.91, 0.09]),
            "满意度评分": np.round(rng.normal(4.3, 0.55, size=rows).clip(1, 5), 1),
            "备注": rng.choice(
                ["", "客户加急", "需开发票", "礼品包装", "未知", "N/A", "-", "?"],
                size=rows,
                p=[0.72, 0.08, 0.06, 0.05, 0.035, 0.025, 0.02, 0.01],
            ),
        }
    )

    # ---- 人为注入一些"脏数据"，让质量体检有东西可发现 ----
    miss_idx = rng.choice(rows, size=int(rows * 0.06), replace=False)
    df.loc[miss_idx[: len(miss_idx) // 2], "满意度评分"] = np.nan
    df.loc[miss_idx[len(miss_idx) // 2 :], "客户等级"] = np.nan
    df.loc[rng.choice(rows, size=12, replace=False), "地区"] = np.nan

    # 离群点：几笔超大额订单
    big = rng.choice(rows, size=6, replace=False)
    df.loc[big, "销售额"] = (df.loc[big, "销售额"] * 35).round(2)
    df.loc[big, "数量"] = 60

    # 重复行
    dup_src = rng.choice(rows, size=18, replace=False)
    df = pd.concat([df, df.loc[dup_src]], ignore_index=True)

    # 整列全空
    df["数据来源系统"] = None
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build()
    target = OUT_DIR / "电商订单_示例数据.csv"
    df.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"已生成 {target}  ({df.shape[0]} 行 x {df.shape[1]} 列)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
