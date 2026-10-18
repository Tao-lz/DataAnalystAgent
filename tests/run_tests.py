"""离线测试入口。

    python tests/run_tests.py

不依赖 pytest，也不需要联网（不触发大模型调用）。
测试使用独立的临时 storage 目录，不会污染项目的 storage/。
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须放在导入 backend.app.config 之前：环境变量优先级高于 .env
# 用项目内的独立目录（而不是系统临时目录），避免受沙箱/权限限制影响
_TMP_STORAGE = ROOT / ".test-storage"
shutil.rmtree(_TMP_STORAGE, ignore_errors=True)
_TMP_STORAGE.mkdir(parents=True, exist_ok=True)
os.environ["STORAGE_DIR"] = str(_TMP_STORAGE)
os.environ["DEBUG"] = "false"

MODULES = [
    "tests.test_dataset",
    "tests.test_tools",
    "tests.test_checkpointer",
    "tests.test_api",
]


def main() -> int:
    print("=" * 72)
    print("  DataAnalysisAgent 离线测试")
    print(f"  临时 storage：{_TMP_STORAGE}")
    print("=" * 72)

    total_pass = total_fail = 0
    details: list[str] = []
    for name in MODULES:
        print(f"\n[{name}]")
        module = importlib.import_module(name)
        passed, failed, failures = _run(module)
        total_pass += passed
        total_fail += failed
        details.extend(failures)

    print("\n" + "=" * 72)
    print(f"  通过 {total_pass} 个，失败 {total_fail} 个")
    if details:
        print("\n失败详情：")
        for item in details:
            print("-" * 72)
            print(item)
    print("=" * 72)

    shutil.rmtree(_TMP_STORAGE, ignore_errors=True)
    return 1 if total_fail else 0


def _run(module):
    from tests._harness import run_module

    return run_module(module)


if __name__ == "__main__":
    sys.exit(main())
