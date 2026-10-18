"""极简测试框架：不引入 pytest，`python tests/run_tests.py` 直接跑。"""

from __future__ import annotations

import traceback
from typing import Any, Callable


class AssertionFail(AssertionError):
    pass


def expect(condition: Any, message: str = "") -> None:
    if not condition:
        raise AssertionFail(message or "断言失败")


def expect_eq(actual: Any, expected: Any, message: str = "") -> None:
    if actual != expected:
        raise AssertionFail(f"{message}（期望 {expected!r}，实际 {actual!r}）")


def expect_in(needle: Any, haystack: Any, message: str = "") -> None:
    if needle not in haystack:
        raise AssertionFail(f"{message}（{needle!r} 不在 {haystack!r} 中）")


def expect_raises(exc_type: type[BaseException], func: Callable, *args, **kwargs) -> BaseException:
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        return exc
    except Exception as exc:  # noqa: BLE001
        raise AssertionFail(f"抛出了 {type(exc).__name__}，期望 {exc_type.__name__}") from exc
    raise AssertionFail(f"没有抛出 {exc_type.__name__}")


def run_module(module) -> tuple[int, int, list[str]]:
    """执行模块里所有 test_* 函数，返回 (通过数, 失败数, 失败详情)。"""

    passed = failed = 0
    failures: list[str] = []
    tests = sorted(
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj)
    )
    for name, func in tests:
        try:
            func()
            print(f"  ✓ {module.__name__}.{name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {module.__name__}.{name} — {type(exc).__name__}: {exc}")
            failures.append(f"{module.__name__}.{name}\n{traceback.format_exc()}")
    return passed, failed, failures
