"""HTTP 路由。"""

from .artifacts import router as artifacts_router
from .chat import router as chat_router
from .datasets import router as datasets_router
from .reports import router as reports_router
from .system import router as system_router

__all__ = [
    "artifacts_router",
    "chat_router",
    "datasets_router",
    "reports_router",
    "system_router",
]
