"""FastAPI 应用入口。

启动：
    python run.py                       # 推荐
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import artifacts_router, chat_router, datasets_router, reports_router, system_router
from .config import PROJECT_ROOT, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("dataagent")

settings = get_settings()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

app = FastAPI(
    title=settings.APP_NAME,
    version=__version__,
    description=(
        "基于 LangGraph + Tool Calling 的数据分析 Agent："
        "上传 CSV/Excel 后，用自然语言完成数据预处理、统计计算、可视化与报告生成。"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, prefix="/api", tags=["system"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(reports_router, prefix="/api/reports", tags=["reports"])
app.include_router(artifacts_router, prefix="/api/artifacts", tags=["artifacts"])

# 内置前端（原生 HTML/CSS/JS，无需构建）
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """返回内置的极简前端页面。"""

    if FRONTEND_INDEX.exists():
        # 页面本身每次都向服务端确认，避免改了前端却还在用旧缓存
        return FileResponse(
            FRONTEND_INDEX,
            media_type="text/html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )
    return JSONResponse(
        {
            "message": f"{settings.APP_NAME} 已启动，但未找到 frontend/index.html。",
            "docs": "/docs",
            "health": "/api/health",
        }
    )


@app.on_event("startup")
def on_startup() -> None:
    logger.info("%s v%s 启动中…", settings.APP_NAME, __version__)
    logger.info("模型：%s @ %s（已配置：%s）", settings.LLM_MODEL, settings.LLM_BASE_URL, settings.llm_configured)
    logger.info("存储目录：%s", settings.storage_path)
    if not settings.llm_configured:
        logger.warning("未检测到可用的 LLM_API_KEY，请在 .env 中配置后重启。")
    # 提前编译图，避免首个请求承担编译开销
    from .graph.builder import get_graph

    get_graph()
    logger.info("LangGraph 工作流已编译，checkpoint 后端：%s", settings.checkpoint_db_path.name)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
