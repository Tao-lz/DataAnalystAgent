"""一键启动脚本。

    python run.py                 # 启动服务并自动打开浏览器
    python run.py --no-browser    # 只启动服务
    python run.py --port 8080     # 换端口
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="启动数据分析 Agent 服务")
    parser.add_argument("--host", default=None, help="监听地址，默认取 .env 的 APP_HOST")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认取 .env 的 APP_PORT")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开启热重载（开发用）")
    args = parser.parse_args()

    settings = get_settings()
    host = args.host or settings.APP_HOST
    port = args.port or settings.APP_PORT
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"

    print("=" * 68)
    print(f"  {settings.APP_NAME} v1.0.0")
    print(f"  模型      : {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
    print(f"  模型已配置: {settings.llm_configured}")
    print(f"  控制台    : {url}")
    print(f"  接口文档  : {url}/docs")
    print(f"  存储目录  : {settings.storage_path}")
    print("=" * 68)
    if not settings.llm_configured:
        print("!! 未检测到可用的 LLM_API_KEY，请先在 .env 中配置，否则对话会失败。\n")

    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
