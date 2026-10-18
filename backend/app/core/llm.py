"""大模型工厂。

统一走 OpenAI 兼容协议（DeepSeek / Moonshot / 通义等均可），
只要在 .env 里改 ``LLM_BASE_URL`` 与 ``LLM_MODEL`` 即可切换。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from ..config import get_settings


class LLMNotConfiguredError(RuntimeError):
    """未配置可用的 API Key。"""


def llm_configured() -> bool:
    return get_settings().llm_configured


@lru_cache(maxsize=8)
def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """返回一个（带缓存的）聊天模型实例。

    ``streaming`` 只影响是否按 token 流式返回，同一个缓存键最多复用两个实例。
    """

    s = get_settings()
    if not s.llm_configured:
        raise LLMNotConfiguredError(
            "未检测到可用的 LLM_API_KEY，请在项目根目录的 .env 中配置后重试。"
        )

    return ChatOpenAI(
        model=s.LLM_MODEL,
        api_key=s.LLM_API_KEY,
        base_url=s.LLM_BASE_URL,
        temperature=s.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=s.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        timeout=s.LLM_TIMEOUT,
        max_retries=s.LLM_MAX_RETRIES,
        streaming=streaming,
    )


def llm_label() -> str:
    s = get_settings()
    return f"{s.LLM_MODEL} @ {s.LLM_BASE_URL}"
