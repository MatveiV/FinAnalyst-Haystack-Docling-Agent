"""Central configuration from environment (ProxyAPI + Pinecone + Telegram)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index_name: str
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str | None
    serperdev_api_key: str
    embedding_model: str
    chat_model: str
    pinecone_dimension: int
    rag_top_k: int
    summary_batch_max_chars: int


def load_settings() -> Settings:
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required")
    openai_api_key = os.getenv("PROXY_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("PROXY_API_KEY (or OPENAI_API_KEY) is required for OpenAI-compatible API")
    telegram = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not telegram:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    serper = os.getenv("SERPERDEV_API_KEY", "")
    if not serper:
        raise ValueError("SERPERDEV_API_KEY is required for web search tool")

    return Settings(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "haystack-agent"),
        telegram_bot_token=telegram,
        openai_api_key=openai_api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        serperdev_api_key=serper,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        pinecone_dimension=int(os.getenv("VECTOR_DIMENSION", "1536")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        summary_batch_max_chars=int(os.getenv("SUMMARY_BATCH_MAX_CHARS", "12000")),
    )


CHAT_HISTORY_NAMESPACE = "chat-history"
DOCUMENTS_NAMESPACE = "documents"
