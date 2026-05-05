"""Pinecone document stores (chat vs uploaded docs) and chat memory helpers."""

from __future__ import annotations

from typing import Any

from loguru import logger
from haystack import Document
from haystack.components.embedders import OpenAITextEmbedder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone, ServerlessSpec

from hay_v2_bot.config import CHAT_HISTORY_NAMESPACE, DOCUMENTS_NAMESPACE, Settings


def _ensure_pinecone_index(settings: Settings) -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def init_stores(settings: Settings) -> tuple[PineconeDocumentStore, PineconeDocumentStore, OpenAITextEmbedder]:
    _ensure_pinecone_index(settings)
    chat_store = PineconeDocumentStore(
        index=settings.pinecone_index_name,
        namespace=CHAT_HISTORY_NAMESPACE,
        dimension=settings.pinecone_dimension,
    )
    documents_store = PineconeDocumentStore(
        index=settings.pinecone_index_name,
        namespace=DOCUMENTS_NAMESPACE,
        dimension=settings.pinecone_dimension,
    )
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_token(settings.openai_api_key),
        api_base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )
    return chat_store, documents_store, embedder


def get_user_history(chat_store: PineconeDocumentStore, user_id: str, limit: int = 5) -> list[ChatMessage]:
    filters: dict[str, Any] = {"field": "user_id", "operator": "==", "value": user_id}
    try:
        docs = chat_store.filter_documents(filters=filters)
    except Exception:
        return []
    history: list[ChatMessage] = []
    for doc in docs[-limit:]:
        user_input = (doc.meta or {}).get("user_input", "") or ""
        assistant_output = (doc.meta or {}).get("assistant_output", "") or ""
        history.append(ChatMessage.from_user(user_input))
        history.append(ChatMessage.from_assistant(assistant_output))
    return history


def save_interaction(
    chat_store: PineconeDocumentStore,
    embedder: OpenAITextEmbedder,
    user_id: str,
    user_input: str,
    assistant_output: str,
) -> None:
    logger.info(
        "[embed] сохранение в chat-history: запрос эмбеддинга для user_input (len={} симв.)",
        len(user_input or ""),
    )
    embedding = embedder.run(text=user_input)["embedding"]
    doc = Document(
        content=user_input,
        embedding=embedding,
        meta={
            "user_id": user_id,
            "user_input": user_input,
            "assistant_output": assistant_output,
        },
    )
    chat_store.write_documents([doc])


def pinecone_user_filter(user_id: str) -> dict[str, Any]:
    return {"field": "user_id", "operator": "==", "value": user_id}
