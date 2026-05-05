"""Query embedding → Pinecone retrieval → Chat prompt → Haystack Agent → AnswerBuilder."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.builders import AnswerBuilder
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.tools import LastChatMessageToReplyList, build_agent_tools
from hay_v2_bot.config import Settings


def _rag_user_template() -> list[ChatMessage]:
    return [
        ChatMessage.from_user(
            "{% if documents %}"
            "Ниже — наиболее релевантные фрагменты из документов, которые пользователь загрузил в чат. "
            "Опирайся на них, если они относятся к вопросу; если нет — игнорируй.\n"
            "{% for doc in documents %}"
            "---\n"
            "Файл: {{ doc.meta.get('filename', 'неизвестно') }}, страница: {{ doc.meta.get('page_no', '—') }}, "
            "чанк: {{ doc.meta.get('chunk_index', '—') }}\n"
            "{{ doc.content }}\n"
            "{% endfor %}"
            "{% endif %}\n"
            "История и текущий запрос:\n"
            "{{ dialog_text }}"
        ),
    ]


def build_generation_pipeline(documents_store: PineconeDocumentStore, settings: Settings) -> Pipeline:
    query_embedder = OpenAITextEmbedder(
        api_key=Secret.from_token(settings.openai_api_key),
        api_base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )
    retriever = PineconeEmbeddingRetriever(document_store=documents_store, top_k=settings.rag_top_k)
    prompt_builder = ChatPromptBuilder(
        template=_rag_user_template(),
        required_variables=["dialog_text"],
        variables=["dialog_text", "documents"],
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(
            model=settings.chat_model,
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
        ),
        tools=build_agent_tools(settings),
        system_prompt=(
            "Ты — умный персональный помощник. Ты можешь предоставлять финансовые факты, анализировать "
            "изображения финансовых графиков и выполнять веб-поиск. При анализе графиков извлекай тикер "
            "(TSLA, AAPL, MSFT и т.д.) из запроса пользователя. Всегда отвечай на русском языке."
        ),
        exit_conditions=["text"],
        max_agent_steps=10,
    )
    reply_extractor = LastChatMessageToReplyList()
    answer_builder = AnswerBuilder()

    pipe = Pipeline()
    pipe.add_component("query_embedder", query_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("agent", agent)
    pipe.add_component("reply_extractor", reply_extractor)
    pipe.add_component("answer_builder", answer_builder)

    pipe.connect("query_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "agent.messages")
    pipe.connect("agent.last_message", "reply_extractor.last_message")
    pipe.connect("reply_extractor.replies", "answer_builder.replies")
    pipe.connect("retriever.documents", "answer_builder.documents")
    return pipe
