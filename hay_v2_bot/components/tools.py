"""Haystack tools (financial fact, chart vision, web search), Docling metadata enricher, summarization."""

from __future__ import annotations

import json
import os
import requests
from typing import Any

import tiktoken
from loguru import logger
from openai import OpenAI

from haystack import Document, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack.components.websearch import SerperDevWebSearch

from hay_v2_bot.config import Settings


@component
class AlphaVantageFinancialFact:
    """Fetches a financial news item from Alpha Vantage (same API as v1)."""

    @component.output_types(fact=str)
    def run(self) -> dict[str, str]:
        t_logger = logger.bind(tool="financial_fact")
        t_logger.info("Fetching financial fact from Alpha Vantage")
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            if "feed" in data and len(data["feed"]) > 0:
                item = data["feed"][0]
                fact = f"Интересный факт из мира финансов: {item['title']}. {item['summary'][:200]}..."
                t_logger.debug(f"Successfully fetched fact: {item['title']}")
                return {"fact": fact}
            t_logger.warning("No news feed items found in Alpha Vantage response")
            return {"fact": "В данный момент свежих финансовых фактов нет."}
        except Exception as e:
            t_logger.error(f"Error fetching financial fact: {e}")
            return {"fact": f"Ошибка при получении факта: {e}"}


@component
class AlphaVantageRandomImage:
    """Fetches a Finviz chart and describes it with OpenAI Vision (ProxyAPI)."""

    @component.output_types(result=str)
    def run(self, symbol: str = "TSLA") -> dict[str, str]:
        if symbol == "AAPL":
            symbol = "TSLA"
        t_logger = logger.bind(tool="image_analysis")
        symbol = "".join(c for c in symbol if c.isalnum()) or "AAPL"
        image_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
        t_logger.info(f"Financial chart analysis for {symbol}")

        openai_api_key = os.getenv("PROXY_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        openai_base_url = os.getenv("OPENAI_BASE_URL") or None
        chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        prompt = (
            f"Ты - эксперт по финансовому анализу. Опиши этот график инструмента {symbol}. "
            "Что на нем изображено, текущий тренд, краткая предыстория этой информации и твоя рекомендация "
            "(buy/sell/hold) с обоснованием. Ответ сформируй в один связный текст."
        )
        try:
            response = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=800,
            )
            description = response.choices[0].message.content or ""
            payload = {
                "image_url": image_url,
                "description": description,
                "type": "financial_chart_analysis",
            }
            return {"result": json.dumps(payload)}
        except Exception as e:
            t_logger.error(f"Error during image analysis: {e}")
            return {"result": json.dumps({"error": str(e), "type": "error"})}


@component
class MetadataEnricher:
    """Adds filename, chunk_index, page_no, user_id on top of Docling dl_meta."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        filename: str = "",
        user_id: str = "",
    ) -> dict[str, list[Document]]:
        out: list[Document] = []
        for idx, doc in enumerate(documents):
            meta = dict(doc.meta or {})
            if filename:
                meta["filename"] = filename
            elif "filename" not in meta:
                meta["filename"] = meta.get("source_id") or meta.get("file_path") or "unknown"
            meta["chunk_index"] = idx
            page_no = _extract_page_no(meta)
            if page_no is not None:
                meta["page_no"] = page_no
            if user_id:
                meta["user_id"] = user_id
            out.append(Document(content=doc.content, meta=meta, embedding=doc.embedding))
        return {"documents": out}


def _extract_page_no(meta: dict[str, Any]) -> int | None:
    if meta.get("page_no") is not None:
        try:
            return int(meta["page_no"])
        except (TypeError, ValueError):
            pass
    dl = meta.get("dl_meta")
    if not dl:
        return None
    try:
        from docling.chunking import DocChunk

        chunk = DocChunk.model_validate(dl)
        if chunk.meta.doc_items:
            prov = chunk.meta.doc_items[0].prov
            if prov:
                return int(prov[0].page_no)
    except Exception:
        return None
    return None


def build_agent_tools(settings: Settings) -> list[ComponentTool]:
    fact = AlphaVantageFinancialFact()
    image = AlphaVantageRandomImage()
    search = SerperDevWebSearch(api_key=Secret.from_token(settings.serperdev_api_key))
    return [
        ComponentTool(component=fact),
        ComponentTool(component=image),
        ComponentTool(component=search),
    ]


def build_docling_chunker():
    """HybridChunker sized by OpenAI tokenizer (tiktoken only; no local embedding model)."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=8191)
    return HybridChunker(tokenizer=tokenizer)


def _last_chat_text(result: dict) -> str:
    replies = result.get("replies") or []
    if replies:
        last = replies[-1]
        if isinstance(last, ChatMessage):
            return (last.text or "").strip()
        return str(last).strip()
    lm = result.get("last_message")
    if lm is not None:
        return (lm.text or "").strip()
    return ""


def summarize_documents(chunks: list[Document], settings: Settings) -> str:
    """Map-reduce summarization to a single Russian sentence (uses ProxyAPI chat model)."""
    if not chunks:
        return "Файл обработан, но не удалось извлечь текстовое содержимое для краткого резюме."
    texts = [c.content or "" for c in chunks if (c.content or "").strip()]
    if not texts:
        return "Файл обработан, но текстовые фрагменты оказались пустыми."

    gen = OpenAIChatGenerator(
        model=settings.chat_model,
        api_key=Secret.from_token(settings.openai_api_key),
        api_base_url=settings.openai_base_url,
    )

    def one_sentence_from_text(block: str) -> str:
        messages = [
            ChatMessage.from_system(
                "Ты помощник по сжатию текста. Ответь ровно одним полным предложением на русском языке."
            ),
            ChatMessage.from_user(
                "Сформулируй одно ёмкое предложение, передающее суть следующего фрагмента документа:\n\n"
                + block
            ),
        ]
        return _last_chat_text(gen.run(messages=messages))

    joined = "\n\n---\n\n".join(texts)
    if len(joined) <= settings.summary_batch_max_chars:
        return one_sentence_from_text(joined)

    batch_summaries: list[str] = []
    buf: list[str] = []
    size = 0
    for t in texts:
        add = len(t) + 4
        if size + add > settings.summary_batch_max_chars and buf:
            batch_summaries.append(one_sentence_from_text("\n\n".join(buf)))
            buf = [t]
            size = len(t)
        else:
            buf.append(t)
            size += add
    if buf:
        batch_summaries.append(one_sentence_from_text("\n\n".join(buf)))

    merged = "\n".join(f"- {s}" for s in batch_summaries)
    messages = [
        ChatMessage.from_system(
            "Ты редактор. На основе списка кратких тезисов напиши ровно одно связное предложение на русском."
        ),
        ChatMessage.from_user(merged),
    ]
    out = _last_chat_text(gen.run(messages=messages))
    return out or "Файл успешно проиндексирован."


@component
class LastChatMessageToReplyList:
    """Maps Agent `last_message` to AnswerBuilder `replies` (list of strings)."""

    @component.output_types(replies=list[str])
    def run(self, last_message: ChatMessage) -> dict[str, list[str]]:
        return {"replies": [last_message.text or ""]}
