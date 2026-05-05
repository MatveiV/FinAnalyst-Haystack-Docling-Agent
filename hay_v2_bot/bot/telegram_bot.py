"""Telegram wiring: text → generation pipeline; files → Docling ingestion + summary."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import telebot
from loguru import logger
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.dataclasses import ChatMessage, ChatRole
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.context import (
    get_user_history,
    pinecone_user_filter,
    save_interaction,
)
from hay_v2_bot.components.tools import summarize_documents
from hay_v2_bot.config import Settings


def _format_dialog_for_prompt(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        text = (m.text or "").strip()
        if not text:
            continue
        if m.role == ChatRole.USER:
            lines.append(f"Пользователь: {text}")
        elif m.role == ChatRole.ASSISTANT:
            lines.append(f"Ассистент: {text}")
        else:
            lines.append(f"{m.role}: {text}")
    return "\n".join(lines) if lines else "Пользователь: (пусто)"


def _sanitize_filename(name: str) -> str:
    base = Path(name).name or "upload.bin"
    return re.sub(r'[<>:"/\\\\|?*]', "_", base)


def _clear_user_vectors(
    chat_store: PineconeDocumentStore,
    documents_store: PineconeDocumentStore,
    user_id: str,
) -> None:
    filt = pinecone_user_filter(user_id)
    for store in (chat_store, documents_store):
        try:
            docs = store.filter_documents(filters=filt)
        except Exception as e:
            logger.warning(f"filter_documents failed for {store}: {e}")
            continue
        ids = [d.id for d in docs if getattr(d, "id", None)]
        if not ids:
            continue
        try:
            store.delete_documents(document_ids=ids)
        except Exception as e:
            logger.warning(f"delete_documents failed for {store}: {e}")


def build_bot(
    settings: Settings,
    ingestion_pipeline: Pipeline,
    generation_pipeline: Pipeline,
    chat_store: PineconeDocumentStore,
    documents_store: PineconeDocumentStore,
    embedder: OpenAITextEmbedder,
) -> telebot.TeleBot:
    bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode=None)

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Привет! Я версия 2 бота на Haystack + Docling + Pinecone.\n"
            "Могу отвечать с учётом загруженных файлов, помню диалог в Pinecone, "
            "а также использовать финансовые факты, анализ графиков и веб-поиск.\n"
            "Отправь документ (PDF, DOCX, …) — я проиндексирую его и кратко резюмирую.\n"
            "Команда /clear удаляет твою историю и загруженные документы из векторного хранилища.",
        )

    @bot.message_handler(commands=["clear"])
    def clear_cmd(message: telebot.types.Message) -> None:
        uid = str(message.from_user.id)
        try:
            _clear_user_vectors(chat_store, documents_store, uid)
            bot.reply_to(message, "Готово: твоя история и документы удалены из Pinecone (для этого бота).")
        except Exception as e:
            logger.exception("clear failed")
            bot.reply_to(message, f"Не удалось очистить память: {e}")

    @bot.message_handler(content_types=["document"])
    def handle_document(message: telebot.types.Message) -> None:
        uid = str(message.from_user.id)
        doc = message.document
        if not doc:
            return
        orig_name = doc.file_name or "upload.bin"
        safe = _sanitize_filename(orig_name)
        bot.reply_to(
            message,
            "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…",
        )
        tmp_dir = Path(tempfile.mkdtemp(prefix="hay_v2_doc_"))
        local_path = tmp_dir / safe
        try:
            file_info = bot.get_file(doc.file_id)
            data = bot.download_file(file_info.file_path)
            local_path.write_bytes(data)
        except Exception as e:
            logger.exception("download failed")
            bot.reply_to(message, f"Не удалось скачать файл: {e}")
            return

        logger.info(
            "[docling] локальный разбор «{}» ({} байт): Docling запускает ML-модели "
            "(layout, таблицы, OCR) через PyTorch на вашей машине — без облачного API. "
            "Далее — чанки и удалённые эмбеддинги через OpenAI-совместимый API.",
            orig_name,
            len(data),
        )

        try:
            ingest_res = ingestion_pipeline.run(
                {
                    "converter": {
                        "sources": [str(local_path)],
                        "meta": {"user_id": uid, "filename": orig_name},
                    },
                    "metadata_enricher": {"filename": orig_name, "user_id": uid},
                }
            )
            written = (ingest_res.get("writer") or {}).get("documents_written", 0)
            chunks = (ingest_res.get("doc_embedder") or {}).get("documents") or (
                ingest_res.get("metadata_enricher") or {}
            ).get("documents")
            logger.info(
                "[embed] запрос эмбеддингов чанков через API: model={} base_url={} — записано чанков в Pinecone: {}",
                settings.embedding_model,
                settings.openai_base_url or "(default)",
                written,
            )
        except Exception as e:
            logger.exception("ingestion failed")
            bot.reply_to(message, f"Ошибка при обработке файла: {e}")
            return
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        bot.send_message(
            message.chat.id,
            "Готово. Я изучил этот файл, теперь можем его обсудить.",
        )
        try:
            summary = summarize_documents(list(chunks or []), settings)
            logger.info(
                "[llm] краткое резюме после индексации сформировано моделью {} (текст чанков → один ответ).",
                settings.chat_model,
            )
            bot.send_message(message.chat.id, summary)
        except Exception as e:
            logger.exception("summarize failed")
            bot.send_message(
                message.chat.id,
                f"Файл проиндексирован, но не удалось сформировать краткое резюме: {e}",
            )

    @bot.message_handler(func=lambda m: m.content_type == "text" and m.text and not m.text.startswith("/"))
    def handle_text(message: telebot.types.Message) -> None:
        uid = str(message.from_user.id)
        user_text = (message.text or "").strip()
        if not user_text:
            return
        logger.info(f"v2 message from {uid}: {user_text[:200]}")

        history = get_user_history(chat_store, uid)
        messages = history + [ChatMessage.from_user(user_text)]
        dialog_text = _format_dialog_for_prompt(messages)

        logger.info(
            "[embed] эмбеддинг пользовательского запроса (удалённый API): model={} len={} base_url={}",
            settings.embedding_model,
            len(user_text),
            settings.openai_base_url or "(default)",
        )
        try:
            gen_res = generation_pipeline.run(
                {
                    "query_embedder": {"text": user_text},
                    "retriever": {"filters": pinecone_user_filter(uid)},
                    "prompt_builder": {"dialog_text": dialog_text},
                    "answer_builder": {"query": user_text},
                }
            )
            answers = (gen_res.get("answer_builder") or {}).get("answers") or []
            if answers:
                ans0 = answers[0]
                response_content = getattr(ans0, "data", None) or getattr(ans0, "content", None) or ""
            else:
                ag_msgs = (gen_res.get("agent") or {}).get("messages") or []
                response_content = ""
                for m in reversed(ag_msgs):
                    if m.role == ChatRole.ASSISTANT and (m.text or "").strip():
                        response_content = m.text or ""
                        break
        except Exception as e:
            logger.exception("generation pipeline failed")
            bot.reply_to(message, f"Произошла ошибка: {e}")
            return

        ret_docs = (gen_res.get("retriever") or {}).get("documents") or []
        logger.info("[retrieve] Pinecone (cosine по векторам): получено чанков документов: {}", len(ret_docs))
        for i, d in enumerate(ret_docs[:5]):
            meta = d.meta or {}
            preview = ((d.content or "")[:160]).replace("\n", " ")
            logger.info(
                "[rag] фрагмент #{} в промпт: file={} page={} chunk={} | {!r}…",
                i,
                meta.get("filename"),
                meta.get("page_no"),
                meta.get("chunk_index"),
                preview,
            )
        logger.info(
            "[llm] ответ на основе промпта (в т.ч. извлечённых фрагментов): превью {!r}",
            (response_content or "")[:320],
        )

        try:
            data = json.loads(response_content)
            if isinstance(data, dict) and data.get("type") == "financial_chart_analysis":
                image_url = data.get("image_url")
                description = (data.get("description") or "")[:1024]
                bot.send_photo(message.chat.id, image_url, caption=description)
                save_interaction(chat_store, embedder, uid, user_text, description)
                return
            if isinstance(data, dict) and data.get("type") == "error":
                bot.reply_to(message, f"Ошибка при работе инструмента: {data.get('error', 'unknown')}")
                return
        except json.JSONDecodeError:
            pass

        save_interaction(chat_store, embedder, uid, user_text, response_content)
        bot.reply_to(message, response_content)

    return bot
