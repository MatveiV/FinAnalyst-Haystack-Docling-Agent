"""Entry point for hay_v2_bot (run from repository root: python -m hay_v2_bot.main)."""

from __future__ import annotations

import sys

from loguru import logger

from hay_v2_bot.bot.telegram_bot import build_bot
from hay_v2_bot.components.context import init_stores
from hay_v2_bot.config import load_settings
from hay_v2_bot.pipelines.generation_pipeline import build_generation_pipeline
from hay_v2_bot.pipelines.ingestion_pipeline import build_ingestion_pipeline


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add("logs/app.log", rotation="500 MB", level="INFO")
    logger.add("logs/tools.log", filter=lambda record: "tool" in record["extra"], level="DEBUG")

    settings = load_settings()

    # ── PyTorch / Docling ML-модели (локальный инференс) ──────────────────────
    try:
        import torch

        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        logger.info(
            "[torch] PyTorch {} доступен, устройство: {}. "
            "Docling использует локальные ML-модели (layout, таблицы, OCR) через PyTorch — "
            "инференс выполняется на вашей машине без облачного API.",
            torch.__version__,
            device,
        )
    except ImportError as exc:  # pragma: no cover
        logger.warning(
            "[torch] torch не найден ({}). Docling подтянет свои ML-зависимости при первой конвертации.",
            exc,
        )

    try:
        import docling as _docling_pkg

        logger.info(
            "[docling] Docling {} доступен. Модели layout/OCR будут загружены при первом обращении к файлу "
            "(кэшируются локально в ~/.cache/docling или аналогичном пути).",
            getattr(_docling_pkg, "__version__", "?"),
        )
    except ImportError:  # pragma: no cover
        logger.warning("[docling] Пакет docling не найден — установите: pip install docling docling-haystack docling-core")

    # ── Векторные эмбеддинги (удалённый OpenAI-совместимый API) ──────────────
    logger.info(
        "[embed] Векторизация текста (эмбеддинги для Pinecone и запросов) идёт через OpenAI-совместимый API: "
        "model={} base_url={}.",
        settings.embedding_model,
        settings.openai_base_url or "(default OpenAI URL)",
    )
    logger.info(
        "[embed] В терминале будут видны логи: "
        "[docling] — локальный разбор файла, "
        "[embed] — запросы к API эмбеддингов, "
        "[retrieve] — поиск по Pinecone, "
        "[rag] — фрагменты в промпт, "
        "[llm] — ответ модели."
    )

    chat_store, documents_store, embedder = init_stores(settings)
    ingestion = build_ingestion_pipeline(documents_store, settings)
    generation = build_generation_pipeline(documents_store, settings)

    bot = build_bot(
        settings=settings,
        ingestion_pipeline=ingestion,
        generation_pipeline=generation,
        chat_store=chat_store,
        documents_store=documents_store,
        embedder=embedder,
    )
    logger.info("hay_v2_bot polling started")
    bot.infinity_polling()


if __name__ == "__main__":
    main()
