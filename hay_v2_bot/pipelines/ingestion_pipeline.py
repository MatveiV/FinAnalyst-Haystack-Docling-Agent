"""Docling → metadata enrich → OpenAI document embeddings → Pinecone (documents namespace)."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

from hay_v2_bot.components.tools import MetadataEnricher, build_docling_chunker
from hay_v2_bot.config import Settings


def _load_docling_converter_types():
    try:
        from haystack_integrations.components.converters.docling import DoclingConverter, ExportType

        return DoclingConverter, ExportType
    except ModuleNotFoundError:
        pass
    try:
        from docling_haystack.converter import DoclingConverter, ExportType

        return DoclingConverter, ExportType
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Docling Haystack integration is not installed. Install with: pip install docling-haystack docling docling-core"
        ) from e


def build_ingestion_pipeline(documents_store: PineconeDocumentStore, settings: Settings) -> Pipeline:
    DoclingConverter, ExportType = _load_docling_converter_types()

    logger.info(
        "[docling] Сборка ingestion pipeline: DoclingConverter (локальный PyTorch OCR/layout) "
        "→ MetadataEnricher → OpenAIDocumentEmbedder (model={}, base_url={}) → Pinecone DocumentWriter.",
        settings.embedding_model,
        settings.openai_base_url or "(default OpenAI URL)",
    )

    converter = DoclingConverter(
        export_type=ExportType.DOC_CHUNKS,
        chunker=build_docling_chunker(),
    )
    enricher = MetadataEnricher()
    doc_embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_token(settings.openai_api_key),
        api_base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )
    writer = DocumentWriter(document_store=documents_store, policy=DuplicatePolicy.OVERWRITE)

    pipe = Pipeline()
    pipe.add_component("converter", converter)
    pipe.add_component("metadata_enricher", enricher)
    pipe.add_component("doc_embedder", doc_embedder)
    pipe.add_component("writer", writer)

    pipe.connect("converter.documents", "metadata_enricher.documents")
    pipe.connect("metadata_enricher.documents", "doc_embedder.documents")
    pipe.connect("doc_embedder.documents", "writer.documents")
    return pipe


def convert_to_chunks(
    sources: list[str],
    filename: str,
    user_id: str,
) -> list:
    """Run only Docling conversion + metadata enrichment locally, return plain Document list.

    Used to obtain chunk texts for summarization *before* embedding them into Pinecone.
    No network calls are made here — pure local PyTorch inference.
    """
    DoclingConverter, ExportType = _load_docling_converter_types()

    converter = DoclingConverter(
        export_type=ExportType.DOC_CHUNKS,
        chunker=build_docling_chunker(),
    )
    enricher = MetadataEnricher()

    conv_result = converter.run(
        sources=sources,
        meta={"user_id": user_id, "filename": filename},
    )
    raw_docs = conv_result.get("documents") or []
    logger.info("[docling] конвертация завершена: {} чанков до эмбеддинга", len(raw_docs))

    enrich_result = enricher.run(
        documents=raw_docs,
        filename=filename,
        user_id=user_id,
    )
    chunks = enrich_result.get("documents") or []
    logger.info("[docling] после MetadataEnricher: {} чанков готово для суммаризации", len(chunks))
    return chunks
