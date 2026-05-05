# hay_v2_bot — Haystack 2 + Docling + Pinecone (v2 Telegram)

Модульная вторая версия бота: пайплайны Haystack для **ингеста** (Docling → OpenAI embeddings → Pinecone) и **генерации** (embedding запроса → retrieval → ChatPromptBuilder → Agent с инструментами → AnswerBuilder). Все вызовы OpenAI (чат и эмбеддинги) идут через `OPENAI_BASE_URL` (например, ProxyAPI).

## Возможности

- Текстовые ответы с **RAG** по загруженным файлам (namespace Pinecone `documents`, фильтр по `user_id`).
- **Инструменты** как в v1: финансовые новости Alpha Vantage, анализ графика Finviz + Vision через ваш прокси-чат, веб-поиск SerperDev.
- **История** в Pinecone (namespace `chat-history`), как в [hay/hay-tg_bot.py](../hay/hay-tg_bot.py), с полем `assistant_output` для восстановления диалога.
- Загрузка **документов** (PDF, DOCX, PPTX, XLSX, HTML, Markdown и др. по возможностям Docling): сообщение о приёме, индексация, затем «Готово…» и **одно предложение**-резюме (map-reduce по чанкам через чат-модель).
- Команда `/clear` — удаление из Pinecone истории и документов пользователя (по `user_id`).

## Зависимости

Все пакеты (`docling`, `haystack-ai`, `torch`, `loguru` и др.) устанавливаются в системный Python. `venv` в репозитории создан с флагом `--system-site-packages` и наследует их автоматически.

Из корня репозитория:

```bash
pip install -r requirements.txt
```

Добавлены пакеты: `docling`, `docling-haystack`, `docling-core`, `tiktoken`, `torch`.

> **Примечание:** если вы используете `venv` и получаете `ModuleNotFoundError`, пересоздайте его с флагом `--system-site-packages`:
> ```bash
> python -m venv --system-site-packages venv
> ```
> Это позволит `venv` видеть все пакеты, установленные в системный Python.

## Переменные окружения

Используйте тот же `.env`, что и для v1 (см. корневой [README.md](../README.md)). Обязательно:

- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (размерность индекса **1536** под `text-embedding-3-small`).
- `PROXY_API_KEY` и `OPENAI_BASE_URL` — доступ к OpenAI-совместимому API (ProxyAPI).
- `TELEGRAM_BOT_TOKEN`
- `SERPERDEV_API_KEY`
- `ALPHAVANTAGE_API_KEY` (можно `demo`)

Опционально: `EMBEDDING_MODEL` (по умолчанию `text-embedding-3-small`), `CHAT_MODEL`, `RAG_TOP_K`, `SUMMARY_BATCH_MAX_CHARS`.

## Запуск

Активируйте окружение и запустите из корня репозитория:

```bash
# Windows
.\venv\Scripts\activate
python -m hay_v2_bot.main

# Linux / macOS
source venv/bin/activate
python -m hay_v2_bot.main
```

Или напрямую через системный Python (если все пакеты установлены глобально):

```bash
python -m hay_v2_bot.main
```

Логи: `logs/app.log`, `logs/tools.log` (как в v1).

## Где что выполняется (локально / по сети)

### Локальная обработка (PyTorch + Docling)

**Разбор загруженного файла выполняется полностью локально:**
- **DoclingConverter** использует ML-модели для распознавания layout, таблиц и OCR
- Модели работают через **PyTorch** (CPU или GPU, если доступна CUDA)
- Модели загружаются автоматически при первом запуске и кэшируются в `~/.cache/docling/`
- **Никаких облачных API** для OCR не используется — всё на вашей машине
- Чанкование документа также происходит локально через `HybridChunker` с `tiktoken`

**В терминале при запуске вы увидите:**
```
[torch] PyTorch 2.x.x доступен, устройство: CPU (или GPU (CUDA))
[docling] Docling доступен. Модели layout/OCR будут загружены при первом обращении к файлу
```

### Удалённая обработка (OpenAI-совместимый API + Pinecone)

**По сети выполняются только векторные операции и LLM:**
- **Эмбеддинги чанков документов** — через `OpenAIDocumentEmbedder` (model: `text-embedding-3-small`)
- **Эмбеддинги запросов пользователя** — через `OpenAITextEmbedder`
- **Чат и резюме файла** — через `OpenAIChatGenerator` (model: `CHAT_MODEL`, по умолчанию `gpt-4o-mini`)
- **Запись/поиск векторов** — в облачном индексе **Pinecone**

Все обращения к OpenAI API идут через ваш `OPENAI_BASE_URL` (например, ProxyAPI для экономии).

### Видимость в терминале

**При работе бота в терминале отображаются детальные логи всех этапов:**

```
[docling] локальный разбор «document.pdf» (1234567 байт): Docling запускает ML-модели...
[embed] запрос эмбеддингов чанков через API: model=text-embedding-3-small — записано чанков в Pinecone: 42
[embed] эмбеддинг пользовательского запроса (удалённый API): model=text-embedding-3-small len=156
[retrieve] Pinecone (cosine по векторам): получено чанков документов: 5
[rag] фрагмент #0 в промпт: file=document.pdf page=3 chunk=12 | "Текст фрагмента..."
[llm] ответ на основе промпта (в т.ч. извлечённых фрагментов): превью "Ответ модели..."
```

**Префиксы логов:**
- `[torch]` — информация о PyTorch и доступности GPU
- `[docling]` — локальная обработка файла (OCR, layout, таблицы)
- `[embed]` — запросы к API эмбеддингов (удалённо)
- `[retrieve]` — поиск по векторам в Pinecone (удалённо)
- `[rag]` — фрагменты документов, попавшие в промпт
- `[llm]` — ответ языковой модели

Это позволяет в реальном времени видеть, когда происходят обращения к векторным вложениям и какие данные используются для RAG.

## Отличия от v1

| | v1 `hay/hay-tg_bot.py` | v2 `hay_v2_bot` |
|---|---|---|
| Архитектура | один файл | `components/`, `pipelines/`, `bot/` |
| Документы | нет | Docling + namespace `documents` |
| Ответ | прямой `Agent.run` | `Pipeline`: retriever + prompt + agent + answer builder |
| Эмбеддинги | OpenAI через ProxyAPI | то же самое |

Старый бот **не изменён**; можно запускать оба параллельно (рекомендуется **разный** `TELEGRAM_BOT_TOKEN` или один бот — только одна версия).

## Поддерживаемые форматы файлов

Определяются установленной версией Docling (типично PDF, DOCX, PPTX, XLSX, HTML, Markdown, изображения с OCR). Неизвестный формат приведёт к ошибке конвертера — пользователь увидит сообщение об ошибке.

## Документация по архитектуре

См. [ARCHITECTURE.md](ARCHITECTURE.md) — там в том числе:

- **C4 (Mermaid):** диаграмма контекста системы (System Context) и контейнеров (Container) для `hay_v2_bot` и внешних зависимостей.
- **UML (Mermaid):** диаграммы последовательностей (sequence) для сценария загрузки файла с резюме и для текстового вопроса с RAG; упрощённый component view модулей.

### Контрольный список требований

1. **Новая версия бота** — каталог `hay_v2_bot/`: модули `components/`, `pipelines/`, `bot/`, вход [`main.py`](main.py).
2. **Docling** — [`pipelines/ingestion_pipeline.py`](pipelines/ingestion_pipeline.py): `DoclingConverter`, `DOC_CHUNKS`, `HybridChunker` + `tiktoken`.
3. **Pinecone** — два namespace в [`components/context.py`](components/context.py): `documents` и `chat-history`.
4. **Резюме и RAG** — после файла: [`summarize_documents`](components/tools.py); на вопросы: [`generation_pipeline.py`](pipelines/generation_pipeline.py) + [`telegram_bot.py`](bot/telegram_bot.py).
5. **C4 и UML** — раздел «Диаграммы C4 и UML» в [ARCHITECTURE.md](ARCHITECTURE.md).
