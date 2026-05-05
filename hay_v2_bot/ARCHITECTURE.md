# Архитектура hay_v2_bot

## Диаграммы C4 и UML (Mermaid)

Ниже — нотации **C4** (контекст и контейнеры) и **UML** (диаграммы последовательностей) для `hay_v2_bot`. Рендер: GitHub, VS Code с Mermaid, многие статические генераторы документации.

### C4: контекст системы (System Context)

Показывает `hay_v2_bot` среди внешних систем и пользователя.

```mermaid
%%{init: {'theme': 'default'}}%%
C4Context
    title SystemContext — hay_v2_bot и окружение

    Person(user, "Пользователь", "Отправляет текст и файлы в Telegram")
    System(v2bot, "hay_v2_bot", "Telegram-бот: Haystack пайплайны, Docling, RAG, Pinecone")
    System_Ext(telegram, "TelegramBotAPI", "Доставка сообщений и файлов")
    System_Ext(openai, "OpenAICompatibleAPI", "Эмбеддинги и LLM через OPENAI_BASE_URL")
    System_Ext(pinecone, "Pinecone", "Векторное хранилище индекса PINECONE_INDEX_NAME")
    System_Ext(serper, "SerperDev", "Веб-поиск")
    System_Ext(alpha, "AlphaVantage", "Финансовые новости")
    System_Ext(finviz, "Finviz", "URL графиков для Vision-анализа")

    Rel(user, telegram, "HTTPS, чат")
    Rel(telegram, v2bot, "long polling, updates")
    Rel(v2bot, openai, "HTTPS, embeddings и chat")
    Rel(v2bot, pinecone, "HTTPS, векторы")
    Rel(v2bot, serper, "HTTPS, веб-поиск")
    Rel(v2bot, alpha, "HTTPS, финновости")
    Rel(v2bot, finviz, "HTTPS, URL графика")
```

### C4: контейнеры внутри hay_v2_bot (Container)

Логические части приложения (каталоги и пайплайны), граница системы — один процесс Python.

```mermaid
%%{init: {'theme': 'default'}}%%
C4Container
    title ContainerDiagram — внутренняя структура hay_v2_bot

    Person(user, "Пользователь", "Telegram")

    System_Boundary(boundary, "hay_v2_bot процесс") {
        Container(main, "main.py", "Python", "Точка входа, логирование, сборка зависимостей")
        Container(tg, "bot/telegram_bot.py", "Python, pyTelegramBotAPI", "Хендлеры: текст, документ, /clear")
        Container(ing, "pipelines/ingestion_pipeline.py", "Haystack Pipeline", "Docling → enrich → embed → Pinecone documents")
        Container(gen, "pipelines/generation_pipeline.py", "Haystack Pipeline", "embed query → retrieve → prompt → Agent → Answer")
        Container(ctx, "components/context.py", "Python", "PineconeDocumentStore chat-history + documents")
        Container(tools, "components/tools.py", "Haystack @component", "Инструменты агента, MetadataEnricher, суммаризация")
    }

    System_Ext(telegram, "TelegramAPI", "Внешний API")
    System_Ext(openai, "OpenAICompatibleAPI", "ProxyAPI или OpenAI")
    System_Ext(pinecone, "Pinecone", "Облачный индекс")

    Rel(user, telegram, "HTTPS, чат")
    Rel(telegram, tg, "HTTPS, updates")
    Rel(main, tg, "инициализирует")
    Rel(main, ing, "build_ingestion_pipeline")
    Rel(main, gen, "build_generation_pipeline")
    Rel(tg, ing, "run при файле")
    Rel(tg, gen, "run при тексте")
    Rel(tg, ctx, "история и clear")
    Rel(ing, ctx, "DocumentWriter documents namespace")
    Rel(gen, ctx, "Retriever documents namespace")
    Rel(gen, tools, "ComponentTool из build_agent_tools")
    Rel(ing, openai, "OpenAIDocumentEmbedder")
    Rel(gen, openai, "OpenAITextEmbedder, OpenAIChatGenerator")
    Rel(ctx, pinecone, "read/write")
```

### UML: последовательность — загрузка файла, индексация, резюме

```mermaid
%%{init: {'theme': 'default'}}%%
sequenceDiagram
    autonumber
    participant User as Пользователь
    participant TG as TelegramAPI
    participant Bot as telegram_bot
    participant Pipe as ingestion_pipeline
    participant DL as DoclingConverter
    participant En as MetadataEnricher
    participant Emb as OpenAIDocumentEmbedder
    participant Pine as Pinecone
    participant Sum as summarize_documents

    User->>TG: документ PDF/DOCX
    TG->>Bot: message document
    Bot->>User: Файл получен…
    Bot->>Bot: download_file
    Bot->>Pipe: run(converter, enricher)
    Pipe->>DL: sources локально
    Note over DL: PyTorch-модели Docling layout/OCR<br/>инференс на вашей машине (CPU/GPU)
    DL-->>En: documents chunks
    En-->>Emb: documents + meta
    Emb->>Pine: HTTP embeddings API
    Emb-->>Pipe: documents с векторами
    Pipe->>Pine: DocumentWriter documents
    Pipe-->>Bot: documents_written
    Bot->>User: Готово…
    Bot->>Sum: чанки текста
    Note over Sum: Вызовы LLM по API без Pinecone
    Sum-->>Bot: одно предложение
    Bot->>User: резюме
```

### UML: последовательность — текстовый вопрос с RAG

```mermaid
%%{init: {'theme': 'default'}}%%
sequenceDiagram
    autonumber
    participant User as Пользователь
    participant TG as TelegramAPI
    participant Bot as telegram_bot
    participant Ctx as context Pinecone
    participant Gen as generation_pipeline
    participant QEmb as OpenAITextEmbedder
    participant Ret as PineconeEmbeddingRetriever
    participant PB as ChatPromptBuilder
    participant Ag as Agent
    participant OAI as OpenAICompatibleAPI
    participant Pine as Pinecone

    User->>TG: текст вопроса
    TG->>Bot: message text
    Bot->>Ctx: get_user_history
    Ctx->>Pine: filter chat-history
    Ctx-->>Bot: ChatMessage list
    Bot->>Gen: run
    Gen->>QEmb: text query
    QEmb->>OAI: embeddings
    OAI-->>QEmb: vector
    QEmb-->>Ret: query_embedding
    Ret->>Pine: similarity search documents
    Pine-->>Ret: top_k chunks
    Ret-->>PB: documents
    Gen->>PB: dialog_text в run
    PB-->>Ag: messages с контекстом чанков
    Ag->>OAI: chat completions tools
    OAI-->>Ag: ответ
    Ag-->>Gen: last_message
    Gen-->>Bot: answers
    Bot->>Ctx: save_interaction embed user
    Bot->>User: ответ
```

### UML: компоненты модулей (упрощённый component view)

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart TB
    subgraph entry [Запуск]
        M[main.py]
    end
    subgraph cfg [Конфигурация]
        CF[config.py Settings]
    end
    subgraph comp [components]
        CT[context.py]
        TL[tools.py]
    end
    subgraph pipe [pipelines]
        IP[ingestion_pipeline.py]
        GP[generation_pipeline.py]
    end
    subgraph ui [bot]
        TB[telegram_bot.py]
    end
    M --> CF
    M --> CT
    M --> IP
    M --> GP
    M --> TB
    TB --> CT
    TB --> IP
    TB --> GP
    IP --> TL
    GP --> TL
```

## Роль Haystack Pipeline

**Pipeline** в Haystack 2 — это ориентированный граф компонентов (`@component`). У каждого компонента есть метод `run()` с именованными входами и выходами. `pipeline.connect("A.out", "B.in")` связывает выход одного шага с входом следующего. Вызов `pipeline.run({...})` передаёт внешние входы (например, текст запроса или фильтры) и исполняет компоненты в топологическом порядке.

Преимущества: явный поток данных, повторное использование компонентов, проще логировать и тестировать отдельные шаги.

## Pinecone: два namespace в одном индексе

- `chat-history` — фрагменты диалога (эмбеддинг по тексту пользователя, метаданные `user_id`, `user_input`, `assistant_output`).
- `documents` — чанки из Docling (эмбеддинг контента чанка, метаданные `user_id`, `filename`, `chunk_index`, `page_no`, `dl_meta`).

Retriever в generation-пайплайне читает **только** `documents` с фильтром по `user_id`, чтобы пользователь не видел чужие файлы.

## Ingestion pipeline

```mermaid
%%{init: {'theme': 'default'}}%%
graph LR
    sources[sources + meta] --> Docling[DoclingConverter DOC_CHUNKS]
    Docling --> Enrich[MetadataEnricher]
    Enrich --> Emb[OpenAIDocumentEmbedder ProxyAPI]
    Emb --> Writer[DocumentWriter Pinecone documents]
```

- **DoclingConverter** (`docling-haystack`): парсит файл/URL в структуру Docling и при `ExportType.DOC_CHUNKS` нарезает на осмысленные чанки. Чанкер — `HybridChunker` с **`OpenAITokenizer` + `tiktoken`**: это только подсчёт токенов для лимита размера чанка, **без локального эмбеддинга**.
- **MetadataEnricher**: добавляет `filename`, `chunk_index`, `page_no` (если удаётся извлечь из `dl_meta` через `DocChunk`), `user_id`.
- **OpenAIDocumentEmbedder**: векторизация через ваш `OPENAI_BASE_URL` и `text-embedding-3-small`.
- **DocumentWriter**: запись в `PineconeDocumentStore` (namespace `documents`, политика overwrite).

Импорт `DoclingConverter` выполняется внутри `build_ingestion_pipeline()` и поддерживает оба распространённых пути пакета: `haystack_integrations...` и `docling_haystack...`.

## Generation pipeline

```mermaid
%%{init: {'theme': 'default'}}%%
graph LR
    Q[text user query] --> TE[OpenAITextEmbedder]
    TE --> R[PineconeEmbeddingRetriever filters user_id]
    R --> PB[ChatPromptBuilder Jinja documents + dialog_text]
    PB --> AG[Agent OpenAIChatGenerator + tools]
    AG --> EX[LastChatMessageToReplyList]
    EX --> AB[AnswerBuilder]
    R --> AB
```

- **OpenAITextEmbedder**: эмбеддинг последнего пользовательского сообщения для семантического поиска по чанкам.
- **PineconeEmbeddingRetriever**: `query_embedding` + `filters` по `user_id`.
- **ChatPromptBuilder**: одно пользовательское сообщение-шаблон с Jinja: вставляет найденные `documents` и строку `dialog_text` (история + текущий запрос в текстовом виде).
- **Agent**: те же три инструмента, что в v1 (Alpha Vantage, Finviz+Vision, SerperDev). `exit_conditions=["text"]`, ограничение шагов.
- **LastChatMessageToReplyList**: берёт стандартный выход агента `last_message` и превращает его в `replies` (список строк) для AnswerBuilder.
- **AnswerBuilder**: собирает `GeneratedAnswer` с текстом и привязкой к документам-источникам.

## Поток в Telegram

1. **Текст**: `get_user_history` → список `ChatMessage` → `dialog_text` → `generation_pipeline.run(...)`. Ответ берётся из `answer_builder.answers[0]` (с запасным вариантом по `agent.messages`). Специальный случай JSON от инструмента графика — как в v1 (фото + caption).
2. **Файл**: скачивание во временную папку → `ingestion_pipeline.run` с `sources` и `meta` → очистка temp → сообщение «Готово…» → `summarize_documents` по чанкам из выхода `doc_embedder` (или `metadata_enricher`), с map-reduce при большом объёме текста.

### Логи в терминале

Все ключевые этапы обработки логируются с префиксами, что позволяет видеть обращения к векторным вложениям в реальном времени:

| Префикс | Этап | Где выполняется |
|---------|------|-----------------|
| `[torch]` | Проверка PyTorch и GPU при старте | Локально |
| `[docling]` | Разбор файла: layout, таблицы, OCR | Локально (PyTorch) |
| `[embed]` | Запрос эмбеддингов чанков или запроса | Удалённо (OpenAI API) |
| `[retrieve]` | Поиск по векторам в Pinecone | Удалённо (Pinecone) |
| `[rag]` | Фрагменты документов в промпт | Локально (логирование) |
| `[llm]` | Ответ языковой модели | Удалённо (OpenAI API) |

## Зависимости Docling и модели

### Локальный инференс через PyTorch

Docling для разбора PDF/DOCX подтягивает **свои** модели распознавания layout/таблиц/OCR и выполняет инференс **локально** через **PyTorch**. Это не облачный «Docling API»: конвертация идёт на вашей машине.

**Компоненты локальной обработки:**
- `docling-ibm-models` — веса моделей layout и таблиц (загружаются при первом запуске в `~/.cache/docling/`)
- `easyocr` или встроенный OCR Docling — распознавание текста на изображениях
- `PyTorch` — фреймворк для инференса всех ML-моделей
- `HybridChunker` + `tiktoken` — нарезка на чанки (только подсчёт токенов, без локального эмбеддинга)

**При старте бота в терминале:**
```
[torch] PyTorch 2.x.x доступен, устройство: CPU
[docling] Docling доступен. Модели layout/OCR будут загружены при первом обращении к файлу
```

**При загрузке файла:**
```
[docling] локальный разбор «report.pdf» (2048000 байт): Docling запускает ML-модели
          (layout, таблицы, OCR) через PyTorch на вашей машине — без облачного API.
```

### Векторные эмбеддинги (удалённый API)

**Эмбеддинги для Pinecone** в этом проекте считаются **не** локальным PyTorch, а запросами к **OpenAI-совместимому** эндпоинту (`OpenAIDocumentEmbedder` / `OpenAITextEmbedder` через `OPENAI_BASE_URL`, например ProxyAPI).

**В терминале видны все обращения к векторным вложениям:**
```
[embed] запрос эмбеддингов чанков через API: model=text-embedding-3-small base_url=https://... — записано чанков в Pinecone: 42
[embed] эмбеддинг пользовательского запроса (удалённый API): model=text-embedding-3-small len=156
[embed] сохранение в chat-history: запрос эмбеддинга для user_input (len=156 симв.)
[retrieve] Pinecone (cosine по векторам): получено чанков документов: 5
[rag] фрагмент #0 в промпт: file=report.pdf page=3 chunk=12 | "Текст фрагмента..."
[llm] ответ на основе промпта (в т.ч. извлечённых фрагментов): превью "Ответ модели..."
```

Чат-память в namespace `chat-history` при сохранении тоже вызывает эмбеддинг пользовательской реплики через тот же API.

## Файлы

| Файл | Назначение |
|------|------------|
| [config.py](config.py) | `Settings`, константы namespace |
| [components/context.py](components/context.py) | Инициализация Pinecone + история |
| [components/tools.py](components/tools.py) | Инструменты, `MetadataEnricher`, суммаризация, мост к AnswerBuilder |
| [pipelines/ingestion_pipeline.py](pipelines/ingestion_pipeline.py) | Сборка ingestion Pipeline |
| [pipelines/generation_pipeline.py](pipelines/generation_pipeline.py) | Сборка generation Pipeline |
| [bot/telegram_bot.py](bot/telegram_bot.py) | Обработчики Telegram |
| [main.py](main.py) | Точка входа |
