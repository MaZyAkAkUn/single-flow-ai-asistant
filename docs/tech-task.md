# Project: Single-Flow Personal Assistant (PyQt6 + LangChain)

## Summary / Goal

Create a cross-platform desktop personal assistant with a single continuous conversation flow (no separate chats). UI built in PyQt6. All AI interactions are routed through LangChain (use OpenRouter/compatible LLM wrappers), with vector DB retrieval for “chat-with-files,” TTS and ASR provider selection, and a LangChain-backed memory system (preferably Memori integration). Provide a robust prompt system, settings UI, secure API-key management, offline options, and modular architecture so components can be swapped (DB, TTS/ASR providers, LLMs).

---

## High-level architecture

* **Frontend:** PyQt6 application (single-window, responsive layout). Components:

  * Message list (continuous timeline)
  * Input bar (text + voice input)
  * Settings modal
  * Files panel (drag & drop to upload)
  * Conversation controls (regenerate, clear last, pin message, etc.)
  * Playback controls for TTS
* **Core logic (local process):**

  * `AI Controller` — central orchestrator that receives user input, assembles prompt, queries LangChain chains, handles memory retrieval/insertion, returns assistant response.
  * `LangChain Layer` — all LLM, TTS, ASR, and retriever calls go through LangChain connectors/chains.
  * `Memory Adapter` — integrates Memori (or LangChain memory interface) for longterm/episodic/working memory.
  * `Vector Store` — local vector DB (FAISS / Chroma / Milvus abstracted via LangChain retriever).
  * `File Ingestion` — file parsing, text extraction, chunking, embedding ingestion.
  * `Audio Subsystem` — ASR (voice -> text) and TTS (text -> audio playback or file), provider-agnostic.
  * `Settings & Secrets` — encrypted local store for API keys and preferences.
  * `Persistence` — local DB (SQLite) for metadata, logs, messages; vector store for embeddings.
  * `Plugins` (optional) — small plugin API for third-party features (calendar, TODOs).
* **Packaging/Distribution:** single binary via PyInstaller or cross-platform packaging (Windows/Ubuntu/macOS).

---

## Key features (priority ordered)

1. **Single-flow chat UI** — continuous conversation timeline (user/assistant messages, files, audio nodes).
2. **LangChain-first AI integration** — all LLM/TTS/ASR/embeddings are used via LangChain connectors and chains.
3. **Multi-LLM support via OpenRouter** — easily switch LLM provider in settings (model selection).
4. **Vector DB & retrieval (chat-with-files)** — upload file(s) → parse → chunk → embed → store → retrieve during prompt building.
5. **Memory system** — integrate Memori for long-term/episodic memory + LangChain memory bridging.
6. **TTS & ASR provider selection** — choose provider per feature; OpenAI default for TTS, plus e.g. Azure, local TTS (optional).
7. **Advanced prompt system** — editable system prompt, injection of retrieved docs & memories, prompt templating and presets.
8. **Privacy & Security** — local encryption for keys, opt-in telemetry, explicit UI for provider privacy policies.
9. **Settings & profiles** — API keys, default model, default vector DB, TTS/ASR providers, memory toggles.
10. **Offline mode** (best-effort) — local embeddings + local LLMs or offline embeddings if available.
11. **Testing & CI** — unit tests for core flows, integration tests for LangChain chains (mocked).
12. **Logging & debug** — toggleable verbose logs for chain calls, retriever traces, and prompt history.

---

## Non-functional requirements

* Cross-platform: Win/macOS/Linux.
* Single-process desktop app (no mandatory external server).
* Modular code: clear separation between UI, orchestration, providers, and persistence.
* Offline-first where possible; explicit warnings when network providers used.
* Secure: API keys encrypted at rest with a user passphrase / OS keyring.
* Extensible: easy to add new provider connectors.

---

## Data & storage

* **Messages & metadata:** SQLite (or tiny local DB) with schema: messages(id, role, content, timestamp, attachments, embedding_ref).
* **Vector store:** configurable (default: FAISS or Chroma local folder). Indexed embeddings stored on disk.
* **Files:** stored under `~/.appname/files/` with hashes, parsed text, last-ingestion status.
* **Memory:** Memori or LangChain memory adapter. Memory items include tags, timestamps, source, importance score.
* **Config/secrets:** encrypted JSON (using user passphrase or OS keyring fallback).

---

## UX / UI details

* Single window split:

  * Left: optional collapsible side panel for files, memories, settings.
  * Center: timeline (message bubbles stacked vertically — user on right, assistant on left).
  * Bottom: input area with icons: microphone (ASR), send, attach file, TTS toggle.
* Message bubble features:

  * Expand/collapse long messages
  * Attachments (download/open)
  * “Use as memory” button (flag message to be stored in Memori with metadata)
  * “Cite sources” toggle — when returning retriever results, show snippet + source file + confidence.
* Settings:

  * Providers: LLM/OpenRouter key, default model, TTS provider selection, ASR provider selection.
  * Vector DB choice & path.
  * Memory: enable/disable, memory persistence path, memory TTL / retention policy slider.
  * Privacy: toggle to upload logs (off by default).
* Developer Mode: show prompts sent to LLM, retriever items, embeddings preview, chain timings.

---

## Memory design (opinionated)

* Use **Memori** as canonical long-term memory and expose a LangChain memory adapter that:

  * On each assistant response: write summary/metadata of the exchange to Memori (configurable).
  * On input: fetch top-K memories (by relevance + recency + importance) and inject into prompt as a memory block.
  * Use separate namespaces for episodic (session-level), personal (long-term facts), and system (config).
  * Provide UI to review, edit, or delete memory entries.

Why Memori? It’s designed for structured memory and integrates well with LangChain patterns — makes it easier to keep memory semantics consistent across chains.

---

## Prompt & chain strategy

* **Prompt builder** composes:

  1. System prompt (editable, template support)
  2. Retrieved documents from Vector DB (formatted: title, snippet, source link)
  3. Retrieved memories (concise summary)
  4. Recent conversation window (N last messages)
  5. User message
  6. Assistant instructions (e.g., persona, verbosity)
* **Chain types**:

  * `RetrievalQAChain` (for direct QA)
  * `ConversationalChain` with memory + re-ranker
  * `MultimodalChain` for file-based inputs: parse file → create context → run QA
* **Safety & token management**:

  * Pre-run token estimator; if prompt > limit, truncate older context and/or retrieved docs with scoring.
  * Provide “short/concise/detailed” user toggles to change response length.

---

## File ingestion pipeline

* Accept file types: pdf, txt, docx, epub, csv, images (OCR optional).
* Steps:

  1. Receive file → store with hash
  2. Extract text (use existing libs: pdfminer, tika, python-docx, pytesseract for images)
  3. Chunking: sliding window + overlap (configurable chunk size)
  4. Embed chunks (via LangChain embedding wrapper)
  5. Upsert to vector store with metadata (filename, chunk_index, text_excerpt, source_url)
* UI: show ingestion status, number of chunks, option to delete file & its embeddings.

---

## TTS / ASR

* **Abstraction layer**: `AudioProvider` interface with implementations for each provider via LangChain.

  * Methods: `transcribe(audio) -> text`, `synthesize(text) -> audio_file/stream`, `list_voices()`
* Default provider: OpenAI TTS (per your note). Allow user to switch or add keys.
* Provide offline/local TTS option (if user has a local engine).
* Provide streaming playback in UI and option to save TTS output as MP3.

---

## Security & privacy

* Encrypt API keys at rest; require passphrase or OS keyring to unlock on start.
* Clear and explicit user consent for uploading files to third-party providers.
* Option: local-only mode where external providers disabled.
* Logs: store locally; telemetry off by default.
* Option to redact or obfuscate PII in memory and transcripts (configurable sensitivity filters).

---

## Developer & testing notes

* Unit tests for:

  * Prompt builder correctness
  * Memory adapter (memori integration) read/write
  * Vector store ingestion & retrieval (mock embeddings)
  * File parsing for each major file type
* Integration tests for:

  * End-to-end flow using mocked LangChain LLM connector (return deterministic completions)
* CI: run tests on push, linting (black/isort/flake8), packaging build.
* Provide sample `.env.example` with placeholders and clear README.

---

## Packaging & distribution

* Provide build scripts for PyInstaller (Windows exe), and a macOS dmg + Linux AppImage.
* Provide auto-update mechanism optional (signed releases).

---

## Logging, debugging, metrics

* Structured logs (JSON) with levels (ERROR, WARN, INFO, DEBUG).
* Option to export full conversation (for backup).
* Timing metrics: LLM latency, retrieval latency, embedding latency.

---

## Acceptance criteria (testable)

1. App launches on Windows & Linux with no errors.
2. User types a question → app builds prompt via LangChain → LLM returns response → message displayed and TTS plays if enabled.
3. Upload a PDF → file is parsed, chunked, embedded → retriever returns relevant chunk when asked a related question (validated by asserting returned metadata).
4. Voice input: record → ASR provider transcribes → transcription appears in input and is sent to chain.
5. Memory: mark a message as “save to memory” → memory adapter stores it → later question referencing that detail should retrieve it.
6. Provider switching: change LLM provider in settings → subsequent queries use new provider (LangChain connector utilized).
7. Secrets stored encrypted and unlocking requires passphrase (or OS keyring where available).
8. Prompt templating: user edits system prompt → assistant behavior changes accordingly (validated by test prompt).

---

## Suggested project structure (file/folder)

```
personal_assistant/
├─ src/
│  ├─ ui/
│  │  ├─ main_window.py
│  │  ├─ widgets/
│  ├─ core/
│  │  ├─ controller.py          # AI Controller
│  │  ├─ prompt_builder.py
│  │  ├─ memory_adapter.py
│  │  ├─ vector_manager.py
│  │  ├─ file_ingestor.py
│  │  ├─ audio_manager.py
│  │  ├─ settings.py
│  ├─ langchain_adapters/
│  │  ├─ llm_adapter.py
│  │  ├─ embed_adapter.py
│  │  ├─ tts_adapter.py
│  │  ├─ asr_adapter.py
│  ├─ storage/
│  │  ├─ sqlite_models.py
│  ├─ tests/
│  ├─ cli.py
├─ packaging/
├─ docker/ (optional)
├─ README.md
```

---

## Tasks for the AI coding agent (stepwise, granular)

1. **Scaffold project**: repo, venv, basic README, packaging config.
2. **UI skeleton**: PyQt6 window, timeline, input bar, file drop.
3. **LangChain plumbing**: simple LLM call via LangChain, show response in UI.
4. **Prompt builder**: implement system prompt + context window + user input.
5. **Vector ingestion**: implement file parser + chunker + embed + upsert to FAISS/Chroma.
6. **Retriever integration**: add retrieval logic into prompt builder and test retrieval QA.
7. **Memory (Memori) integration**: adapter to save & fetch memories; inject into prompts.
8. **Audio (ASR/TTS)**: implement provider interface; basic OpenAI path via LangChain for both.
9. **Settings & secrets**: encrypted store and provider switching UI.
10. **Polish UI**: message controls, playback, files pane, developer mode.
11. **Tests & CI**: add unit tests and CI configuration.
12. **Packaging**: create build script for at least Windows & Linux.


---

## Key constraints:
> * Use ** Context7 MCP ** to fetch recent actual docs for each lib u wil use! 
> * **All AI calls must go through LangChain.** Do not call OpenAI, Whisper, or other SDKs directly. Use LangChain connectors/adapters and an abstraction layer in `langchain_adapters/`.
> * Use a local vector store (FAISS or Chroma) by default, configurable in settings. Implement file ingestion (pdf, docx, txt) with chunking and embedding.
> * Integrate Memori for longterm/episodic memory via a `memory_adapter`. Provide UI to view and edit memories.
> * Implement TTS and ASR provider abstractions. Default TTS provider: OpenAI (via LangChain wrapper), allow provider switching in settings.
> * Single continuous conversation flow — no multiple chats. Keep windowed context and truncation logic.
> * Securely store provider API keys encrypted at rest.
>   Produce incremental commits: scaffold → UI skeleton → LangChain plumbing → retrieval → memory → audio → settings → tests → package. For each commit include concise unit tests and a README update describing features implemented.

## Example LangChain pattern (pseudo-code)

```python
# high-level pseudocode
from langchain import LLMChain, PromptTemplate, OpenAI, Embeddings, FAISS

# 1. Build prompt
system_prompt = load_system_prompt()
memories = memori_adapter.get_relevant(user_query, top_k=5)
docs = vector_retriever.get_relevant(user_query, top_k=3)
history = conversation.get_recent_messages(n=10)

prompt = PromptTemplate.from_template("""
{system_prompt}

Memories:
{memories}

Retrieved docs:
{docs}

Conversation history:
{history}

User:
{user_input}
""")

chain = LLMChain(llm=llm_adapter.get_llm(), prompt=prompt)
response = chain.run({
    "system_prompt": system_prompt,
    "memories": format(memories),
    "docs": format_docs(docs),
    "history": format_history(history),
    "user_input": user_input,
})
```
