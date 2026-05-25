# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Memory-Bridge (记忆之桥) is a Streamlit app that creates a "digital echo" of a deceased loved one by analyzing uploaded WeChat chat logs. It uses DeepSeek's API for persona extraction and conversational generation, backed by local ChromaDB + BGE embeddings for RAG retrieval.

## Commands

```bash
# Activate virtual environment and run the app
source venv/bin/activate && streamlit run app.py

# The app runs on http://localhost:8501 by default
```

There are no tests, linters, or build steps — this is a single-file Streamlit MVP.

## Architecture

`app.py` (~1220 lines) is the entire application. It follows a Streamlit-native pattern with these layers:

**Data layer (SQLite)** — `memory_bridge.db` with three tables:
- `chat_history(session_id, role, content, timestamp)` — per-session message log
- `daily_usage(session_id, date_str, chat_count)` — anti-addiction daily cap (30 msgs)
- `system_memory(session_id, corpus)` — the extracted persona profile text, one row per session

All tables are keyed on `session_id` for multi-tenant isolation. WAL mode is enabled on every connection.

**RAG layer (ChromaDB + BGE)** — `chroma_db/` directory:
- Chunks uploaded chat text (500-char windows, 50-char overlap, recursive splitting on paragraphs → newlines → Chinese punctuation → hard truncation)
- Embeds chunks with `BAAI/bge-base-zh-v1.5` via `sentence_transformers`, cached as a `@st.cache_resource` singleton
- Retrieves top-3 relevant chunks per user message, filtered by `session_id` metadata
- Each new upload clears old chunks for that session before re-indexing

**LLM layer (DeepSeek API)** — two distinct calls:
1. **Persona extraction** (`extract_persona_from_txt`): Non-streaming call to `deepseek-v4-flash` with a detailed system prompt instructing psychological profiling (70% faithful / 30% softened). Output is ~300 chars of persona description stored in `system_memory`.
2. **Conversational generation** (`generate_ai_response`): Streaming call to `deepseek-v4-flash` with the persona + top-3 RAG chunks injected into the system prompt. A late-binding system override message is appended before the API call to resist jailbreak prompts.

**Session management** — Session identity survives F5 refreshes via URL query params (`?sid=UUID`). On first visit, a random UUID is generated and written into `st.query_params`. All DB/RAG operations use this as the isolation key.

**Name detection** — `extract_top_names()` parses WeChat log format (`YYYY-MM-DD HH:MM Name(wxid): content`) to detect the top 2 speakers. The sidebar radio lets the user choose who the AI should embody vs. who they are.

**Proxy support** — `_build_deepseek_client()` checks `HTTP_PROXY` env var and configures an `httpx.Client` proxy for the OpenAI SDK. This accommodates mainland China network environments accessing DeepSeek's API.

**Error handling** — `_friendly_llm_error_message()` maps exceptions (httpx, OpenAI SDK) to user-facing Chinese messages without exposing internals. This is used consistently in both persona extraction and chat generation.

## Key design decisions

- Single-file architecture: trading modularity for deployability (just `streamlit run app.py` on a Windows/WSL machine with a GPU)
- Session affinity via URL params rather than cookies, avoiding cookie consent and browser compatibility issues
- Local embeddings (BGE) rather than API embeddings: avoids extra API costs and keeps all user chat data local
- The "ethical lock" system prompt structure is critical — the late-binding override message at the end of the message array is the main defense against role-change attacks
