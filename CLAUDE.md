# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Memory-Bridge (记忆之桥) is a Streamlit app that creates a "digital echo" of a deceased loved one by analyzing uploaded WeChat chat logs. It uses DeepSeek's API for persona extraction and conversational generation, backed by local ChromaDB + BGE embeddings for RAG retrieval.

## Commands

```bash
# Activate virtual environment and run the app
source venv/bin/activate && streamlit run memory_bridge/app.py

# The app runs on http://localhost:8501 by default
```

There are no tests, linters, or build steps.

## Architecture

Modular structure under `memory_bridge/`:

```
memory_bridge/
├── app.py              # UI 层：main()、render_sidebar()、render_main()、render_chat_messages()、弹窗组件
├── config.py           # 全局常量与配置（路径、正则、模型参数），不 import 任何项目内模块
├── db.py               # SQLite 持久化层：get_db_connection()、init_db()、CRUD 函数
├── embedding.py        # ChromaDB + BGE：get_embedding_model()、chunk_text()、index_corpus()、retrieve_relevant()
├── llm.py              # DeepSeek API 接入：_build_deepseek_client()、extract_persona_from_txt()、generate_ai_response()
└── text_processing.py  # 文本工具：local_clean_chat_text()、extract_top_names()、format_wechat_time()
```

**Import 依赖链（无循环依赖）：**
```
config ← db, text_processing, embedding
config, db, embedding, text_processing ← llm
config, db, embedding, llm, text_processing ← app
```

**Data layer (SQLite)** — `db.py`，操作 `memory_bridge.db` 中的两个表：
- `chat_history(session_id, role, content, timestamp)` — per-session message log
- `system_memory(session_id, corpus)` — the extracted persona profile text, one row per session

All tables are keyed on `session_id` for multi-tenant isolation. WAL mode is enabled on every connection.

**RAG layer (ChromaDB + BGE)** — `embedding.py`，`chroma_db/` 目录：
- Chunks uploaded chat text (500-char windows, 50-char overlap, recursive splitting on paragraphs → newlines → Chinese punctuation → hard truncation)
- Embeds chunks with `BAAI/bge-base-zh-v1.5` via `sentence_transformers`, cached as a `@st.cache_resource` singleton
- Retrieves top-3 relevant chunks per user message, filtered by `session_id` metadata
- Each new upload clears old chunks for that session before re-indexing

**LLM layer (DeepSeek API)** — `llm.py`，两个独立调用：
1. **Persona extraction** (`extract_persona_from_txt`): Non-streaming call to `deepseek-v4-flash` with a detailed system prompt instructing psychological profiling (70% faithful / 30% softened). Output is ~300 chars of persona description stored in `system_memory`.
2. **Conversational generation** (`generate_ai_response`): Streaming call to `deepseek-v4-flash` with the persona + top-3 RAG chunks injected into the system prompt. A late-binding system override message is appended before the API call to resist jailbreak prompts.

**Session management** — `app.py` 的 `init_session_state()`，Session identity survives F5 refreshes via URL query params (`?sid=UUID`). On first visit, a random UUID is generated and written into `st.query_params`. All DB/RAG operations use this as the isolation key.

**Name detection** — `text_processing.py` 的 `extract_top_names()` parses WeChat log format to detect the top 2 speakers.

**Error handling** — `llm.py` 的 `_friendly_llm_error_message()` maps exceptions to user-facing Chinese messages without exposing internals.

## Key design decisions

- Modular split with strict dependency order (config is leaf, app is root); no circular imports
- Session affinity via URL params rather than cookies, avoiding cookie consent and browser compatibility issues
- Local embeddings (BGE) rather than API embeddings: avoids extra API costs and keeps all user chat data local
- The "ethical lock" system prompt structure is critical — the late-binding override message at the end of the message array is the main defense against role-change attacks

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
