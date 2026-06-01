"""
SQLite 数据库持久化层
"""

import sqlite3
from contextlib import contextmanager

from config import DB_PATH


@contextmanager
def get_db_connection():
    """获取 SQLite 数据库连接的上下文管理器。"""
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """初始化 SQLite 数据库，创建 chat_history 和 system_memory 表。

    所有业务表均包含 session_id 字段用于多租户隔离。
    WAL 模式设置一次即持久化到数据库文件，后续连接自动生效。
    """
    with get_db_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                corpus TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_history_session
                ON chat_history(session_id, timestamp)
        """)
        conn.commit()


def load_messages_from_db(session_id: str, limit: int = 0) -> list[dict]:
    """按时间顺序读取指定 session 的历史对话。limit=0 表示读取全部。"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if limit > 0:
            cursor.execute(
                "SELECT role, content, timestamp FROM chat_history "
                "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cursor.fetchall()
            rows.reverse()
        else:
            cursor.execute(
                "SELECT role, content, timestamp FROM chat_history "
                "WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
    return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows]


def save_message_to_db(session_id: str, role: str, content: str) -> None:
    """将一条新消息插入 chat_history（绑定到当前 session）。"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def clear_chat_history_db(session_id: str) -> None:
    """仅清除当前 session 的历史对话，不波及他人数据。"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()


def clear_memory_db(session_id: str) -> None:
    """删除当前 session 的 system_memory 记录。"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM system_memory WHERE session_id = ?", (session_id,))
        conn.commit()


def save_memory_to_db(session_id: str, corpus: str) -> None:
    """将记忆语料持久化到当前 session 的 system_memory 行。"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO system_memory (session_id, corpus) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET corpus = excluded.corpus",
            (session_id, corpus),
        )
        conn.commit()


def load_memory_from_db(session_id: str) -> str | None:
    """从 system_memory 表读取当前 session 的记忆语料，无记录时返回 None。"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT corpus FROM system_memory WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
    return row[0] if row else None
