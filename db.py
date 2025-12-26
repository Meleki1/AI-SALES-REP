import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat_memory.db"


def init_db():
    """Create conversations table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            chat_id INTEGER PRIMARY KEY,
            history TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_conversation(chat_id: int) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT history FROM conversations WHERE chat_id = ?",
        (chat_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return row[0] if row else ""


def save_conversation(chat_id: int, history: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO conversations (chat_id, history)
        VALUES (?, ?)
        ON CONFLICT(chat_id)
        DO UPDATE SET history = excluded.history
    """, (chat_id, history))

    conn.commit()
    conn.close()
