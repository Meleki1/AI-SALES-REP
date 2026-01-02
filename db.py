import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat_memory.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            chat_id INTEGER PRIMARY KEY,
            history TEXT,
            payment_confirmed INTEGER DEFAULT 0
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
    

def set_payment_confirmed(chat_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE conversations SET payment_confirmed = 1 WHERE chat_id = ?",
        (chat_id,)
    )

    conn.commit()
    conn.close()


def is_payment_confirmed(chat_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT payment_confirmed FROM conversations WHERE chat_id = ?",
        (chat_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return bool(row and row[0] == 1)

