import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        chat_id INTEGER PRIMARY KEY,
        state TEXT,
        name TEXT,
        phone TEXT,
        email TEXT,
        address TEXT,
        amount REAL,
        history TEXT
    )
    """)

    conn.commit()
    conn.close()


def get_user(chat_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    conn.close()
    return row


def upsert_user(chat_id, **kwargs):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO users (chat_id, state, name, phone, email, address, amount, history)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(chat_id) DO UPDATE SET
        state=excluded.state,
        name=excluded.name,
        phone=excluded.phone,
        email=excluded.email,
        address=excluded.address,
        amount=excluded.amount,
        history=excluded.history
    """, (
        chat_id,
        kwargs.get("state"),
        kwargs.get("name"),
        kwargs.get("phone"),
        kwargs.get("email"),
        kwargs.get("address"),
        kwargs.get("amount"),
        kwargs.get("history"),
    ))

    conn.commit()
    conn.close()
