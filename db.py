import sqlite3

DB_NAME = "chat_memory.db"


def get_conversation(chat_id):
    """
    Load conversation history for a specific user.
    Returns a string (or empty string if new user).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT messages FROM conversations WHERE chat_id = ?",
        (chat_id,)
    )
    row = cursor.fetchone()

    conn.close()

    if row:
        return row[0]
    return ""


def save_conversation(chat_id, messages):
    """
    Save or update conversation history for a user.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO conversations (chat_id, messages)
        VALUES (?, ?)
        ON CONFLICT(chat_id)
        DO UPDATE SET messages = excluded.messages
        """,
        (chat_id, messages)
    )

    conn.commit()
    conn.close()
