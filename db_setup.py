import sqlite3


conn = sqlite3.connect("chat_memory.db")


cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    chat_id INTEGER PRIMARY KEY,
    messages TEXT
)
""")


conn.commit()


conn.close()

print("✅ Database created successfully!")
