# In src/visionx/database.py

import sqlite3
import datetime

DB_FILE = "visionx_log.db"

def init_db():
    """Initializes the SQLite database and creates the log table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            exit_direction TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_exit_event(person_name: str, entry_time: datetime, exit_time: datetime, direction: str):
    """Logs a person's exit event to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO session_logs (person_name, entry_time, exit_time, exit_direction) VALUES (?, ?, ?, ?)",
        (person_name, entry_time.isoformat(), exit_time.isoformat(), direction)
    )
    conn.commit()
    conn.close()
    print(f"DATABASE: Logged exit for {person_name}")