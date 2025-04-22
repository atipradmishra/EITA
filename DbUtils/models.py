import sqlite3
from config import DB_NAME

def create_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_detail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model TEXT NOT NULL,
            temperature REAL DEFAULT 0.7,
            prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT,
            user_feedback BOOLEAN DEFAULT 1,
            feedback_comment TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_tracking (
            upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            date_processed TEXT,
            json_contents TEXT,
            is_processed INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            s3_path TEXT,
            json_data TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            json_contents TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            REPORT_DATE TEXT,
            SEGMENT TEXT,
            BOOK TEXT,
            BOOK_ATTR6 TEXT,
            BOOK_ATTR7 TEXT,
            BOOK_ATTR8 TEXT,
            USR_VAL4 TEXT,
            TGROUP1 TEXT,
            TGROUP2 TEXT,
            BUCKET TEXT,
            HORIZON TEXT,
            METHOD TEXT,
            VOLUME_BL REAL,
            VOLUME_PK REAL,
            VOLUME_OFPK REAL,
            MKT_VAL_BL REAL,
            MKT_VAL_PK REAL,
            MKT_VAL_OFPK REAL,
            TRD_VAL_BL REAL,
            TRD_VAL_PK REAL,
            TRD_VAL_OFPK REAL,
            source_file TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_file_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT UNIQUE,
            processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_processed INTEGER DEFAULT 1
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_ai_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client TEXT,
            summary TEXT NOT NULL,
            date TIMESTAMP 
        )
        """)
    conn.commit()
    conn.close()