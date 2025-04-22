import sqlite3
import streamlit as st
import pandas as pd
import io
import json
from config import DB_NAME

def get_feedback_logs():
    conn = sqlite3.connect(DB_NAME)
    query = """
    SELECT id, query, answer, feedback_comment, user_feedback, timestamp
    FROM feedback_logs
    """
    feedback_df = pd.read_sql_query(query, conn)
    conn.close()
    feedback_df["User Feedback"] = feedback_df["user_feedback"].apply(lambda x: "👍 Yes" if x == 1 else "👎 No")
    feedback_df["Timestamp"] = pd.to_datetime(feedback_df["timestamp"])
    return feedback_df

def get_existing_metadata(dataset_type: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT json_data FROM metadata_files
        WHERE filename LIKE ? ORDER BY uploaded_at DESC LIMIT 1
    """, (f"%{dataset_type.upper()}%",))
    row = cursor.fetchone()
    conn.close()
    if row:
        json_data = row[0]
        df = pd.read_json(io.StringIO(json_data), orient="records")
        return df
    return None

def add_feedback_log(query, answer, category):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback_logs (query, answer, category)
        VALUES (?, ?, ?)
    """, (query, answer, category))

    # Get the ID of the row we just inserted
    log_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return log_id

def update_feedback_log(feedback, comment, log_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Update the feedback entry
    cursor.execute("""
        UPDATE feedback_logs
        SET user_feedback = ?, feedback_comment = ?
        WHERE id = ?
    """, (feedback, comment, log_id))

    conn.commit()
    conn.close()

def add_agent_detail(name, model, temperature, prompt):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO agent_detail (name, model, temperature, prompt)
        VALUES (?, ?, ?, ?)
    ''', (name, model, temperature, prompt))
    conn.commit()
    conn.close()
    st.success("✅ Settings Saved!!")

def load_feedback_data():
    """Loads existing feedback logs from the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback_logs")
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to list of dictionaries
    columns = ["ID", "Query", "Answer", "Category", "User Feedback", "Feedback Comment", "Timestamp"]
    feedback_logs = []
    for row in rows:
        feedback_logs.append(dict(zip(columns, row)))

    return feedback_logs

def load_data_for_dashboard():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT json_contents FROM graph_data WHERE file_name = 'latest_combined'")
    json_data = cursor.fetchone()[0]
    data = json.loads(json_data)
    conn.close()
    return data


