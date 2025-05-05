import sqlite3
import streamlit as st
import pandas as pd
import io
import json
from config import DB_NAME

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

def get__metadata_file_path(dataset_type: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s3_path, filename FROM metadata_files
        WHERE filename LIKE ? ORDER BY uploaded_at DESC LIMIT 1
    """, (f"%{dataset_type.upper()}%",))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row
    return None

def add_feedback_log(query, answer, agent, latency, category):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback_logs (query, answer, agent_id,latency, category)
        VALUES (?, ?, ?,?,?)
    """, (query, answer, agent,latency, category))

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

def load_feedback_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT
            log_id,
            query,
            answer,
            category,
            user_feedback,
            feedback_comment,
            latency,
            created_at
        FROM feedback_logs
        '''
        )
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to list of dictionaries
    columns = ["ID", "User Query", "Bot Answer", "Category", "User Feedback", "Feedback Comment", "Latency", "Query Timestamp"]
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

def load_business_context():
    """
    Load business context from the database and return a dictionary of key-value pairs (context_name -> description).
    """
    context = {}
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Business_Context';")
        table_exists = cursor.fetchone()

        if not table_exists:
            print("❌ Table 'Business_Context' does not exist.")
            return context

        # Fetch context_name and description
        cursor.execute("SELECT context_name, description FROM Business_Context")
        rows = cursor.fetchall()
        conn.close()

        # Process rows if they exist
        if rows:
            for context_name, description in rows:
                context[context_name] = description
        else:
            print("❌ No data found in the Business_Context table.")

    except Exception as e:
        print(f"❌ Error loading business context: {e}")

    return context

def fetch_latest_reports():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT report_date, json_data
        FROM daily_graph_data
        ORDER BY report_date DESC
        LIMIT 2
    """)
    rows = cursor.fetchall()
    conn.close()

    report_dicts = []
    for report_date, json_blob in rows:
        data = json.loads(json_blob)
        data["report_date"] = report_date
        report_dicts.append(data)

    # Sort chronologically (oldest to newest)
    return sorted(report_dicts, key=lambda x: x["report_date"])

def save_grouped_data_to_db(grouped_output, category):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    for report_date, data in grouped_output.items():
        json_blob = json.dumps(data)
        cursor.execute('''
            INSERT OR REPLACE INTO daily_graph_data (report_date, json_data, category)
            VALUES (?, ?,?)
        ''', (report_date, json_blob, category))

    conn.commit()
    conn.close()

def add_data_file_metadata(file_name, category):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO data_files_metadata (file_name, category)
        VALUES (?, ?)
    """, (file_name, category))

    conn.commit()
    conn.close()

    return "✅ File tracking added successfully!"

def get_all_agents():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            ra.agent_id,
            ra.name,
            ra.s3_folder,
            ra.model,
            ra.temperature,
            ra.prompt,
            ra.is_active
        FROM 
            rag_agents ra
    """)
    rows = cursor.fetchall()
    conn.close()
    columns = ["id", "name", "s3_folder", "model", "temperature", "prompt",'is_active']
    agents = []
    for row in rows:
        agents.append(dict(zip(columns, row)))
    return agents  