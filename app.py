import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import boto3
import numpy as np
import time
import json
from datetime import datetime
from io import BytesIO
import io
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from openai import OpenAI
import os

# --- Streamlit page configuration ---
st.set_page_config(page_title="EITA", layout="wide")

EMBEDDING_MODEL = "text-embedding-ada-002"
DB_NAME = "vector_chunks.db"
COMPUTED_DB = "data.db"
VALID_BUCKET = "etrm-etai-poc-chub"
REJECTED_BUCKET = "etai-rejected-files"

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")

client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])


# --- Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "main_section" not in st.session_state:
    st.session_state.main_section = "Data Management AI Agent"
if "sub_section" not in st.session_state:
    st.session_state.sub_section = "Pipeline Dashboard"
if "feedback_logs" not in st.session_state:
    try:
        with open("energy_trading_feedback.json", "r") as f:
            st.session_state.feedback_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.feedback_logs = []
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = 0.7
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {
            "role": "system",
            "content": "You are a strict assistant. If feedback directly answers the question, always prioritize it over the data."
        }
    ]


# --- Database Functions ---
def init_sqlite_db():
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
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text TEXT,
            embedding BLOB,
            source_file TEXT
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
            user_feedback TEXT,
            feedback_comment TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    # Create training data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            incorrect_answer TEXT NOT NULL,
            correction TEXT NOT NULL,
            timestamp TEXT NOT NULL
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
    conn.commit()
    conn.close()

def init_computed_db():
    # Placeholder for additional database initialization if needed
    conn = sqlite3.connect(COMPUTED_DB)
    conn.close()

init_sqlite_db()

def register_user(username, password):
    """Registers a new user with hashed password."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticates user with hashed password verification."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_password = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False


def process_files_from_s3_folder(bucket_name, folder_prefix):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix).get("Contents", [])
        print(f"🔍 Found {len(objects)} files in folder: {folder_prefix}")
        for obj in objects:
            file_key = obj["Key"]
            if not file_key.lower().endswith(".csv"):
                continue

            file_name = file_key.split("/")[-1]
            cursor.execute("SELECT 1 FROM file_tracking WHERE file_name = ? AND is_processed = 1", (file_name,))
            if cursor.fetchone():
                print(f"⏭️ Skipping already processed file: {file_name}")
                continue

            try:
                obj_data = s3.get_object(Bucket=bucket_name, Key=file_key)
                file_stream = io.BytesIO(obj_data["Body"].read())
                df = pd.read_csv(file_stream)
                df_processed = process_and_store_file(file_name, df, cursor,folder_prefix)
                conn.commit()
                print(f"✅ Processed and saved: {file_name}")
            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")

    except Exception as e:
        print(f"❌ Error accessing S3: {e}")
    finally:
        conn.close()


# --- Sidebar and User Menu ---
def sidebar_menu():
    with st.sidebar:
        st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95")
        st.markdown("<h2 style='text-align: center;'>ETAI Energy Trading AI Platform</h2>", unsafe_allow_html=True)
        st.divider()
        main_section_options = ["Data Management AI Agent", "RAG AI Agent", "Application AI Agent"]
        main_section = st.radio("Select AI Agent", main_section_options, index=main_section_options.index(st.session_state.main_section))
        st.session_state.main_section = main_section

        sub_sections = {
            "Data Management AI Agent": ["Pipeline Dashboard", "Data Pipeline", "Processed Data"],
            "RAG AI Agent": ["Dashboard", "Configure & Upload", "Fine Tuning", "Settings"],
            "Application AI Agent": ["Energy Tradeing Analysis", "Graph Query", "Deviation Analysis", "Root Cause Analysis", "Analysis History", "User Feedback"]
        }
        if st.session_state.sub_section not in sub_sections[main_section]:
            st.session_state.sub_section = sub_sections[main_section][0]
        sub_section = st.radio(f"Select {main_section} Section", sub_sections[main_section], index=sub_sections[main_section].index(st.session_state.sub_section))
        st.session_state.sub_section = sub_section
        st.divider()

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()


def top_right_menu():
    username = st.session_state.get('username', 'Guest')
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"👤 **{username}**", unsafe_allow_html=True)
    with col2:
        if st.button("🔴 Logout", key="logout_btn", help="Logout from the platform"):
            logout()

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { box-shadow: 2px 0 10px rgba(0,0,0,0.1); }
        button { border-radius: 10px !important; }
        button:hover { background: #007BFF !important; color: white !important; border: none !important; }
        h1, h2, h3 { color: #007BFF; }
        hr { border: 1px solid #ccc; }
        .login-btn { width: 100%; background-color: #4CAF50; color: white; padding: 12px 20px; margin: 10px 0; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: 0.3s; }
        .login-btn:hover { background-color: #45a049; }
        .login-input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
    </style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
def process_metadata_alias(metadata_df: pd.DataFrame):
    alias_mapping = {}
    for _, row in metadata_df.iterrows():
        col_name = str(row.get("Column", "")).strip()
        description = str(row.get("Description", "")).strip().lower()
        if col_name and description:
            alias_mapping[description] = col_name
    return alias_mapping

def upload_to_s3(file, filename, bucket):
    try:
        folder = "misc/"
        if "NOP_CO2" in filename.upper():
            folder = "CO2/"
        elif "NOP_NG" in filename.upper():
            folder = "NG/"
        elif "NOP_PW" in filename.upper():
            folder = "PW/"
        s3_key = folder + filename
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        s3.upload_fileobj(file, bucket, s3_key)
        return True, f"✅ Uploaded {filename} to S3 bucket '{bucket}' in folder '{folder}'"
    except Exception as e:
        return False, f"❌ Upload failed: {e}"

def validate_against_metadata(file_df: pd.DataFrame, metadata_df: pd.DataFrame, file_id: str) -> bool:
    file_cols = set(file_df.columns.str.strip().str.upper())
    metadata_cols = set(metadata_df["Field_Name"].str.strip().str.upper())
    st.subheader(f"🔍 Column Comparison for `{file_id}`")
    st.write("🗂 **Data File Columns**:")
    st.code("\n".join(sorted(file_cols)))
    st.write("📘 **Metadata Columns**:")
    st.code("\n".join(sorted(metadata_cols)))
    missing_in_file = metadata_cols - file_cols
    extra_in_file = file_cols - metadata_cols
    if missing_in_file:
        st.error(f"❌ Columns missing in data file: {', '.join(missing_in_file)}")
    if extra_in_file:
        st.warning(f"⚠️ Extra columns in data file: {', '.join(extra_in_file)}")
    if not missing_in_file:
        st.success("✅ Validation Passed.")
        return True
    else:
        st.error("❌ Validation Failed.")
        return False

def store_processed_data_in_sqlite(file_name, report_date, processed_data):
    conn = sqlite3.connect("data_store.db")
    cursor = conn.cursor()
    processed_json = json.dumps(processed_data)
    cursor.execute('''
        INSERT INTO processed_data (file_name, report_date, processed_data, processed_flag)
        VALUES (?, ?, ?, ?)
    ''', (file_name, report_date, processed_json, 1))
    conn.commit()
    conn.close()
    print(f"✅ Processed data for {file_name} stored in SQLite")

def sanitize_for_json(obj, decimal_places=2):
    if isinstance(obj, list):
        return [sanitize_for_json(i, decimal_places) for i in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, float):
        return str(round(obj, decimal_places))
    elif isinstance(obj, (int, np.int64, np.int32)):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def process_and_store_file(file_name: str, df: pd.DataFrame, cursor, folder_prefix):
    df.columns = df.columns.str.strip().str.upper()
    def parse_dates(report_date):
        if isinstance(report_date, str):
            report_date = report_date.replace('–', '-')
        for fmt in ['%d%b%y', '%d-%m-%Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(report_date, format=fmt)
            except (ValueError, TypeError):
                continue
        print(f"Warning: Could not parse date: {report_date}")
        return None
    df['REPORT_DATE'] = df['REPORT_DATE'].astype(str).str.strip()
    df['REPORT_DATE'] = df['REPORT_DATE'].apply(parse_dates)
    df['REPORT_DATE'] = df['REPORT_DATE'].apply(lambda dt: dt.strftime('%Y-%m-%d') if pd.notnull(dt) else None)

    if folder_prefix == 'CO2':
        numeric_cols = ['VOLUME', 'MKTVAL', 'TRDVAL', 'TRDPRC']
    elif folder_prefix == 'NG':
        numeric_cols = ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL']
    elif folder_prefix == 'PW':
        numeric_cols = ['VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK', 'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
    else:
        raise ValueError(f"❌ Unknown folder prefix: {folder_prefix}")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=True), errors='coerce').fillna(0)

    if folder_prefix == 'CO2':
        df_daily_stats = df.groupby('REPORT_DATE').agg(
            TOTAL_VOLUME=('VOLUME', 'sum'), AVG_VOLUME=('VOLUME', 'mean'),
            MIN_VOLUME=('VOLUME', 'min'), MAX_VOLUME=('VOLUME', 'max'), STD_VOLUME=('VOLUME', 'std'),
            TOTAL_TRDVAL=('TRDVAL', 'sum'), AVG_TRDVAL=('TRDVAL', 'mean'),
            MIN_TRDVAL=('TRDVAL', 'min'), MAX_TRDVAL=('TRDVAL', 'max'), STD_TRDVAL=('TRDVAL', 'std'),
            TOTAL_MKTVAL=('MKTVAL', 'sum'), AVG_MKTVAL=('MKTVAL', 'mean'),
            MIN_MKTVAL=('MKTVAL', 'min'), MAX_MKTVAL=('MKTVAL', 'max'), STD_MKTVAL=('MKTVAL', 'std'),
            TOTAL_TRDPRC=('TRDPRC', 'sum'), AVG_TRDPRC=('TRDPRC', 'mean'),
            MIN_TRDPRC=('TRDPRC', 'min'), MAX_TRDPRC=('TRDPRC', 'max'), STD_TRDPRC=('TRDPRC', 'std')
        ).reset_index()
    elif folder_prefix == 'NG':
        df_daily_stats = df.groupby('REPORT_DATE').agg(
            TOTAL_VOLUME=('VOLUME', 'sum'), AVG_VOLUME=('VOLUME', 'mean'),
            MIN_VOLUME=('VOLUME', 'min'), MAX_VOLUME=('VOLUME', 'max'), STD_VOLUME=('VOLUME', 'std'),
            TOTAL_VOLUME_TOTAL=('VOLUME_TOTAL', 'sum'), AVG_VOLUME_TOTAL=('VOLUME_TOTAL', 'mean'),
            MIN_VOLUME_TOTAL=('VOLUME_TOTAL', 'min'), MAX_VOLUME_TOTAL=('VOLUME_TOTAL', 'max'), STD_VOLUME_TOTAL=('VOLUME_TOTAL', 'std'),
            TOTAL_QTY_PHY=('QTY_PHY', 'sum'), AVG_QTY_PHY=('QTY_PHY', 'mean'),
            MIN_QTY_PHY=('QTY_PHY', 'min'), MAX_QTY_PHY=('QTY_PHY', 'max'), STD_QTY_PHY=('QTY_PHY', 'std'),
            TOTAL_MKT_VAL=('MKT_VAL', 'sum'), AVG_MKT_VAL=('MKT_VAL', 'mean'),
            MIN_MKT_VAL=('MKT_VAL', 'min'), MAX_MKT_VAL=('MKT_VAL', 'max'), STD_MKT_VAL=('MKT_VAL', 'std'),
            TOTAL_QTY_FIN=('QTY_FIN', 'sum'), AVG_QTY_FIN=('QTY_FIN', 'mean'),
            MIN_QTY_FIN=('QTY_FIN', 'min'), MAX_QTY_FIN=('QTY_FIN', 'max'), STD_QTY_FIN=('QTY_FIN', 'std'),
            TOTAL_TRD_VAL=('TRD_VAL', 'sum'), AVG_TRD_VAL=('TRD_VAL', 'mean'),
            MIN_TRD_VAL=('TRD_VAL', 'min'), MAX_TRD_VAL=('TRD_VAL', 'max'), STD_TRD_VAL=('TRD_VAL', 'std')
        ).reset_index()
    elif folder_prefix == 'PW':
        grouping_columns = ['REPORT_DATE']
        additional_groupings = ['BOOK', 'SEGMENT', 'HORIZON', 'TGROUP1']
        for col in additional_groupings:
            if col in df.columns:
                grouping_columns.append(col)
        df_daily_stats = df.groupby(grouping_columns).agg(
            TOTAL_VOLUME_BL=('VOLUME_BL', 'sum'), AVG_VOLUME_BL=('VOLUME_BL', 'mean'),
            MIN_VOLUME_BL=('VOLUME_BL', 'min'), MAX_VOLUME_BL=('VOLUME_BL', 'max'),
            TOTAL_VOLUME_PK=('VOLUME_PK', 'sum'), AVG_VOLUME_PK=('VOLUME_PK', 'mean'),
            MIN_VOLUME_PK=('VOLUME_PK', 'min'), MAX_VOLUME_PK=('VOLUME_PK', 'max'),
            TOTAL_VOLUME_OFPK=('VOLUME_OFPK', 'sum'), AVG_VOLUME_OFPK=('VOLUME_OFPK', 'mean'),
            MIN_VOLUME_OFPK=('VOLUME_OFPK', 'min'), MAX_VOLUME_OFPK=('VOLUME_OFPK', 'max'),
            TOTAL_MKT_VAL_BL=('MKT_VAL_BL', 'sum'), AVG_MKT_VAL_BL=('MKT_VAL_BL', 'mean'),
            MIN_MKT_VAL_BL=('MKT_VAL_BL', 'min'), MAX_MKT_VAL_BL=('MKT_VAL_BL', 'max'),
            TOTAL_MKT_VAL_PK=('MKT_VAL_PK', 'sum'), AVG_MKT_VAL_PK=('MKT_VAL_PK', 'mean'),
            MIN_MKT_VAL_PK=('MKT_VAL_PK', 'min'), MAX_MKT_VAL_PK=('MKT_VAL_PK', 'max'),
            TOTAL_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'sum'), AVG_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'mean'),
            MIN_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'min'), MAX_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'max'),
            TOTAL_TRD_VAL_BL=('TRD_VAL_BL', 'sum'), AVG_TRD_VAL_BL=('TRD_VAL_BL', 'mean'),
            MIN_TRD_VAL_BL=('TRD_VAL_BL', 'min'), MAX_TRD_VAL_BL=('TRD_VAL_BL', 'max'),
            TOTAL_TRD_VAL_PK=('TRD_VAL_PK', 'sum'), AVG_TRD_VAL_PK=('TRD_VAL_PK', 'mean'),
            MIN_TRD_VAL_PK=('TRD_VAL_PK', 'min'), MAX_TRD_VAL_PK=('TRD_VAL_PK', 'max'),
            TOTAL_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'sum'), AVG_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'mean'),
            MIN_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'min'), MAX_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'max')
        ).reset_index()

    combined_json = {
        "daily_totals": df_daily_stats.to_dict(orient='records')
    }
    sanitized_json = sanitize_for_json(combined_json)
    cursor.execute(
        "INSERT INTO file_tracking (file_name, date_processed, json_contents, is_processed) VALUES (?, datetime('now'), ?, 1)",
        (file_name, json.dumps(sanitized_json))
    )
    return df

def query_sqlite_json_with_openai(user_question, category=None):
    """
    Query JSON data from SQLite and get an answer using OpenAI.
    Incorporates feedback from similar past queries and maintains an in-memory chat.
    """

    # Load feedback data and create FAISS index
    feedback_data = load_feedback_data()
    faiss_index, feedback_data_indexed = create_faiss_index(feedback_data)
    feedback_insights = retrieve_feedback_insights(user_question, faiss_index, feedback_data_indexed)

    # Query SQLite based on the provided category
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if category:
        if category == "CO2":
            cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_CO2%'")
        elif category == "Natural Gas":
            cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND (file_name LIKE 'NOP_GAS%' OR file_name LIKE 'NOP_NG%')")
        elif category == "Power":
            cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND (file_name LIKE 'NOP_POWER%' OR file_name LIKE 'NOP_PW%')")
    else:
        cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "⚠️ No processed JSON data found for the selected category."

    # Build the context from all retrieved JSON files
    all_context = ""
    file_names = []
    for file_name, json_text in rows:
        file_names.append(file_name)
        try:
            json_data = json.loads(json_text)
            summary = json.dumps(json_data, separators=(',', ':'))[:6000]
            all_context += f"\n---\n📄 File: {file_name}\n{summary}"
        except Exception as e:
            all_context += f"\n---\n📄 File: {file_name}\n⚠️ Error reading JSON: {e}"

    # Build the feedback text if there are any insights
    feedback_text = ""
    if feedback_insights:
        feedback_text = "\n\nBased on feedback from similar queries, please be aware of these issues:\n"
        feedback_text += "\n".join(f"- {insight}" for insight in feedback_insights)

    category_context = f"for the {category} category" if category and category != "All" else ""

    # Prepare a context message that provides background information.
    context_message = f"""
📌 VERIFIED FEEDBACK (Authoritative Corrections):

Use this section as the most accurate reference if it directly answers the question.
{feedback_text}

🗂️ PREPROCESSED DATA (Official Trading Statistics):
This section contains structured trading data retrieved from SQLite for multiple files {category_context}.
{all_context}

⚠️ INSTRUCTION:
If the verified feedback directly answers the user question, **use that answer** — do not override it with the data below.
If feedback is not relevant or does not address the current question, refer to the preprocessed data to generate your answer.

🔄 SUGGESTED FOLLOW-UP QUESTIONS:
Based on the current question and available data, suggest 2–3 natural follow-up questions.
"""

    # Instead of combining everything in one prompt, we separate the context from the user question.
    # First, add the context as a system-level message.
    st.session_state.conversation.append({"role": "system", "content": context_message})
    # Then, add the raw user question as a user message.
    st.session_state.conversation.append({"role": "user", "content": user_question})

    # OPTIONAL: Print out the full conversation for debugging purposes.
    for idx, message in enumerate(st.session_state.conversation):
        role = message["role"]
        content = message["content"]
        print(f"\n🔹 Message {idx+1} ({role}):\n{content}\n{'-'*60}")

    # Call OpenAI with the full conversation history
    response = client.chat.completions.create(
        model="gpt-4",
        messages=st.session_state.conversation
    )
    gpt_answer = response.choices[0].message.content.strip()

    # Append GPT's response to the conversation memory.
    st.session_state.conversation.append({"role": "assistant", "content": gpt_answer})

    # Optional: Truncate conversation history to avoid hitting token limits (keeping system message + last 18 messages).
    if len(st.session_state.conversation) > 20:
        st.session_state.conversation = [st.session_state.conversation[0]] + st.session_state.conversation[-18:]

    # Calculate a confidence score using your custom function
    confidence_score = calculate_confidence_score(0.7, feedback_insights, file_names)
    st.session_state.confidence_score = confidence_score

    # Save output to a file for logging purposes
    with open("query_output.txt", "w", encoding="utf-8") as f:
        f.write("🔹 Extracted Data:\n" + all_context)
        f.write("\n\n🔹 GPT Answer:\n" + gpt_answer)

    return gpt_answer


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

def create_faiss_index(feedback_data):
    if not feedback_data:
        return None, []
    texts = [item.get("Query", "") for item in feedback_data if item.get("Query", "")]
    if not texts:
        return None, []
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform(texts).astype(np.float32).toarray()
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        return index, feedback_data
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None, []

def retrieve_feedback_insights(query, faiss_index, feedback_data_indexed, top_k=3):
    if not faiss_index or not feedback_data_indexed:
        return []
    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit([item.get("Query", "") for item in feedback_data_indexed if item.get("Query", "")])
        query_vector = vectorizer.transform([query]).astype(np.float32).toarray()
        distances, indices = faiss_index.search(query_vector, top_k)
        insights = []
        for idx in indices[0]:
            if idx < len(feedback_data_indexed):
                feedback_item = feedback_data_indexed[idx]
                if feedback_item.get("User Feedback") == "👎 No" and feedback_item.get("Feedback Comment", "").strip():
                    insights.append(f"Previous issue: {feedback_item.get('Query')} - {feedback_item.get('Feedback Comment')}")
        return insights
    except Exception as e:
        print(f"Error retrieving feedback insights: {e}")
        return []

def calculate_confidence_score(base_score, feedback_insights, relevant_documents):
    confidence = base_score
    if feedback_insights:
        confidence -= 0.05 * len(feedback_insights)
    if relevant_documents:
        confidence += 0.1 * min(len(relevant_documents), 3)
    return max(0.1, min(0.99, confidence))

def prepare_training_data():
    """
    Retrieves all training data pairs from the database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT query, incorrect_answer, correction FROM training_data")
    rows = cursor.fetchall()

    conn.close()

    # Convert to list of dictionaries
    training_pairs = []
    for row in rows:
        query, incorrect_answer, correction = row
        training_pairs.append({
            "query": query,
            "incorrect_answer": incorrect_answer,
            "correction": correction
        })

    return training_pairs

def save_training_data():
    """
    Saves the training data to a file for future fine-tuning.
    Returns the number of training pairs saved.
    """
    training_pairs = prepare_training_data()

    if not training_pairs:
        return 0

    # For file compatibility, you might still want to export to JSON
    # Or you could implement a different export method
    try:
        import json
        with open("energy_trading_training_data.json", "w") as f:
            json.dump(training_pairs, f, indent=4)
        return len(training_pairs)
    except Exception as e:
        print(f"Error saving training data: {e}")
        return 0


def add_feedback_log(query, answer, category=None):
    """
    Adds a new feedback entry for the given AI response to SQLite.

    Returns a unique log_id for the entry.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback_logs (query, answer, category, user_feedback, feedback_comment, timestamp)
        VALUES (?, ?, ?, NULL, "", ?)
    """, (query, answer, category, timestamp))

    # Get the ID of the row we just inserted
    log_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return log_id

def update_feedback_log(feedback, comment, log_id):
    """
    Updates a feedback log entry (identified by log_id) with the user's feedback and comment.
    If negative feedback is provided, training data is prepared and saved.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Update the feedback entry
    cursor.execute("""
        UPDATE feedback_logs
        SET user_feedback = ?, feedback_comment = ?
        WHERE id = ?
    """, (feedback, comment, log_id))

    conn.commit()

    # If the feedback is negative and includes a comment, add to training data
    if feedback == "👎 No" and comment.strip():
        # Get the original query and answer
        cursor.execute("SELECT query, answer FROM feedback_logs WHERE id = ?", (log_id,))
        result = cursor.fetchone()
        if result:
            query, incorrect_answer = result

            # Add to training data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO training_data (query, incorrect_answer, correction, timestamp)
                VALUES (?, ?, ?, ?)
            """, (query, incorrect_answer, comment, timestamp))

            conn.commit()

    conn.close()

def plot_graph_based_on_prompt_all(prompt, category_key):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        st.warning("⚠️ No processed JSON data found in the database.")
        return

    categories = {
        "CO2": {
            "prefix": "CO2",
            "fields": {
                'VOLUME': 'TOTAL_VOLUME',
                'AVG_VOLUME': 'AVG_VOLUME',
                'MIN_VOLUME': 'MIN_VOLUME',
                'MAX_VOLUME': 'MAX_VOLUME',
                'STD_VOLUME': 'STD_VOLUME',
                'TRDVAL': 'TOTAL_TRDVAL',
                'AVG_TRDVAL': 'AVG_TRDVAL',
                'MIN_TRDVAL': 'MIN_TRDVAL',
                'MAX_TRDVAL': 'MAX_TRDVAL',
                'STD_TRDVAL': 'STD_TRDVAL',
                'MKTVAL': 'TOTAL_MKTVAL',
                'AVG_MKTVAL': 'AVG_MKTVAL',
                'MIN_MKTVAL': 'MIN_MKTVAL',
                'MAX_MKTVAL': 'MAX_MKTVAL',
                'STD_MKTVAL': 'STD_MKTVAL',
                'TRDPRC': 'TOTAL_TRDPRC',
                'AVG_TRDPRC': 'AVG_TRDPRC',
                'MIN_TRDPRC': 'MIN_TRDPRC',
                'MAX_TRDPRC': 'MAX_TRDPRC',
                'STD_TRDPRC': 'STD_TRDPRC'
            }
        },
        "Natural Gas": {
            "prefix": "NG",
            "fields": {
                'VOLUME': 'TOTAL_VOLUME',
                'VOLUME_TOTAL': 'TOTAL_VOLUME_TOTAL',
                'QTY_PHY': 'TOTAL_QTY_PHY',
                'MKT_VAL': 'TOTAL_MKT_VAL',
                'QTY_FIN': 'TOTAL_QTY_FIN',
                'TRD_VAL': 'TOTAL_TRD_VAL'
            }
        },
        "Power": {
            "prefix": "PW",
            "fields": {
                'VOLUME_BL': 'TOTAL_VOLUME_BL',
                'VOLUME_PK': 'TOTAL_VOLUME_PK',
                'VOLUME_OFPK': 'TOTAL_VOLUME_OFPK',
                'MKT_VAL_BL': 'TOTAL_MKT_VAL_BL',
                'MKT_VAL_PK': 'TOTAL_MKT_VAL_PK',
                'MKT_VAL_OFPK': 'TOTAL_MKT_VAL_OFPK',
                'TRD_VAL_BL': 'TOTAL_TRD_VAL_BL',
                'TRD_VAL_PK': 'TOTAL_TRD_VAL_PK',
                'TRD_VAL_OFPK': 'TOTAL_TRD_VAL_OFPK'
            }
        }
    }
    if category_key not in categories:
        st.error("❌ Invalid category selected.")
        return
    prefix = categories[category_key]["prefix"]
    fields_to_plot = categories[category_key]["fields"]
    expected_prefix = f"NOP_{prefix}"
    st.write(f"Fetching data from files starting with: **{expected_prefix}**")
    date_patterns = [
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})',
        r'(\d{1,2})([A-Za-z]{3})(\d{2,4})'
    ]
    filter_conditions = {}
    filter_patterns = [
        r'where\s+(\w+)\s+is\s+[\'"]([^\'"]+)[\'"]',
        r'where\s+(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]'
    ]
    for pattern in filter_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                field, value = match
                filter_conditions[field.upper()] = value
    st.write(f"Detected filter conditions: {filter_conditions}")
    target_date = None
    for pattern in date_patterns:
        matches = re.findall(pattern, prompt)
        if matches:
            try:
                if len(matches[0]) == 3:
                    if matches[0][1].isalpha():
                        day, month_str, year = matches[0]
                        if len(year) == 2:
                            year = '20' + year if int(year) < 50 else '19' + year
                        date_str = f"{day}{month_str}{year}"
                        target_date = pd.to_datetime(date_str, format='%d%b%Y')
                    else:
                        day, month, year = matches[0]
                        if len(year) == 2:
                            year = '20' + year if int(year) < 50 else '19' + year
                        date_str = f"{day}-{month}-{year}"
                        target_date = pd.to_datetime(date_str)
                break
            except ValueError:
                continue

    has_dimension_filter = any(dim in filter_conditions for dim in ['BOOK', 'TGROUP1', 'SEGMENT', 'BUCKET'])
    all_data = []
    dimension_data = []
    for file_name, json_text in rows:
        if not file_name.startswith(expected_prefix):
            continue
        try:
            json_data = json.loads(json_text)
            st.write(f"Processing file: **{file_name}**")
            if "daily_totals" in json_data:
                all_data.extend(json_data["daily_totals"])
            if has_dimension_filter:
                if 'BOOK' in filter_conditions and "book_stats" in json_data:
                    dimension_data.extend(json_data["book_stats"])
                if 'TGROUP1' in filter_conditions and "tgroup1_stats" in json_data:
                    dimension_data.extend(json_data["tgroup1_stats"])
                if 'SEGMENT' in filter_conditions and "segment_stats" in json_data:
                    dimension_data.extend(json_data["segment_stats"])
                if 'BUCKET' in filter_conditions and "bucket_stats" in json_data:
                    dimension_data.extend(json_data["bucket_stats"])
                if "row_level_data" in json_data and not dimension_data:
                    dimension_data.extend(json_data["row_level_data"])
        except Exception as e:
            st.error(f"⚠️ Error reading {file_name}: {e}")
    if not all_data and not dimension_data:
        st.error(f"❌ No valid data found for {category_key}.")
        return
    if has_dimension_filter and dimension_data:
        st.write(f"Applying dimension filters to {len(dimension_data)} records...")
        df_dimensions = pd.DataFrame(dimension_data)
        filtered_dimensions = df_dimensions.copy()
        for field, value in filter_conditions.items():
            if field in filtered_dimensions.columns:
                filtered_dimensions = filtered_dimensions[filtered_dimensions[field] == value]
                st.write(f"Filtered dimension where {field} = '{value}' - {len(filtered_dimensions)} records")
            else:
                st.warning(f"⚠️ Filter field '{field}' not found in dimension data")
        if not filtered_dimensions.empty:
            st.write("Using dimension data for plotting...")
            return plot_dimension_data(filtered_dimensions, prompt, category_key, fields_to_plot)
    if all_data:
        df_graph = pd.DataFrame(all_data)
        if 'REPORT_DATE' not in df_graph.columns:
            st.error("❌ REPORT_DATE column missing in JSON data.")
            return
        df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])
        if target_date is not None:
            df_graph = df_graph[df_graph['REPORT_DATE'].dt.date == target_date.date()]
            if df_graph.empty:
                st.warning(f"⚠️ No data found for date: {target_date.strftime('%Y-%m-%d')}")
                return
        normalized_prompt = prompt.lower().replace(" ", "").replace("_", "")
        metrics_to_plot = []
        for key, col_name in fields_to_plot.items():
            normalized_key = key.lower().replace("_", "")
            if col_name in df_graph.columns and (normalized_key in normalized_prompt or not metrics_to_plot):
                metrics_to_plot.append((key, col_name))
        if target_date is not None and df_graph['REPORT_DATE'].nunique() == 1:
            plot_single_date_metrics(df_graph, metrics_to_plot, category_key, target_date)
        else:
            plot_time_series(df_graph, metrics_to_plot, category_key, target_date)
    else:
        st.warning("⚠️ No time series data available for plotting.")

def plot_dimension_data(df, prompt, category_key, fields_to_plot):
    if df.empty:
        st.error("❌ No dimension data to plot.")
        return
    dimension_col = next((col for col in ['BOOK', 'TGROUP1', 'SEGMENT', 'BUCKET'] if col in df.columns), None)
    if not dimension_col:
        st.error("❌ No dimension column found in filtered data.")
        return
    normalized_prompt = prompt.lower().replace(" ", "").replace("_", "")
    metrics_to_plot = []
    for key, col_name in fields_to_plot.items():
        normalized_key = key.lower().replace("_", "")
        if col_name in df.columns and (normalized_key in normalized_prompt or not metrics_to_plot):
            metrics_to_plot.append((key, col_name))
    if not metrics_to_plot:
        st.error(f"❌ No valid metrics found in dimension data for {category_key}.")
        return
    for _, metric_col in metrics_to_plot:
        if metric_col in df.columns:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(f'{dimension_col}:N', sort='-y'),
                y=alt.Y(f'{metric_col}:Q'),
                tooltip=[dimension_col, metric_col]
            ).properties(
                title=f"{category_key} - {metric_col} by {dimension_col}",
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning(f"⚠️ Metric {metric_col} not found in dimension data.")


def plot_single_date_metrics(df, metrics_to_plot, category_key, target_date):
    if not metrics_to_plot:
        st.error(f"❌ No valid metrics found to plot for {category_key}.")
        return
    data = []
    for key, col_name in metrics_to_plot:
        if col_name in df.columns:
            value = df[col_name].iloc[0] if not pd.isna(df[col_name].iloc[0]) else 0
            data.append({"Metric": key, "Value": value})
    if not data:
        st.error(f"❌ No valid data found for the selected metrics on {target_date.strftime('%Y-%m-%d')}.")
        return
    df_chart = pd.DataFrame(data)
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Value:Q', title="Value"),
        y=alt.Y('Metric:N', sort='-x', title="Metric"),
        color=alt.condition(
            alt.datum.Value > 0,
            alt.value("steelblue"),
            alt.value("red")
        ),
        tooltip=['Metric', 'Value']
    ).properties(
        title=f"{category_key} Metrics for {target_date.strftime('%d-%b-%Y')}",
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def plot_time_series(df, metrics_to_plot, category_key, target_date):
    if not metrics_to_plot:
        st.error(f"❌ No valid metrics found to plot for {category_key}.")
        return
    df_grouped = df.groupby(df['REPORT_DATE'].dt.date, as_index=False).mean()
    df_long = df_grouped.melt(id_vars=['REPORT_DATE'],
                              value_vars=[col for _, col in metrics_to_plot],
                              var_name='Metric', value_name='Value')
    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X('REPORT_DATE:T', title="Report Date"),
        y=alt.Y('Value:Q', title="Values"),
        color='Metric:N',
        tooltip=['REPORT_DATE', 'Metric', 'Value']
    ).properties(
        title=f"{category_key} - Daily Aggregated Values",
        width=800,
        height=400
    ).configure_axis(labelAngle=45)
    st.altair_chart(chart, use_container_width=True)

def plot_combined_graph_CO2(graph_query_input=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_CO2%'")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        st.warning("⚠️ No CO2 files found in the tracking database.")
        return
    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            st.error(f"⚠️ Error reading {file_name}: {e}")
    if not data:
        st.error("❌ No valid 'daily_totals' found in CO2 files.")
        return
    df_graph = pd.DataFrame(data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in CO2 JSON data.")
        return
    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])
    plt.figure(figsize=(12, 6))
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_VOLUME'], label='Total Volume', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_TRDVAL'], label='Total TRDVAL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_MKTVAL'], label='Total MKTVAL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_TRDPRC'], label='Total TRDPRC', alpha=0.6)
    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title('CO2: Comparison of Volume, TRDVAL, MKTVAL, and TRDPRC')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

def plot_combined_graph_NG(graph_query_input=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_NG%'")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        st.warning("⚠️ No NG files found in the tracking database.")
        return
    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            st.error(f"⚠️ Error reading {file_name}: {e}")
    if not data:
        st.error("❌ No valid 'daily_totals' found in NG files.")
        return
    df_graph = pd.DataFrame(data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in NG JSON data.")
        return
    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])
    plt.figure(figsize=(12, 6))
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_VOLUME'], label='Total Volume', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_VOLUME_TOTAL'], label='Total Volume Total', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_QTY_PHY'], label='Total QTY PHY', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_MKT_VAL'], label='Total MKT VAL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_QTY_FIN'], label='Total QTY FIN', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_TRD_VAL'], label='Total TRD VAL', alpha=0.6)
    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title('NG: Comparison of Volume, QTY, MKTVAL, TRDVAL, etc.')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

def plot_combined_graph_PW(graph_query_input=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_PW%'")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        st.warning("⚠️ No PW files found in the tracking database.")
        return
    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            st.error(f"⚠️ Error reading {file_name}: {e}")
    if not data:
        st.error("❌ No valid 'daily_totals' found in PW files.")
        return
    df_graph = pd.DataFrame(data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in PW JSON data.")
        return
    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])
    plt.figure(figsize=(14, 7))
    plt.bar(df_graph['REPORT_DATE'], df_graph['VOLUME_BL'], label='VOLUME_BL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['VOLUME_PK'], label='VOLUME_PK', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['VOLUME_OFPK'], label='VOLUME_OFPK', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['MKT_VAL_BL'], label='MKT_VAL_BL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['MKT_VAL_PK'], label='MKT_VAL_PK', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['MKT_VAL_OFPK'], label='MKT_VAL_OFPK', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TRD_VAL_BL'], label='TRD_VAL_BL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TRD_VAL_PK'], label='TRD_VAL_PK', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TRD_VAL_OFPK'], label='TRD_VAL_OFPK', alpha=0.6)
    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title('PW: Comparison of Volume, MKT_VAL, TRD_VAL across types')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

def get_feedback_logs():
    conn = sqlite3.connect(DB_NAME)
    query = """
    SELECT id, user_query, query_responce, user_reaction, user_feedback, created_at
    FROM feedback_log
    """
    feedback_df = pd.read_sql_query(query, conn)
    conn.close()
    feedback_df["User Feedback"] = feedback_df["user_reaction"].apply(lambda x: "👍 Yes" if x == 1 else "👎 No")
    feedback_df["Timestamp"] = pd.to_datetime(feedback_df["created_at"])
    return feedback_df

# --- User Authentication Interface ---
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        col1_img, col2_img, col3_img = st.columns(3)
        with col2_img:
            st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95", width=150)
        st.title("🔐 Welcome to ETAI Energy Trading AI Platform")
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            st.header("🔑 Login")
            username = st.text_input("Username", key="Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="password", placeholder="Enter your password")
            if st.button("Login", use_container_width=True, key="login", help="Click to login"):
                if authenticate_user(username, password):
                    st.success("✅ Login successful!")
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
        with tab2:
            st.header("📝 Register")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Register", use_container_width=True):
                if register_user(new_username, new_password):
                    st.success("✅ User registered successfully!")
                else:
                    st.error("❌ Username already exists. Choose a different one.")

# --- Dashboard ---
else:
    sidebar_menu()
    top_right_menu()
    st.title(f"📊 {st.session_state.main_section}")

    if st.session_state.sub_section == "Pipeline Dashboard":
        st.subheader("📈 Pipeline Dashboard")

    elif st.session_state.sub_section == "Data Pipeline":
        st.subheader("🔧 Data Pipeline -> Energy Training")
        st.write("Upload and manage your data files efficiently.")
        st.markdown("---")

        # Section 1: CO2 Data Upload
        st.markdown("### 🗂️ CO2 Data Upload")
        col1, col2 = st.columns(2)
        with col1:
            metadata_file1 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file1")
        with col2:
            raw_files1 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files1", accept_multiple_files=True)
        if metadata_file1:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file1.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df)
                st.success("✅ Metadata processed successfully!")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing metadata: {e}")
        if st.button("Submit", key="co2"):
            if metadata_file1 and raw_files1:
                try:
                    metadata_df = pd.read_csv(BytesIO(metadata_file1.getvalue()))
                    for f in raw_files1:
                        f.seek(0)
                        df = pd.read_csv(f)
                        file_name = f.name
                        if validate_against_metadata(df, metadata_df, file_name):
                            f.seek(0)
                            success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                            st.success(f"✅ VALID: {file_name} | {msg}")
                        else:
                            f.seek(0)
                            success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                            st.error(f"❌ INVALID: {file_name} | {msg}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Please upload metadata and data files!")
        st.markdown("---")

        # Section 2: Natural Gas Data Upload
        st.markdown("### 🔥 Natural Gas Data Upload")
        col3, col4 = st.columns(2)
        with col3:
            metadata_file2 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file2")
        with col4:
            raw_files2 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files2", accept_multiple_files=True)
        if metadata_file2:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file2.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df)
                st.success("✅ Metadata processed successfully!")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing metadata: {e}")
        if st.button("Submit", key="ng"):
            if metadata_file2 and raw_files2:
                try:
                    metadata_df = pd.read_csv(BytesIO(metadata_file2.getvalue()))
                    for f in raw_files2:
                        f.seek(0)
                        df = pd.read_csv(f)
                        file_name = f.name
                        if validate_against_metadata(df, metadata_df, file_name):
                            f.seek(0)
                            success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                            st.success(f"✅ VALID: {file_name} | {msg}")
                        else:
                            f.seek(0)
                            success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                            st.error(f"❌ INVALID: {file_name} | {msg}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Please upload metadata and data files!")
        st.markdown("---")

        # Section 3: Power Data Upload
        st.markdown("### 🚀 Power Data Upload")
        col5, col6 = st.columns(2)
        with col5:
            metadata_file3 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file3")
        with col6:
            raw_files3 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files3", accept_multiple_files=True)
        if metadata_file3:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file3.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df)
                st.success("✅ Metadata processed successfully!")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing metadata: {e}")
        if st.button("Submit", key="po"):
            if metadata_file3 and raw_files3:
                try:
                    metadata_df = pd.read_csv(BytesIO(metadata_file3.getvalue()))
                    for f in raw_files3:
                        f.seek(0)
                        df = pd.read_csv(f)
                        file_name = f.name
                        if validate_against_metadata(df, metadata_df, file_name):
                            f.seek(0)
                            success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                            st.success(f"✅ VALID: {file_name} | {msg}")
                        else:
                            f.seek(0)
                            success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                            st.error(f"❌ INVALID: {file_name} | {msg}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Please upload metadata and data files!")
        st.markdown("---")

    elif st.session_state.sub_section == "Processed Data":
        st.subheader("📊 Processed Data")
        st.write("Access and analyze the processed data records.")

    elif st.session_state.sub_section == "Dashboard":
        st.subheader("📚 Dashboard")
        st.write("Manage vector embeddings and storage.")

    elif st.session_state.sub_section == "Configure & Upload":
        st.subheader("🏗️ Configure & Upload Data")
        prefix_fields = {
            "CO2": ['VOLUME', 'TRDVAL', 'MKTVAL', 'TRDPRC'],
            "Natural Gas": ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL'],
            "Power": ['VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK', 'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
        }
        name = st.text_input("RAG  Agent Name")
        col1, col2 = st.columns(2)
        with col1:
            bucket = st.selectbox("Select Bucket Name(S3)", ["etrm-eita-poc-chub","etrm-etai-poc", "etrm-etai-poc-ng"])
        with col2:
            prefix = st.selectbox("Select Prefix", list(prefix_fields.keys()), index=0)
        col3, col4 = st.columns(2)
        with col3:
            model = st.selectbox("Model", ["OpenAI GPT-3.5", "OpenAI GPT-4", "Llama 2", "Claude 3.5", "Claude 4", "Custom Model"])
        with col4:
            temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
        col5, col6 = st.columns(2)
        with col5:
            metadata_file = st.file_uploader("Upload Data Dictionary (CSV)", type=["csv"])
        with col6:
            uploaded_file = st.file_uploader("Upload Transaction Log (TXT, PDF, CSV, DOCX)", type=["txt", "pdf", "csv", "docx"])
        prompt = st.text_area("📝 Provide Prompt Instructions", key='prompt')
        if st.button("Submit & Process Data"):
            prefix_value = {"CO2": "CO2", "Natural Gas": "NG", "Power": "PW"}.get(prefix, "misc")
            process_files_from_s3_folder(VALID_BUCKET, prefix_value)
            add_agent_detail(name, model, temp, prompt)

    elif st.session_state.sub_section == "Fine Tuning":
        st.subheader("📄 Fine Tuning")

    elif st.session_state.sub_section == "Settings":
        st.subheader("🚀 Settings")
        st.write("Get insights into your application's performance.")

    elif st.session_state.sub_section in ["Energy Trading Analysis", "Energy Tradeing Analysis"]:
        if "query_answer" not in st.session_state:
            st.session_state["query_answer"] = None
        if "feedback_submitted" not in st.session_state:
            st.session_state["feedback_submitted"] = False

        st.subheader("📊 Energy Trading Analysis")
        st.write("💬 Ask Your Financial or Commodity Question")

        category_options = ["CO2", "Natural Gas", "Power"]
        selected_category = st.selectbox("Select Category", category_options)
        user_query = st.text_input("Example: What is the total Price Value on 13 Nov 2024?")

        query_answer = None
        if st.button("Submit Query") and user_query:
            with st.spinner("Thinking..."):
                # Your existing query function
                query_answer = query_sqlite_json_with_openai(user_query, selected_category)

                # Store in SQLite database
                log_id = add_feedback_log(user_query, query_answer, selected_category)

                st.session_state["query_answer"] = query_answer
                st.session_state["log_id"] = log_id
                st.session_state["feedback_submitted"] = False

        if st.session_state["query_answer"]:
            st.success(st.session_state["query_answer"])

            st.subheader("🧠 Confidence Score:")
            st.markdown(f'<div class="confidence-score">Confidence: {round(st.session_state.confidence_score * 100)}%</div>', unsafe_allow_html=True)

            st.subheader("Was this response helpful?")
            feedback = st.radio("Feedback:", ["👍 Yes", "👎 No"], key="feedback_radio", horizontal=True)
            feedback_comment = st.text_area("Additional comments or corrections:", height=100, key="feedback_comment")

            if st.button("Submit Feedback"):
                if feedback == "👎 No" and not feedback_comment.strip():
                    st.warning("Please add comments or corrections before submitting negative feedback.")
                else:
                    if not st.session_state["feedback_submitted"]:
                        # Update feedback in SQLite
                        update_feedback_log(feedback, feedback_comment, st.session_state["log_id"])

                        st.session_state["feedback_submitted"] = True
                        st.success("✅ Feedback submitted successfully!")

                        if feedback == "👎 No" and feedback_comment.strip():
                            st.info("Your feedback will be used to improve future responses.")
                    else:
                        st.info("Feedback already submitted.")

    elif st.session_state.sub_section == "Graph Query":
        st.subheader("📊 Graph Query")
        category_selected = st.selectbox("Select Category", ["CO2", "Natural Gas", "Power"])
        graph_query_input = st.text_input('Ask for a bar graph (e.g., "Show bar graph for TRDVAL")')
        if st.button('Generate Custom Graph'):
            plot_graph_based_on_prompt_all(graph_query_input, category_selected)
        if st.button('Combined CO2 Graph'):
            plot_combined_graph_CO2(graph_query_input)
        if st.button('Combined NG Graph'):
            plot_combined_graph_NG(graph_query_input)
        if st.button('Combined PW Graph'):
            plot_combined_graph_PW(graph_query_input)

    elif st.session_state.sub_section == "Deviation Analysis":
        st.subheader("📊 Deviation Analysis")

    elif st.session_state.sub_section == "Root Cause Analysis":
        st.subheader("⚠️ Root Cause Analysis")
        st.write("Review and troubleshoot errors.")
        st.header("📊 Analysis History & Insights")

    elif st.session_state.sub_section == "User Feedback":
        st.subheader("📝 User Feedback Dashboard")
        feedback_df = get_feedback_logs()
        if not feedback_df.empty:
            st.subheader("📋 Collected Feedback")
            display_df = feedback_df[["id", "user_query", "query_responce", "User Feedback", "user_feedback", "Timestamp"]]
            st.dataframe(display_df, use_container_width=True)
            st.subheader("📊 Feedback Summary")
            positive_feedback_count = (feedback_df["user_reaction"] == 1).sum()
            negative_feedback_count = (feedback_df["user_reaction"] == 0).sum()
            st.write(f"✅ **Positive Feedback:** {positive_feedback_count}")
            st.write(f"❌ **Negative Feedback:** {negative_feedback_count}")
            st.subheader("📊 Feedback Trends Over Time")
            feedback_over_time = (
                feedback_df.groupby(feedback_df["Timestamp"].dt.date)["User Feedback"]
                .value_counts()
                .unstack()
                .fillna(0)
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            feedback_over_time.plot(kind="bar", ax=ax, stacked=True)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Feedbacks")
            ax.set_title("User Feedback Trends")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            csv_feedback = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Feedback as CSV", csv_feedback, "user_feedback.csv", "text/csv", key="download-feedback")
        else:
            st.warning("No feedback logs found in the database.")
