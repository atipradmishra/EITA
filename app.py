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
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from openai import OpenAI
from business_context import load_business_context
from DataProcessing.GraphDataProcess import process_and_save, save_jsondata_to_db, process_csv_data_to_json , save_rawdata_to_db
from GraphFunctions.dashboardgraphs import show_top5_movers,plot_delta_market_value_by_horizon, plot_delta_market_value_by_horizon_by_tgroup1, plot_delta_volume_by_horizon, plot_delta_volume_by_horizon_by_tgroup1, plot_delta_volume_by_horizon_by_bookattr8,plot_delta_market_value_by_horizon_by_bookattr8
from GraphFunctions.heatmaptable import plot_heatmap
from GraphFunctions.dashboardcards import show_nop_cards,render_summary_card

# --- Streamlit page configuration ---
st.set_page_config(page_title="EITA", layout="wide")

EMBEDDING_MODEL = "text-embedding-ada-002"
DB_NAME = "vector_chunks.db"
COMPUTED_DB = "data.db"
VALID_BUCKET = "etrm-etai-poc"
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
    conn.commit()
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
        aws_secret_access_key=aws_secret_key,
        region_name='us-east-1'
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
            print(file_name)
            cursor.execute("SELECT 1 FROM file_tracking WHERE file_name = ? AND is_processed = 1", (file_name,))

            if cursor.fetchone():
                print(f"⏭️ Skipping already processed file: {file_name}")
                continue

            try:
                obj_data = s3.get_object(Bucket=bucket_name, Key=file_key)
                file_stream = io.BytesIO(obj_data["Body"].read())
                df = pd.read_csv(file_stream)
                df_processed = process_and_store_file(file_name, df, cursor,folder_prefix)
                print(f"✅ df_processed: {df_processed}")
                conn.commit()
                print(f"✅ Processed and saved: {file_name}")
            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")

    except Exception as e:
        print(f"❌ Error accessing S3: {e}")
    finally:
        conn.close()

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        
        /* Title styling */
        .card-title {
            color: #666;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        /* Value styling */
        .card-value {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        /* Delta styling */
        .card-delta {
            color: var(--secondary-color);
            font-size: 18px;
            font-weight: 500;
        }
        
        /* Update styling */
        .card-update {
            color: #aaa;
            font-size: 12px;
            margin-top: 8px;
        }
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

# --- Sidebar and User Menu ---
def sidebar_menu():
    with st.sidebar:
        col1 , col2, col3 = st.columns(3)
        with col1:
            st.write(" ")
        with col2:
            st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95")
        with col3:
            st.write(" ")
        st.markdown("<h2 style='text-align: center;'>ETAI Energy Trading AI Platform</h2>", unsafe_allow_html=True)
        st.divider()
        main_section_options = ["Data Management AI Agent", "RAG AI Agent", "Application AI Agent"]
        main_section = st.radio("Select AI Agent", main_section_options, index=main_section_options.index(st.session_state.main_section))
        st.session_state.main_section = main_section

        sub_sections = {
            "Data Management AI Agent": ["Pipeline Dashboard", "Data Pipeline", "Processed Data"],
            "RAG AI Agent": ["RAG Dashboard", "Configure & Upload", "Fine Tuning", "Settings"],
            "Application AI Agent": ["Dashboard","Energy Tradeing Analysis", "Graph Query", "Deviation Analysis", "Root Cause Analysis", "Analysis History", "User Feedback"]
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
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )
        s3.upload_fileobj(file, bucket, s3_key)


        return True, f"✅ Uploaded {filename} to S3 bucket '{bucket}' in folder '{folder}'"
    except Exception as e:
        return False, f"❌ Upload failed: {e}"

def save_metadata_to_db(filename, df, s3_path):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    json_data = df.to_json(orient="records")
    cursor.execute("""
        INSERT INTO metadata_files (filename, s3_path, json_data)
        VALUES (?, ?, ?)
    """, (filename, s3_path, json_data))
    conn.commit()
    conn.close()


def upload_metadatafile_to_s3(file, filename, bucket):
    try:
        folder = "misc/"
        if "CO2" in filename.upper():
            folder = "CO2/metadata/"
        elif "NG" in filename.upper():
            folder = "NG/metadata/"
        elif "PW" in filename.upper():
            folder = "PW/metadata/"

        s3_key = folder + filename

        file.seek(0)
        file_bytes = file.read()

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )
        s3.upload_fileobj(io.BytesIO(file_bytes), bucket, s3_key)

        df = pd.read_csv(io.BytesIO(file_bytes))
        save_metadata_to_db(filename, df, s3_key)

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
                return pd.to_datetime(report_date, format=fmt, dayfirst=True)
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(report_date, dayfirst=True, errors='coerce')
        except:
            pass
        print(f"Warning: Could not parse date: {report_date}")
        return None

    df['REPORT_DATE'] = df['REPORT_DATE'].astype(str).str.strip()
    df['REPORT_DATE'] = df['REPORT_DATE'].apply(parse_dates)

    # Filter to keep only the latest report date
    latest_date = df['REPORT_DATE'].max()
    df = df[df['REPORT_DATE'] == latest_date]

    # Convert back to string format for grouping
    df['REPORT_DATE'] = df['REPORT_DATE'].apply(lambda dt: dt.strftime('%Y-%m-%d') if pd.notnull(dt) else None)

    # Determine numeric columns and groupings by folder
    if folder_prefix == 'CO2':
        numeric_cols = ['VOLUME', 'MKTVAL', 'TRDVAL', 'TRDPRC']
        grouping_columns = ['REPORT_DATE']
        additional_groupings = ['BOOK', 'SEGMENT', 'TGROUP1']

    elif folder_prefix == 'NG':
        numeric_cols = ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL']
        grouping_columns = ['REPORT_DATE']
        additional_groupings = ['BOOK', 'SEGMENT', 'HORIZON', 'TGROUP1']

    elif folder_prefix == 'PW':
        numeric_cols = ['VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK',
                        'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK',
                        'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
        grouping_columns = ['REPORT_DATE']
        additional_groupings = ['BOOK', 'SEGMENT', 'HORIZON', 'TGROUP1']

    else:
        raise ValueError(f"❌ Unknown folder prefix: {folder_prefix}")

    # Clean and convert numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=True), errors='coerce').fillna(0)

    # Add optional groupings if available
    for col in additional_groupings:
        if col in df.columns:
            grouping_columns.append(col)

    # Aggregation logic by folder type
    if folder_prefix == 'CO2':
        df_daily_stats = df.groupby(grouping_columns).agg(
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
        df_daily_stats = df.groupby(grouping_columns).agg(
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

    # Convert output to JSON for storage
    combined_json = {
        "daily_totals": df_daily_stats.to_dict(orient='records')
    }
    sanitized_json = sanitize_for_json(combined_json)

    # Use the actual REPORT_DATE from the data
    date_processed = df_daily_stats['REPORT_DATE'].max() if 'REPORT_DATE' in df_daily_stats.columns else None

    cursor.execute(
        "INSERT INTO file_tracking (file_name, date_processed, json_contents, is_processed) VALUES (?, ?, ?, 1)",
        (file_name, date_processed, json.dumps(sanitized_json))
    )

    return df


def query_sqlite_json_with_openai(user_question, category=None):
    # Load business dictionary
    business_dict = load_business_context()

    # Step 1: Load Feedback and FAISS Index
    feedback_data = load_feedback_data()
    faiss_index, feedback_data_indexed = create_faiss_index(feedback_data)
    feedback_insights = retrieve_feedback_insights(user_question, faiss_index, feedback_data_indexed)

    # Step 2: Query SQLite data
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
    
    #Prompt Instruction
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if category:
        if category == "CO2":
            cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'CO2 AGENT'")
        elif category == "Natural Gas":
            cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'NG AGENT'")
        elif category == "Power":
            cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'PW AGENT'")
    else:
        cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'PW AGENT'")

    prompt_Instr = cursor.fetchall()
    print(f"Prompt Instruction, {prompt_Instr}")

    conn.close()

    if not prompt_Instr:
        return "⚠️ No Prompt Instruction Found."
    
    # Step 3: Build context from retrieved JSON
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

    # Step 4: Prepare business glossary
    business_context_text = "\n".join(
        f"{key}: {value}" if isinstance(value, str)
        else f"{key}: {', '.join(f'{k} = {v}' for k, v in value.items())}"
        for key, value in business_dict.items()
    )

    # Step 5: Compose system prompt
    feedback_text = ""
    
    if feedback_insights:
        feedback_text = "\n\nBased on feedback from similar queries, please be aware of these issues:\n"
        feedback_text += "\n".join(f"- {insight}" for insight in feedback_insights)
        print(feedback_text)
    category_context = f"for the {category} category" if category and category != "All" else ""

    context_message = f"""
    📌 VERIFIED FEEDBACK (Authoritative Corrections):
    Use this section as the most accurate reference if it directly answers the question.
    {feedback_text}

    📘 BUSINESS GLOSSARY (Terms & Labels):
    This glossary helps you interpret technical terms into business language when answering user questions.
    {business_context_text}

    🗂️ PREPROCESSED DATA (Official Trading Statistics):
    This section contains structured trading data retrieved from SQLite for multiple files {category_context}.
    {all_context}

    ⚠️ INSTRUCTION (STRICTLY FOLLOW THESE STEPS — DO NOT IGNORE):
    1. If the VERIFIED FEEDBACK section directly answers the question, use it as the ONLY source.
    2. If the feedback is not sufficient, use PREPROCESSED DATA and BUSINESS GLOSSARY to answer.
    3. Always translate technical codes (e.g., ELCE, NG, TGROUP1) into business-friendly terms using the glossary.
    4. Provide a clear, professional, and concise answer suitable for business users.
    5. End your answer with 2–3 relevant follow-up questions based on the question and data.

    DO NOT DEVIATE FROM THESE STEPS. ANSWERS MUST FOLLOW THIS EXACT LOGIC.
    {prompt_Instr}
    """
    
    # Step 6: Construct the conversation and call OpenAI
    st.session_state.conversation.append({"role": "system", "content": context_message})
    st.session_state.conversation.append({"role": "user", "content": user_question})

    for idx, message in enumerate(st.session_state.conversation):
        print(f"\n🔹 Message {idx+1} ({message['role']}):\n{message['content']}\n{'-'*60}")

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=st.session_state.conversation,
        temperature=0.7
    )

    gpt_answer = response.choices[0].message.content.strip()
    st.session_state.conversation.append({"role": "assistant", "content": gpt_answer})

    if len(st.session_state.conversation) > 6:
        st.session_state.conversation = [st.session_state.conversation[0]] + st.session_state.conversation[-4:]

    confidence_score = calculate_confidence_score(0.7, feedback_insights, file_names)
    st.session_state.confidence_score = confidence_score

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
        # Build the TF-IDF representation using the "Query" texts
        vectorizer.fit([item.get("Query", "") for item in feedback_data_indexed if item.get("Query", "")])
        query_vector = vectorizer.transform([query]).astype(np.float32).toarray()
        distances, indices = faiss_index.search(query_vector, top_k)
        insights = []
        for idx in indices[0]:
            if idx < len(feedback_data_indexed):
                feedback_item = feedback_data_indexed[idx]
                # Check for negative feedback by verifying if the stored value is 0.
                if feedback_item.get("User Feedback") == 0 and feedback_item.get("Feedback Comment", "").strip():
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

def plot_graph_based_on_prompt_all(prompt, category_key):
    import sqlite3
    import json
    import re
    import pandas as pd
    import streamlit as st

    # Connect to the database and fetch processed file JSON contents
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        st.warning("⚠️ No processed JSON data found in the database.")
        return

    # Define category metadata for field mapping (for plotting)
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
                'AVG_VOLUME': 'AVG_VOLUME',
                'MIN_VOLUME': 'MIN_VOLUME',
                'MAX_VOLUME': 'MAX_VOLUME',
                'STD_VOLUME': 'STD_VOLUME',
                'VOLUME_TOTAL': 'TOTAL_VOLUME_TOTAL',
                'AVG_VOLUME_TOTAL': 'AVG_VOLUME_TOTAL',
                'MIN_VOLUME_TOTAL': 'MIN_VOLUME_TOTAL',
                'MAX_VOLUME_TOTAL': 'MAX_VOLUME_TOTAL',
                'STD_VOLUME_TOTAL': 'STD_VOLUME_TOTAL',
                'QTY_PHY': 'TOTAL_QTY_PHY',
                'AVG_QTY_PHY': 'AVG_QTY_PHY',
                'MIN_QTY_PHY': 'MIN_QTY_PHY',
                'MAX_QTY_PHY': 'MAX_QTY_PHY',
                'STD_QTY_PHY': 'STD_QTY_PHY',
                'MKT_VAL': 'TOTAL_MKT_VAL',
                'AVG_MKT_VAL': 'AVG_MKT_VAL',
                'MIN_MKT_VAL': 'MIN_MKT_VAL',
                'MAX_MKT_VAL': 'MAX_MKT_VAL',
                'STD_MKT_VAL': 'STD_MKT_VAL',
                'QTY_FIN': 'TOTAL_QTY_FIN',
                'AVG_QTY_FIN': 'AVG_QTY_FIN',
                'MIN_QTY_FIN': 'MIN_QTY_FIN',
                'MAX_QTY_FIN': 'MAX_QTY_FIN',
                'STD_QTY_FIN': 'STD_QTY_FIN',
                'TRD_VAL': 'TOTAL_TRD_VAL',
                'AVG_TRD_VAL': 'AVG_TRD_VAL',
                'MIN_TRD_VAL': 'MIN_TRD_VAL',
                'MAX_TRD_VAL': 'MAX_TRD_VAL',
                'STD_TRD_VAL': 'STD_TRD_VAL'
            }
        },
        "Power": {
            "prefix": "PW",
            "fields": {
                'VOLUME_BL': 'TOTAL_VOLUME_BL',
                'AVG_VOLUME_BL': 'AVG_VOLUME_BL',
                'MIN_VOLUME_BL': 'MIN_VOLUME_BL',
                'MAX_VOLUME_BL': 'MAX_VOLUME_BL',
                'VOLUME_PK': 'TOTAL_VOLUME_PK',
                'AVG_VOLUME_PK': 'AVG_VOLUME_PK',
                'MIN_VOLUME_PK': 'MIN_VOLUME_PK',
                'MAX_VOLUME_PK': 'MAX_VOLUME_PK',
                'VOLUME_OFPK': 'TOTAL_VOLUME_OFPK',
                'AVG_VOLUME_OFPK': 'AVG_VOLUME_OFPK',
                'MIN_VOLUME_OFPK': 'MIN_VOLUME_OFPK',
                'MAX_VOLUME_OFPK': 'MAX_VOLUME_OFPK',
                'MKT_VAL_BL': 'TOTAL_MKT_VAL_BL',
                'AVG_MKT_VAL_BL': 'AVG_MKT_VAL_BL',
                'MIN_MKT_VAL_BL': 'MIN_MKT_VAL_BL',
                'MAX_MKT_VAL_BL': 'MAX_MKT_VAL_BL',
                'MKT_VAL_PK': 'TOTAL_MKT_VAL_PK',
                'AVG_MKT_VAL_PK': 'AVG_MKT_VAL_PK',
                'MIN_MKT_VAL_PK': 'MIN_MKT_VAL_PK',
                'MAX_MKT_VAL_PK': 'MAX_MKT_VAL_PK',
                'MKT_VAL_OFPK': 'TOTAL_MKT_VAL_OFPK',
                'AVG_MKT_VAL_OFPK': 'AVG_MKT_VAL_OFPK',
                'MIN_MKT_VAL_OFPK': 'MIN_MKT_VAL_OFPK',
                'MAX_MKT_VAL_OFPK': 'MAX_MKT_VAL_OFPK',
                'TRD_VAL_BL': 'TOTAL_TRD_VAL_BL',
                'AVG_TRD_VAL_BL': 'AVG_TRD_VAL_BL',
                'MIN_TRD_VAL_BL': 'MIN_TRD_VAL_BL',
                'MAX_TRD_VAL_BL': 'MAX_TRD_VAL_BL',
                'TRD_VAL_PK': 'TOTAL_TRD_VAL_PK',
                'AVG_TRD_VAL_PK': 'AVG_TRD_VAL_PK',
                'MIN_TRD_VAL_PK': 'MIN_TRD_VAL_PK',
                'MAX_TRD_VAL_PK': 'MAX_TRD_VAL_PK',
                'TRD_VAL_OFPK': 'TOTAL_TRD_VAL_OFPK',
                'AVG_TRD_VAL_OFPK': 'AVG_TRD_VAL_OFPK',
                'MIN_TRD_VAL_OFPK': 'MIN_TRD_VAL_OFPK',
                'MAX_TRD_VAL_OFPK': 'MAX_TRD_VAL_OFPK'
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

    # Extract target date from prompt (if provided)
    date_patterns = [
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})',
        r'(\d{1,2})([A-Za-z]{3})(\d{2,4})'
    ]
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

    # Process only the "daily_totals" data from each file's JSON content
    all_data = []
    for file_name, json_text in rows:
        if not file_name.startswith(expected_prefix):
            continue
        try:
            json_data = json.loads(json_text)
            st.write(f"Processing file: **{file_name}**")
            if "daily_totals" in json_data:
                all_data.extend(json_data["daily_totals"])
        except Exception as e:
            st.error(f"⚠️ Error reading {file_name}: {e}")

    if not all_data:
        st.error(f"❌ No valid data found for {category_key}.")
        return

    # Convert the aggregated list into a DataFrame
    df_graph = pd.DataFrame(all_data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in JSON data.")
        return

    # Ensure that REPORT_DATE is interpreted as a datetime type
    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])

    # Optionally: Convert key metric columns from strings to numeric
    for key, col_name in fields_to_plot.items():
        if col_name in df_graph.columns:
            df_graph[col_name] = pd.to_numeric(df_graph[col_name], errors='coerce')

    # If a target date was provided in the prompt, filter the DataFrame for that date
    if target_date is not None:
        df_graph = df_graph[df_graph['REPORT_DATE'].dt.date == target_date.date()]
        if df_graph.empty:
            st.warning(f"⚠️ No data found for date: {target_date.strftime('%Y-%m-%d')}")
            return

    # Determine which metrics to plot based on the prompt's content;
    # if none are specifically mentioned, default to the first metric available
    normalized_prompt = prompt.lower().replace(" ", "").replace("_", "")
    metrics_to_plot = []
    for key, col_name in fields_to_plot.items():
        normalized_key = key.lower().replace("_", "")
        if col_name in df_graph.columns and (normalized_key in normalized_prompt or not metrics_to_plot):
            metrics_to_plot.append((key, col_name))

    # Plot using a single-date approach if all data belongs to one unique date,
    # otherwise plot as a time series.
    if target_date is not None and df_graph['REPORT_DATE'].nunique() == 1:
        plot_single_date_metrics(df_graph, metrics_to_plot, category_key, target_date)
    else:
        plot_time_series(df_graph, metrics_to_plot, category_key, target_date)


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
    import altair as alt
    import streamlit as st

    if not metrics_to_plot:
        st.error(f"❌ No valid metrics found to plot for {category_key}.")
        return

    # Ensure REPORT_DATE is datetime and create a new 'Date' column for grouping
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
    df['Date'] = df['REPORT_DATE'].dt.date

    # Group only the numeric columns by the new 'Date' column.
    # The numeric_only=True ensures that non-numeric columns are excluded.
    df_grouped = df.groupby('Date', as_index=False).mean(numeric_only=True)

    # Transform the grouped DataFrame into a long format using 'Date' as the id_var
    df_long = df_grouped.melt(id_vars=['Date'],
                              value_vars=[col for _, col in metrics_to_plot],
                              var_name='Metric', value_name='Value')
    
    # Plotting the time series chart
    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X('Date:T', title="Report Date"),
        y=alt.Y('Value:Q', title="Values"),
        color='Metric:N',
        tooltip=['Date', 'Metric', 'Value']
    ).properties(
        title=f"{category_key} - Daily Aggregated Values",
        width=800,
        height=400
    ).configure_axis(labelAngle=45)
    
    st.altair_chart(chart, use_container_width=True)

def extract_date_from_query(query):
    if not query:
        return None
    # Match dates in DD-MM-YYYY or DD/MM/YYYY format
    match = re.search(r'(\d{2})[-/](\d{2})[-/](\d{4})', query)
    if match:
        day, month, year = match.groups()
        try:
            return datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y").date()
        except ValueError:
            return None
    return None

def plot_combined_graph_CO2(graph_query_input=None):
    # ⏱ Extract date from query if available
    target_date = extract_date_from_query(graph_query_input)

    # 📥 Fetch data from database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_CO2%'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        st.warning("⚠️ No CO2 files found in the tracking database.")
        return

    # 📦 Load data from JSON
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

    # 📊 Create DataFrame
    df_graph = pd.DataFrame(data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in CO2 JSON data.")
        return

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE']).dt.date

    # 🎯 Filter by date if given
    if target_date:
        df_graph = df_graph[df_graph['REPORT_DATE'] == target_date]
        if df_graph.empty:
            st.warning(f"⚠️ No data available for {target_date.strftime('%d-%m-%Y')}")
            return

    # 🧹 Convert metric columns to numeric
    metrics = ['TOTAL_VOLUME', 'TOTAL_TRDVAL', 'TOTAL_MKTVAL', 'TOTAL_TRDPRC']
    for col in metrics:
        df_graph[col] = pd.to_numeric(df_graph[col], errors='coerce')

    # 📈 Group and sum (if multi-day)
    df_grouped = df_graph.groupby('REPORT_DATE')[metrics].sum().reset_index()

    # 🖼 Set positions for grouped bars
    x = np.arange(len(df_grouped['REPORT_DATE']))
    width = 0.2

    plt.figure(figsize=(14, 6))

    plt.bar(x - 1.5 * width, df_grouped['TOTAL_VOLUME'], width, label='Volume')
    plt.bar(x - 0.5 * width, df_grouped['TOTAL_TRDVAL'], width, label='TRDVAL')
    plt.bar(x + 0.5 * width, df_grouped['TOTAL_MKTVAL'], width, label='MKTVAL')
    plt.bar(x + 1.5 * width, df_grouped['TOTAL_TRDPRC'], width, label='TRDPRC')

    # 💄 Format y-axis
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.xlabel('Report Date')
    plt.ylabel('Values')
    title_str = f"CO2: Combined Graph"
    if target_date:
        title_str += f" on {target_date.strftime('%d-%b-%Y')}"
    plt.title(title_str)

    plt.xticks(ticks=x, labels=[d.strftime('%Y-%m-%d') for d in df_grouped['REPORT_DATE']], rotation=45)
    plt.legend()
    plt.tight_layout()

    st.pyplot(plt)


def plot_combined_graph_NG(graph_query_input=None):
    # ⏱ Extract date from query if available
    target_date = extract_date_from_query(graph_query_input)

    # 📥 Fetch data from database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_NG%'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        st.warning("⚠️ No NG files found in the tracking database.")
        return

    # 📦 Load data from JSON
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

    # 📊 Create DataFrame
    df_graph = pd.DataFrame(data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in NG JSON data.")
        return

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE']).dt.date

    # 🎯 Filter by date if given
    if target_date:
        df_graph = df_graph[df_graph['REPORT_DATE'] == target_date]
        if df_graph.empty:
            st.warning(f"⚠️ No data available for {target_date.strftime('%d-%m-%Y')}")
            return

    # 🧹 Convert metric columns to numeric
    metrics = [
        'TOTAL_VOLUME',
        'TOTAL_VOLUME_TOTAL',
        'TOTAL_QTY_PHY',
        'TOTAL_MKT_VAL',
        'TOTAL_QTY_FIN',
        'TOTAL_TRD_VAL'
    ]
    found_metrics = [col for col in metrics if col in df_graph.columns]
    if not found_metrics:
        st.error("❌ None of the expected NG metrics were found in the data.")
        st.info(f"Available columns:\n\n{list(df_graph.columns)}")
        return

    for col in found_metrics:
        df_graph[col] = pd.to_numeric(df_graph[col], errors='coerce')

    # 📈 Group and sum (if multi-day)
    df_grouped = df_graph.groupby('REPORT_DATE')[found_metrics].sum().reset_index()

    # 🖼 Set positions for grouped bars
    x = np.arange(len(df_grouped['REPORT_DATE']))
    width = 0.12
    plt.figure(figsize=(16, 6))

    for i, col in enumerate(found_metrics):
        plt.bar(x + (i - len(found_metrics) / 2) * width, df_grouped[col], width, label=col)

    # 💄 Format y-axis
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.xlabel('Report Date')
    plt.ylabel('Values')
    title_str = "NG: Combined Graph"
    if target_date:
        title_str += f" on {target_date.strftime('%d-%b-%Y')}"
    plt.title(title_str)

    plt.xticks(ticks=x, labels=[d.strftime('%Y-%m-%d') for d in df_grouped['REPORT_DATE']], rotation=45)
    plt.legend()
    plt.tight_layout()

    st.pyplot(plt)


def plot_combined_graph_PW(graph_query_input=None):
    target_date = extract_date_from_query(graph_query_input)

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

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE']).dt.date

    if target_date:
        df_graph = df_graph[df_graph['REPORT_DATE'] == target_date]
        if df_graph.empty:
            st.warning(f"⚠️ No data available for {target_date.strftime('%d-%m-%Y')}")
            return

    # ✅ Use the correct column names based on your actual data
    metrics = [
        'TOTAL_VOLUME_BL', 'TOTAL_VOLUME_PK', 'TOTAL_VOLUME_OFPK',
        'TOTAL_MKT_VAL_BL', 'TOTAL_MKT_VAL_PK', 'TOTAL_MKT_VAL_OFPK',
        'TOTAL_TRD_VAL_BL', 'TOTAL_TRD_VAL_PK', 'TOTAL_TRD_VAL_OFPK'
    ]

    available_metrics = [col for col in metrics if col in df_graph.columns]

    if not available_metrics:
        st.error("❌ None of the expected volume/trade/market value metrics were found in the data.")
        st.write("Available columns are:", df_graph.columns.tolist())
        return

    for col in available_metrics:
        df_graph[col] = pd.to_numeric(df_graph[col], errors='coerce')

    df_grouped = df_graph.groupby('REPORT_DATE')[available_metrics].sum().reset_index()
    x = np.arange(len(df_grouped['REPORT_DATE']))
    width = 0.08

    plt.figure(figsize=(15, 6))

    for idx, metric in enumerate(available_metrics):
        offset = (idx - len(available_metrics) / 2) * width
        plt.bar(x + offset, df_grouped[metric], width, label=metric)

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.xlabel('Report Date')
    plt.ylabel('Values')

    title = "PW: Combined Graph"
    if target_date:
        title += f" on {target_date.strftime('%d-%b-%Y')}"
    plt.title(title)

    plt.xticks(ticks=x, labels=[d.strftime('%Y-%m-%d') for d in df_grouped['REPORT_DATE']], rotation=45)
    plt.legend()
    plt.tight_layout()

    st.pyplot(plt)


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


def load_data_for_dashboard():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT json_contents FROM graph_data WHERE file_name = 'latest_combined'")
    json_data = cursor.fetchone()[0]
    data = json.loads(json_data)
    conn.close()
    return data


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


        metadata_df = None
        if metadata_file1:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file1.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df)
                upload_metadatafile_to_s3(metadata_file1, metadata_file1.name, VALID_BUCKET)
                st.success("✅ Metadata processed successfully and uploaded.")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing uploaded metadata: {e}")
        else:
            try:
                metadata_df = get_existing_metadata('CO2')
                if metadata_df is not None:
                    st.success(f"✅ Loaded existing metadata from database.")
                    alias_mapping = process_metadata_alias(metadata_df)
                    st.write("Alias Mapping (From DB):")
                    st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
                else:
                    st.warning(f"⚠️ No metadata found. Please upload the metadata file.")
            except Exception as e:
                st.error(f"❌ Error loading metadata from DB: {e}")

        if st.button("Submit", key="co2"):
            if metadata_df is not None and raw_files1:
                try:
                    for f in raw_files1:
                        try:
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
                        except Exception as file_err:
                            st.error(f"❌ Error processing {file_name}: {file_err}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Metadata and raw files are required to proceed.")

        st.markdown("---")

        st.markdown("### 🔥 Natural Gas Data Upload")
        col3, col4 = st.columns(2)
        with col3:
            metadata_file2 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file2")
        with col4:
            raw_files2 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files2", accept_multiple_files=True)

        dataset_type_ng = "NG"
        metadata_df_ng = None

        if metadata_file2:
            try:
                metadata_df_ng = pd.read_csv(BytesIO(metadata_file2.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df_ng)
                upload_metadatafile_to_s3(metadata_file2, metadata_file2.name, VALID_BUCKET)
                st.success("✅ Metadata processed successfully and uploaded.")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing uploaded metadata: {e}")
        else:
            try:
                metadata_df_ng = get_existing_metadata(dataset_type_ng)
                if metadata_df_ng is not None:
                    st.success(f"✅ Loaded existing metadata for {dataset_type_ng} from database.")
                    alias_mapping = process_metadata_alias(metadata_df_ng)
                    st.write("Alias Mapping (From DB):")
                    st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
                else:
                    st.warning(f"⚠️ No metadata found for {dataset_type_ng}. Please upload the metadata file.")
            except Exception as e:
                st.error(f"❌ Error loading metadata from DB: {e}")

        # Submission
        if st.button("Submit", key="ng"):
            if metadata_df_ng is not None and raw_files2:
                try:
                    for f in raw_files2:
                        try:
                            f.seek(0)
                            df = pd.read_csv(f)
                            file_name = f.name
                            if validate_against_metadata(df, metadata_df_ng, file_name):
                                f.seek(0)
                                success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                                st.success(f"✅ VALID: {file_name} | {msg}")
                            else:
                                f.seek(0)
                                success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                                st.error(f"❌ INVALID: {file_name} | {msg}")
                        except Exception as file_err:
                            st.error(f"❌ Error processing {file_name}: {file_err}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Metadata and raw files are required to proceed.")

        st.markdown("---")

        # Section 3: Power Data Upload
        st.markdown("### 🚀 Power Data Upload")
        col5, col6 = st.columns(2)
        with col5:
            metadata_file3 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file3")
        with col6:
            raw_files3 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files3", accept_multiple_files=True)

        dataset_type_pw = "PW"
        metadata_df_pw = None

        if metadata_file3:
            try:
                metadata_df_pw = pd.read_csv(BytesIO(metadata_file3.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df_pw)
                upload_metadatafile_to_s3(metadata_file3, metadata_file3.name, VALID_BUCKET)
                st.success("✅ Metadata processed successfully and uploaded.")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing uploaded metadata: {e}")
        else:
            try:
                metadata_df_pw = get_existing_metadata(dataset_type_pw)
                if metadata_df_pw is not None:
                    st.success(f"✅ Loaded existing metadata for {dataset_type_pw} from database.")
                    alias_mapping = process_metadata_alias(metadata_df_pw)
                    st.write("Alias Mapping (From DB):")
                    st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
                else:
                    st.warning(f"⚠️ No metadata found for {dataset_type_pw}. Please upload the metadata file.")
            except Exception as e:
                st.error(f"❌ Error loading metadata from DB: {e}")

        if st.button("Submit", key="po"):
            if metadata_df_pw is not None and raw_files3:
                try:
                    for f in raw_files3:
                        file_name = f.name
                        conn = sqlite3.connect(DB_NAME)
                        try:
                                df = pd.read_csv(f)
                                df.replace(",", "", regex=True, inplace=True)
                                df['MKT_VAL_BL'] = df['MKT_VAL_BL'].astype(str).str.replace(',', '').astype(float)
                                df[['VOLUME_BL', 'MKT_VAL_BL']] = df[['VOLUME_BL', 'MKT_VAL_BL']].apply(pd.to_numeric, errors='coerce')
                                df['source_file'] = file_name
                                desired_columns = ['REPORT_DATE', 'SEGMENT', 'TGROUP1', 'BUCKET', 'HORIZON', 'VOLUME_BL', 'MKT_VAL_BL', 'source_file']
                                save_rawdata_to_db(df[desired_columns], conn, file_name)
                        except Exception as e:
                                print(f"❌ Error processing file {file_name}: {e}")
                        try:
                            f.seek(0)
                            df = pd.read_csv(f)
                            if validate_against_metadata(df, metadata_df_pw, file_name):
                                f.seek(0)
                                success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                                st.success(f"✅ VALID: {file_name} | {msg}")
                            else:
                                f.seek(0)
                                success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                                st.error(f"❌ INVALID: {file_name} | {msg}")
                        except Exception as file_err:
                            st.error(f"❌ Error processing {file_name}: {file_err}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Metadata and raw files are required to proceed.")

        st.markdown("---")

    elif st.session_state.sub_section == "Processed Data":
        st.subheader("📊 Processed Data")
        st.write("Access and analyze the processed data records.")
    
    elif st.session_state.sub_section == "RAG Dashboard":
        st.subheader("📚 Dashboard")
    
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
            bucket = st.selectbox("Select Bucket Name(S3)", ["etrm-etai-poc", "etrm-etai-poc-ng"])
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

    elif st.session_state.sub_section == "Dashboard":
        st.subheader("📚 Dashboard")
        if st.button("🔄 Update from S3"):
            with sqlite3.connect(DB_NAME) as conn:
                process_and_save(conn, VALID_BUCKET, aws_access_key, aws_secret_key)
            st.success("Data updated!")
        
        data = load_data_for_dashboard()
        segment_json = data.get("by_segment", {})
        bookattr_json = data.get("by_book_attr8", {})
        tgroup_json = data.get("by_tgroup1", {})
        heatmapdata = data.get("heatmap_table", {})
        summary = data.get("daily_summary_totals", {})

        segment_options = list(segment_json.keys())
        bookattr_options = list(bookattr_json.keys())
        tgroup1_options = list(tgroup_json.keys())

        render_summary_card(summary,client)

        show_nop_cards(data)

        selected_segment = st.selectbox("Select Segment", segment_options)

        col1, col2= st.columns(2)
        with col1:
            plot_delta_volume_by_horizon( segment_json, selected_segment)
        with col2:
            plot_delta_market_value_by_horizon(segment_json, selected_segment)
        
        selected_tgroup1 = st.selectbox("Select Primary Strategy", tgroup1_options)
        col5, col6= st.columns(2)
        with col5:
            plot_delta_volume_by_horizon_by_tgroup1( tgroup_json, selected_tgroup1)
        with col6:
            plot_delta_market_value_by_horizon_by_tgroup1(tgroup_json, selected_tgroup1)

        selected_book_attr = st.selectbox("Select Business Classification", bookattr_options)
        col3, col4= st.columns(2)
        with col3:
            plot_delta_volume_by_horizon_by_bookattr8(bookattr_json, selected_book_attr)
        with col4:
            plot_delta_market_value_by_horizon_by_bookattr8(bookattr_json, selected_book_attr)

        col5, col6= st.columns(2)
        with col5:
            row_option = st.radio("Choose row dimension", ["BUCKET", "HORIZON"], horizontal=True)
            plot_heatmap(heatmapdata, row_option)
        with col6:
            show_top5_movers(data)
        
    elif st.session_state.sub_section in ["Energy Trading Analysis", "Energy Tradeing Analysis"]:
        if "query_answer" not in st.session_state:
            st.session_state["query_answer"] = None
        if "feedback_submitted" not in st.session_state:
            st.session_state["feedback_submitted"] = False
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("📊 Energy Trading Analysis")
        st.write("💬 Ask Your Financial or Commodity Question")

        col1, col2 = st.columns([6, 1])
        with col1:
            category_options = ["CO2", "Natural Gas", "Power"]
            selected_category = st.selectbox("Select Data Category", category_options)

        with col2:
            st.write(" ")
            if st.button("🧹 Clear Chat"):
                st.session_state.messages.clear()

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

            if msg["role"] == "assistant":
                feedback_key = f"feedback_{i}"
                col1, col2 = st.columns([1, 6])

                with col1:
                    if not msg.get("feedback"):
                        if st.button("👍", key=f"{feedback_key}_up"):
                            st.session_state.messages[i]["feedback"] = "positive"
                            st.session_state[f"show_comment_{i}"] = False
                            st.success("Thanks for your feedback!")

                        if st.button("👎", key=f"{feedback_key}_down"):
                            st.session_state.messages[i]["feedback"] = "negative"
                            st.session_state[f"show_comment_{i}"] = True
                    else:
                        st.markdown("✅ Feedback recorded")

                if (
                    st.session_state.get(f"show_comment_{i}", False)
                    and "feedback_comment" not in msg
                ):
                    comment_key = f"comment_input_{i}"
                    comment = st.text_area("😞 What went wrong?", key=comment_key)
                    if st.button("Submit Feedback", key=f"submit_feedback_{i}"):
                        st.session_state.messages[i]["feedback_comment"] = comment
                        update_feedback_log(0, comment, st.session_state.messages[i].get("log_id", None))
                        st.success("Thanks for your feedback!")
                        st.session_state[f"show_comment_{i}"] = False

        if prompt := st.chat_input("Ask a question... e.g(What is the total Price Value on 13 Nov 2024?)"):

            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.spinner("Thinking..."):
                response = query_sqlite_json_with_openai(prompt, selected_category)
                log_id = add_feedback_log(prompt, response, selected_category)

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "log_id": log_id
            })
            st.rerun()

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
            display_df = feedback_df[["id", "query", "answer", "feedback_comment", "user_feedback", "Timestamp"]]
            st.dataframe(display_df, use_container_width=True)
            st.subheader("📊 Feedback Summary")
            positive_feedback_count = (feedback_df["user_feedback"] == 1).sum()
            negative_feedback_count = (feedback_df["user_feedback"] == 0).sum()
            st.write(f"✅ **Positive Feedback:** {positive_feedback_count}")
            st.write(f"❌ **Negative Feedback:** {negative_feedback_count}")
            st.subheader("📊 Feedback Trends Over Time")
            feedback_over_time = (
                feedback_df.groupby(feedback_df["Timestamp"].dt.date)["feedback_comment"]
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