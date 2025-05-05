import pandas as pd
import boto3
import io
import sqlite3
import json
import numpy as np
from config import aws_access_key, aws_secret_key, client, DB_NAME, VALID_BUCKET, REJECTED_BUCKET

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
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                df.columns = df.columns.str.upper().str.replace(' ', '_')
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

def process_file_to_filetracking(file,folder_prefix):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    file_name = file.name
    try:
        cursor.execute("SELECT 1 FROM file_tracking WHERE file_name = ? AND is_processed = 1", (file_name,))
        if cursor.fetchone():
            print(f"⏭️ Skipping already processed file: {file_name}")
            return
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.upper().str.replace(' ', '_')
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

def process_metadata_alias(metadata_df: pd.DataFrame):
    alias_mapping = {}
    for _, row in metadata_df.iterrows():
        col_name = str(row.get("Column", "")).strip()
        description = str(row.get("Description", "")).strip().lower()
        if col_name and description:
            alias_mapping[description] = col_name
    return alias_mapping

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
