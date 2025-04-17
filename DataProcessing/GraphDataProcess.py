import pandas as pd
import sqlite3
import json
import io
import boto3
from collections import defaultdict

def clean_number(value):
    if pd.isna(value):
        return None
    return float(str(value).replace(',', ''))

def compute_all_daily_totals(df):
    final = {}

    for dimension, key_name in [
        ("SEGMENT", "by_segment_and_horizon"),
        ("TGROUP1", "by_tgroup1_and_horizon"),
        ("BOOK_ATTR8", "by_book_attr8_and_horizon"),
        ("BOOK", "by_book_and_horizon")
        ]:
        grouped = df.groupby(['REPORT_DATE', dimension, 'HORIZON'])

        for (report_date, dim_val, horizon), group in grouped:
            date_str = report_date.strftime('%Y-%m-%d')
            total_volume_bl = group['VOLUME_BL'].sum()
            total_market = group['MKT_VAL_BL'].sum()
            # volume_pk = group['VOLUME_PK'].sum()
            # volume_ofpk = group['VOLUME_OFPK'].sum()
            # mkt_val_pk = group['MKT_VAL_PK'].sum()
            # mkt_val_ofpk = group['MKT_VAL_OFPK'].sum()
            # trd_val_bl = group['TRD_VAL_BL'].sum()
            # trd_val_pk = group['TRD_VAL_PK'].sum()
            # trd_val_ofpk = group['TRD_VAL_OFPK'].sum()

            if date_str not in final:
                final[date_str] = {
                    "by_segment_and_horizon": {},
                    "by_tgroup1_and_horizon": {},
                    "by_book_attr8_and_horizon": {},
                    "by_book_and_horizon": {}
                }

            if dim_val not in final[date_str][key_name]:
                final[date_str][key_name][dim_val] = {}

            final[date_str][key_name][dim_val][horizon] = {
                "total_volume_bl": total_volume_bl,
                "total_market_value_bl": total_market
                # "total_volume_pk": volume_pk,
                # "total_volume_ofpk": volume_ofpk,
                # "total_market_value_pk": mkt_val_pk,
                # "total_market_value_ofpk": mkt_val_ofpk,
                # "total_trading_value_bl": trd_val_bl,
                # "total_trading_value_pk": trd_val_pk,
                # "total_trading_value_ofpk": trd_val_ofpk    
            }

    return final

def save_rawdata_to_db(df,conn, file_name):
    df.to_sql('raw_data', conn, if_exists='append', index=False)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO graph_file_metadata (file_name, is_processed) VALUES (?, ?)",
        (file_name, 1)
    )

def process_csv_data_to_json(df):
    # df = pd.read_csv(csv_path, parse_dates=["REPORT_DATE", "Start Date", "End Date"],dayfirst=True)
    df.replace(",", "", regex=True, inplace=True)
    df['MKT_VAL_BL'] = df['MKT_VAL_BL'].astype(str).str.replace(',', '').astype(float)
    float_cols = ['VOLUME_BL', 'MKT_VAL_BL']
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')

    df.sort_values(by=['REPORT_DATE'], inplace=True)

    def compute_delta(group_dim):
        result = {}
        for key, group in df.groupby([group_dim, 'HORIZON']):
            group = group.sort_values(by="REPORT_DATE")
            group['delta_volume_bl'] = group['VOLUME_BL'].diff().fillna(0)
            group['delta_market_bl'] = group['MKT_VAL_BL'].diff().fillna(0)
            
            horizon = key[1]
            dim_value = key[0]
            data = group[['REPORT_DATE', 'delta_volume_bl', 'delta_market_bl']]
            formatted = [
                {
                    "date": row['REPORT_DATE'].strftime("%Y-%m-%d"),
                    "delta_volume_bl": row['delta_volume_bl'],
                    "delta_market_bl": row['delta_market_bl']
                }
                for _, row in data.iterrows()
            ]
            result.setdefault(dim_value, {}).setdefault(horizon, []).extend(formatted)
        return result

    segment_json = compute_delta('SEGMENT')
    tgroup1_json = compute_delta('TGROUP1')
    book_attr8_json = compute_delta('BOOK_ATTR8')

    heatmap_data = []

    for (segment, horizon, bucket), group in df.groupby(['SEGMENT', 'HORIZON', 'BUCKET']):
        group = group.sort_values(by='REPORT_DATE')
        group['delta_market_bl'] = group['MKT_VAL_BL'].diff().fillna(0)
        latest_row = group.iloc[-1]

        delta = latest_row['delta_market_bl']
        if pd.isna(delta):
            formatted_val = "0"
            color = "grey"
        elif delta > 0:
            formatted_val = f"+{delta:,.2f}"
            color = "green"
        elif delta < 0:
            formatted_val = f"-{abs(delta):,.2f}"
            color = "red"
        else:
            formatted_val = "0"
            color = "grey"

        heatmap_data.append({
            "bucket": bucket,
            "segment": segment,
            "horizon": horizon,
            "delta": formatted_val,
            "color": color
        })

    heatmap_structure = {}
    for row in heatmap_data:
        horizon = row['horizon']
        bucket = row['bucket']
        segment = row['segment']
        entry = {"delta": row['delta'], "color": row['color']}

        heatmap_structure.setdefault(horizon, {}).setdefault(bucket, {})[segment] = entry


    daily_summary = (
        df.groupby('REPORT_DATE')[['VOLUME_BL', 'MKT_VAL_BL']]
        .sum()
        .sort_index()
        .reset_index()
    )

    # Calculate deltas
    daily_summary['delta_volume_bl'] = daily_summary['VOLUME_BL'].diff().fillna(0)
    daily_summary['delta_market_bl'] = daily_summary['MKT_VAL_BL'].diff().fillna(0)

    # Format to JSON
    daily_nop_summary = [
        {
            "date": row['REPORT_DATE'].strftime("%Y-%m-%d"),
            "total_volume_bl": row['VOLUME_BL'],
            "delta_volume_bl": row['delta_volume_bl'],
            "total_market_bl": row['MKT_VAL_BL'],
            "delta_market_bl": row['delta_market_bl']
        }
        for _, row in daily_summary.iterrows()
    ]

    daily_summary_totals = compute_all_daily_totals(df)

    output = {
        "by_segment": segment_json,
        "by_tgroup1": tgroup1_json,
        "by_book_attr8": book_attr8_json,
        "heatmap_table": heatmap_structure,
        "daily_summary_totals": daily_summary_totals,
        'daily_nop_summary': daily_nop_summary
    }

    return output

def save_jsondata_to_db(json_data,conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM graph_data WHERE file_name = 'latest_combined'")
    cursor.execute(
        "INSERT INTO graph_data (file_name, json_contents) VALUES (?, ?)",
        ('latest_combined', json.dumps(json_data))
    )
    conn.commit()

def process_and_save_byfile(conn, file):
    try:
        file_name = file.name

        with conn:
            cursor = conn.cursor()

            # Check if file already processed
            cursor.execute(
                "SELECT 1 FROM graph_file_metadata WHERE file_name = ? AND is_processed = 1",
                (file_name,)
            )
            if cursor.fetchone():
                print(f"⏭️ Skipping already processed file: {file_name}")
                return

            try:
                # Read and preprocess CSV
                df = pd.read_csv(file)
                df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], format='mixed', dayfirst=False)
                print(f"🔄 Processing: {file_name}")

                df.replace(",", "", regex=True, inplace=True)
                df['MKT_VAL_BL'] = df['MKT_VAL_BL'].astype(str).str.replace(',', '').astype(float)
                df[['VOLUME_BL', 'MKT_VAL_BL']] = df[['VOLUME_BL', 'MKT_VAL_BL']].apply(pd.to_numeric, errors='coerce')

                df['source_file'] = file_name

                desired_columns = [
                    'REPORT_DATE', 'SEGMENT', 'TGROUP1', 'BUCKET', 'HORIZON',
                    'VOLUME_BL', 'MKT_VAL_BL', 'source_file', 'BOOK_ATTR8',
                    'USR_VAL4', 'BOOK', 'VOLUME_PK', 'VOLUME_OFPK',
                    'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL',
                    'TRD_VAL_PK', 'TRD_VAL_OFPK'
                ]
                save_rawdata_to_db(df[desired_columns], conn, file_name)

            except Exception as e:
                print(f"❌ Error processing file {file_name}: {e}")

        # After successful processing, reload all data and generate graph JSON
        raw_df = pd.read_sql_query("SELECT * FROM raw_data", conn)
        raw_df['REPORT_DATE'] = pd.to_datetime(raw_df['REPORT_DATE'], format='mixed', dayfirst=False)

        json_data = process_csv_data_to_json(raw_df)
        save_jsondata_to_db(json_data, conn)
        print("✅ All data processed and saved to database.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")

def process_and_save_from_s3(conn,bucket_name, aws_access_key, aws_secret_key, folder_prefix='PW/'):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name='us-east-1'
    )

    try:
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix).get("Contents", [])
        with conn:
            cursor = conn.cursor()
            for obj in objects:
                file_key = obj["Key"]
                if not file_key.lower().endswith(".csv"):
                    continue

                file_name = file_key.split("/")[-1]
                print(f"🔍 Checking: {file_name}")

                cursor.execute("SELECT 1 FROM graph_file_metadata WHERE file_name = ? AND is_processed = 1", (file_name,))
                if cursor.fetchone():
                    print(f"⏭️ Skipping already processed file: {file_name}")
                    continue

                try:
                    obj_data = s3.get_object(Bucket=bucket_name, Key=file_key)
                    print(f"🔄 Processing: {file_name}")
                    file_stream = io.BytesIO(obj_data["Body"].read())
                    df = pd.read_csv(file_stream, parse_dates=["REPORT_DATE"], dayfirst=True)
                    df.replace(",", "", regex=True, inplace=True)
                    df['MKT_VAL_BL'] = df['MKT_VAL_BL'].astype(str).str.replace(',', '').astype(float)
                    df[['VOLUME_BL', 'MKT_VAL_BL']] = df[['VOLUME_BL', 'MKT_VAL_BL']].apply(pd.to_numeric, errors='coerce')
                    df['source_file'] = file_name
                    desired_columns = ['REPORT_DATE', 'SEGMENT', 'TGROUP1', 'BUCKET', 'HORIZON', 'VOLUME_BL', 'MKT_VAL_BL', 'source_file', 'BOOK_ATTR8', 'USR_VAL4', 'BOOK','VOLUME_PK', 'VOLUME_OFPK', 'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
                    save_rawdata_to_db(df[desired_columns], conn, file_name)
                except Exception as e:
                    print(f"❌ Error processing file {file_name}: {e}")

        # After all files, reload full raw data and regenerate graph JSON
        raw_df = pd.read_sql_query("SELECT * FROM raw_data", conn)
        raw_df['REPORT_DATE'] = pd.to_datetime(raw_df['REPORT_DATE'], dayfirst=True)
        json_data = process_csv_data_to_json(raw_df)
        save_jsondata_to_db(json_data, conn)
        print("✅ All data processed and saved to database.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")