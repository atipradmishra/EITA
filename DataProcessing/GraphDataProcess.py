import pandas as pd
import sqlite3
import json
import io
import boto3
from collections import defaultdict
from DbUtils.DbOperations import save_grouped_data_to_db, add_data_file_metadata
from config import aws_access_key, aws_secret_key, client, DB_NAME


def process_and_save_byfile(file):
    conn = sqlite3.connect(DB_NAME)
    try:
        file_name = file.name

        category = ""
        if "NOP_CO2" in file_name.upper():
            category = "CO2"
        elif "NOP_NG" in file_name.upper():
            category = "NG"
        elif "NOP_PW" in file_name.upper():
            category = "PW"
        else:
            print(f"⚠️ Unknown file: {file_name}")
            return

        with conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM data_files_metadata WHERE file_name = ? AND is_processed = 1",
                (file_name,)
            )
            if cursor.fetchone():
                print(f"⏭️ Skipping already processed file: {file_name}")
                return

            try:
                df = pd.read_csv(file)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                df.columns = df.columns.str.upper().str.replace(' ', '_')
                df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce', dayfirst=True).dt.date
                if category == "CO2":
                    df['MKTVAL'] = df['MKTVAL'].astype(str).str.replace(',', '').astype(float)
                    df['VOLUME'] = df['VOLUME'].astype(str).str.replace(',', '').astype(float)
                    df[['VOLUME', 'MKTVAL']] = df[['VOLUME', 'MKTVAL']].apply(pd.to_numeric, errors='coerce')

                    print(f"🔄 Saving {file_name} data to Database")
                    df.to_sql('CO2_raw_data', conn, if_exists='append', index=False)
                    print(f"✅ Saved {file_name} data to CO2_raw_data table.")

                elif category == "NG":
                    df['MKT_VAL'] = df['MKT_VAL'].astype(str).str.replace(',', '').astype(float)
                    df['VOLUME'] = df['VOLUME'].astype(str).str.replace(',', '').astype(float)
                    df[['VOLUME', 'MKT_VAL']] = df[['VOLUME', 'MKT_VAL']].apply(pd.to_numeric, errors='coerce')

                    print(f"🔄 Saving {file_name} data to Database")
                    df.to_sql('NG_raw_data', conn, if_exists='append', index=False)
                    print(f"✅ Saved {file_name} data to NG_raw_data table.")

                elif category == "PW":
                    df['MKT_VAL_BL'] = df['MKT_VAL_BL'].astype(str).str.replace(',', '').astype(float)
                    df['VOLUME_BL'] = df['VOLUME_BL'].astype(str).str.replace(',', '').astype(float)
                    df[['VOLUME_BL', 'MKT_VAL_BL']] = df[['VOLUME_BL', 'MKT_VAL_BL']].apply(pd.to_numeric, errors='coerce')

                    print(f"🔄 Saving {file_name} data to Database")
                    df.to_sql('PW_raw_data', conn, if_exists='append', index=False)
                    print(f"✅ Saved {file_name} data to PW_raw_data table.")

                    print(f"🔄 Processing: {file_name}")
                    processed_data = process_csv_data_to_json(df)
                    save_processed_json_to_db(processed_data, 'PW')
                    print("✅ All data processed and saved to database.")
                else:
                    print(f"⚠️ Unknown category: {category}")
                    return

                add_data_file_metadata(file_name, category)

            except Exception as e:
                print(f"❌ Error processing file {file_name}: {e}")
    except Exception as e:
        print(f"❌ Error during processing: {e}")

def save_processed_json_to_db(json_dict, category):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    for report_date, data in json_dict.items():
        cur.execute("""
            INSERT OR REPLACE INTO daily_graph_data (report_date, json_data, category)
            VALUES (?, ?,?)
        """, (report_date, json.dumps(data), category))

    conn.commit()
    conn.close()

#new process function
def process_csv_data_to_json(df):

    df['MKT_VAL_BL'] = df['MKT_VAL_BL'].astype(str).str.replace(',', '')
    float_cols = ['VOLUME_BL', 'MKT_VAL_BL', 'VOLUME_PK', 'VOLUME_OFPK', 'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')

    df.sort_values(by=['REPORT_DATE'], inplace=True)

    final_output = {}

    for report_date, day_df in df.groupby('REPORT_DATE'):
        day_output = {}

        # Daily NOP Summary
        total_volume = day_df['VOLUME_BL'].sum()
        total_market_val = day_df['MKT_VAL_BL'].sum()
        day_output['daily_nop_summary'] = {
            "total_volume_bl": total_volume,
            "total_market_bl": total_market_val
        }

        def group_and_format_by(field):
            result = {}
            for (dim_val, horizon), group in day_df.groupby([field, 'HORIZON']):
                volume = group['VOLUME_BL'].sum()
                market_val = group['MKT_VAL_BL'].sum()
                volume_pk = group['VOLUME_PK'].sum()
                volume_ofpk = group['VOLUME_OFPK'].sum()
                mkt_val_pk = group['MKT_VAL_PK'].sum()
                mkt_val_ofpk = group['MKT_VAL_OFPK'].sum()
                trd_val_bl = group['TRD_VAL_BL'].sum()
                trd_val_pk = group['TRD_VAL_PK'].sum()
                trd_val_ofpk = group['TRD_VAL_OFPK'].sum()
                result.setdefault(dim_val, {})[horizon] = {
                    "volume_bl": volume,
                    "market_val_bl": market_val,
                    "volume_pk": volume_pk,
                    "volume_ofpk": volume_ofpk,
                    "mkt_val_pk": mkt_val_pk,
                    "mkt_val_ofpk": mkt_val_ofpk,
                    "trd_val_bl": trd_val_bl,
                    "trd_val_pk": trd_val_pk,
                    "trd_val_ofpk": trd_val_ofpk
                }
            return result
        
        def group_and_format_by_heatmap(field):
            result = {}
            for (dim_val, horizon), group in day_df.groupby([field, 'BUCKET']):
                market_val = group['MKT_VAL_BL'].sum()
                result.setdefault(dim_val, {})[horizon] = {
                    "market_val_bl": market_val
                }
            return result

        day_output['by_segment_horizon'] = group_and_format_by('SEGMENT')
        day_output['by_tgroup1_horizon'] = group_and_format_by('TGROUP1')
        day_output['by_book_attr8_horizon'] = group_and_format_by('BOOK_ATTR8')
        day_output['by_segment_bucket'] = group_and_format_by_heatmap('SEGMENT')

        final_output[report_date.strftime("%Y-%m-%d")] = day_output

    return final_output

def process_and_save_pw_from_db():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            raw_df = pd.read_sql_query(f"SELECT * FROM pw_raw_data", conn)
            raw_df['REPORT_DATE'] = pd.to_datetime(raw_df['REPORT_DATE'], dayfirst=False)
            processed_data = process_csv_data_to_json(raw_df)
            save_processed_json_to_db(processed_data, 'PW')
        print("✅ All data processed and saved to database.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")