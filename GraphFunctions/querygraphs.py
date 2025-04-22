from config import DB_NAME
import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import altair as alt
import re

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
