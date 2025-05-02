import streamlit as st
import pandas as pd
from datetime import datetime

def show_delta_heatmap(data_list):
    row_option = st.radio("Choose row grouping:", ["Bucket","Horizon", ], horizontal=True)
    view_map = {"Horizon": "by_segment_horizon", "Bucket": "by_segment_bucket"}
    view_key = view_map[row_option]

    # Sort and ensure at least two report dates
    sorted_data = sorted(data_list, key=lambda d: datetime.strptime(d["report_date"], "%Y-%m-%d"), reverse=True)
    if len(sorted_data) < 2:
        st.warning("Need at least two days of data.")
        return

    latest, previous = sorted_data[0], sorted_data[1]
    latest_date, previous_date = latest["report_date"], previous["report_date"]
    latest_view = latest.get(view_key, {})
    previous_view = previous.get(view_key, {})

    all_segments = set(latest_view.keys()).union(previous_view.keys())
    all_rows = set()

    # Collect all row labels (horizons or buckets)
    for seg in all_segments:
        all_rows.update(latest_view.get(seg, {}).keys())
        all_rows.update(previous_view.get(seg, {}).keys())

    data_matrix = {}

    for row_label in all_rows:
        row_data = {}
        for seg in all_segments:
            v1 = latest_view.get(seg, {}).get(row_label, {}).get("market_val_bl", 0)
            v2 = previous_view.get(seg, {}).get(row_label, {}).get("market_val_bl", 0)
            delta = v1 - v2
            row_data[seg] = f"{'+' if delta > 0 else ''}{round(delta, 2)}"
        data_matrix[row_label] = row_data

    matrix = pd.DataFrame.from_dict(data_matrix, orient='index').fillna("0.0")

    def style_func(val):
        try:
            if isinstance(val, str):
                if "+" in val:
                    color = '#dcfce7'  # green
                elif "-" in val:
                    color = '#fee2e2'  # red
                else:
                    color = '#f1f5f9'  # neutral
                return f'background-color: {color}; color: #272420; font-weight: bold'
        except:
            pass
        return ''

    st.subheader(f"Segment Delta Heatmap by {row_option}")
    st.dataframe(matrix.style.map(style_func), use_container_width=True)
