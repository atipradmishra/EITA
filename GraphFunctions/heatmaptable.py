# heatmap_viewer.py
import pandas as pd
import streamlit as st

def plot_heatmap(heatmap_data, row_option):


    selected_dim = "bucket" if row_option == "BUCKET" else "horizon"

    # Get all row values based on dimension
    all_row_values = sorted({
        row_key
        for horizon_data in heatmap_data.values()
        for row_key in horizon_data.keys()
    }) if selected_dim == "bucket" else sorted(heatmap_data.keys())

    # Get all segments
    all_segments = sorted({
        seg
        for horizon_data in heatmap_data.values()
        for row_data in horizon_data.values()
        for seg in row_data.keys()
    })

    # Build a matrix DataFrame
    matrix = pd.DataFrame(index=all_row_values, columns=all_segments)

    for horizon, row_data in heatmap_data.items():
        for row_key, seg_data in row_data.items():
            row_label = row_key if selected_dim == "bucket" else horizon
            for segment, info in seg_data.items():
                val = info.get('delta', 0)
                matrix.loc[row_label, segment] = val if pd.notna(val) else 0

    matrix.fillna(0, inplace=True)

    # Styling function
    def style_func(val):
        val = str(val)
        try:
            if "+" in val:
                color = '#dcfce7'  # greenish
            elif "-" in val:
                color = '#fee2e2'  # reddish
            else:
                color = '#f1f5f9'  # neutral
            return f'background-color: {color}; color: #272420; font-weight: bold'
        except:
            return ''

    st.subheader(f"Segment Delta Heatmap by {row_option}")
    st.dataframe(matrix.style.map(style_func), use_container_width=True)
