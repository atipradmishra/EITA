import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import sqlite3
from config import DB_NAME,client
import plotly.express as px
from DbUtils.DbOperations import load_data_for_dashboard
from GraphFunctions.dashboardcards import show_nop_cards,render_summary_card

# Chart 1: Delta Volume by Horizon for Segment
def plot_delta_volume_by_horizon(segment_json, selected_segment):
    segment_data = segment_json.get(selected_segment, {})
    if not segment_data:
        st.warning("Selected segment data is not available.")
        return

    horizon_labels = []
    delta_volume_data = []

    for horizon, records in segment_data.items():
        horizon_labels.append(horizon)
        delta_volume_data.append(sum(record['delta_volume_bl'] for record in records))

    sorted_data = sorted(zip(horizon_labels, delta_volume_data), key=lambda x: x[0])
    horizon_labels, delta_volume_data = zip(*sorted_data)

    fig = go.Figure(data=go.Scatter(x=horizon_labels, y=delta_volume_data, mode='lines+markers'))
    fig.update_layout(
        title=f"📈 Delta Volume Baseload by Horizon for Segment: {selected_segment}",
        xaxis_title="Horizon",
        yaxis_title="Delta Volume Baseload",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 2: Delta Market Value by Horizon for Segment
def plot_delta_market_value_by_horizon(segment_json, selected_segment):
    segment_data = segment_json.get(selected_segment, {})
    if not segment_data:
        st.warning("Selected segment data is not available.")
        return

    horizon_labels = []
    delta_market_val_data = []

    for horizon, records in segment_data.items():
        horizon_labels.append(horizon)
        delta_market_val_data.append(sum(record['delta_market_bl'] for record in records))

    sorted_data = sorted(zip(horizon_labels, delta_market_val_data), key=lambda x: x[0])
    horizon_labels, delta_market_val_data = zip(*sorted_data)

    fig = go.Figure(data=go.Scatter(x=horizon_labels, y=delta_market_val_data, mode='lines+markers'))
    fig.update_layout(
        title=f"💰 Delta Market Value Baseload by Horizon for Segment: {selected_segment}",
        xaxis_title="Horizon",
        yaxis_title="Delta Market Value Baseload",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 3: Delta Volume by Horizon by BOOK_ATTR8
def plot_delta_volume_by_horizon_by_bookattr8(bookattr_json, selected_book_attr):
    fig = go.Figure()
    horizon_data = bookattr_json.get(selected_book_attr, {})
    if not horizon_data:
        st.warning("Selected Business Classification data is not available.")
        return
    horizon_labels = []
    delta_values = []

    for horizon, records in horizon_data.items():
        horizon_labels.append(horizon)
        delta = sum(record['delta_volume_bl'] for record in records)
        delta_values.append(delta)
    sorted_data = sorted(zip(horizon_labels, delta_values), key=lambda x: x[0])
    horizon_labels, delta_values = zip(*sorted_data)
    fig.add_trace(go.Scatter(
        x=horizon_labels,
        y=delta_values,
        mode='lines+markers'
    ))

    fig.update_layout(
        title=f"📘 Delta Volume Baseload by Horizon for Business Classification : {selected_book_attr}",
        xaxis_title="Horizon",
        yaxis_title="Delta Volume Baseload",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 4: Delta Market Value by Horizon by BOOK_ATTR8
def plot_delta_market_value_by_horizon_by_bookattr8(bookattr_json, selected_book_attr):
    fig = go.Figure()
    horizon_data = bookattr_json.get(selected_book_attr, {})
    if not horizon_data:
        st.warning("Selected Business Classification data is not available.")
        return
    horizon_labels = []
    delta_values = []

    for horizon, records in horizon_data.items():
        horizon_labels.append(horizon)
        delta = sum(record['delta_market_bl'] for record in records)
        delta_values.append(delta)
    sorted_data = sorted(zip(horizon_labels, delta_values), key=lambda x: x[0])
    horizon_labels, delta_values = zip(*sorted_data)
    fig.add_trace(go.Scatter(
        x=horizon_labels,
        y=delta_values,
        mode='lines+markers'
    ))
    fig.update_layout(
        title=f"📕 Delta Market Value Baseload by Horizon for Business Classification : {selected_book_attr}",
        xaxis_title="Horizon",
        yaxis_title="Delta Market Value Baseload",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 5: Delta Volume by Horizon by TGROUP1
def plot_delta_volume_by_horizon_by_tgroup1(tgroup_json, selected_tgroup1):
    tgroup_data = tgroup_json.get(selected_tgroup1, {})
    if not tgroup_data:
        st.warning("Selected TGROUP1 data is not available.")
        return

    horizon_labels = []
    delta_values = []

    for horizon, records in tgroup_data.items():
        horizon_labels.append(horizon)
        delta_values.append(sum(record['delta_volume_bl'] for record in records))

    sorted_data = sorted(zip(horizon_labels, delta_values), key=lambda x: x[0])
    horizon_labels, delta_values = zip(*sorted_data)

    fig = go.Figure(data=go.Scatter(x=horizon_labels, y=delta_values, mode='lines+markers'))
    fig.update_layout(
        title=f"📊 Delta Volume Baseload by Horizon for Primary Strategy: {selected_tgroup1}",
        xaxis_title="Horizon",
        yaxis_title="Delta Volume Baseload",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 6: Delta Market Value by Horizon by TGROUP1
def plot_delta_market_value_by_horizon_by_tgroup1(tgroup_json, selected_tgroup1):
    tgroup_data = tgroup_json.get(selected_tgroup1, {})
    if not tgroup_data:
        st.warning("Selected TGROUP1 data is not available.")
        return

    horizon_labels = []
    delta_values = []

    for horizon, records in tgroup_data.items():
        horizon_labels.append(horizon)
        delta_values.append(sum(record['delta_market_bl'] for record in records))

    sorted_data = sorted(zip(horizon_labels, delta_values), key=lambda x: x[0])
    horizon_labels, delta_values = zip(*sorted_data)

    fig = go.Figure(data=go.Scatter(x=horizon_labels, y=delta_values, mode='lines+markers'))
    fig.update_layout(
        title=f"📉 Delta Market Value Baseload by Horizon for Primary Strategy: {selected_tgroup1}",
        xaxis_title="Horizon",
        yaxis_title="Delta Market Value Baseload",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# top5 movers bar graph
def show_top5_movers(reports_data):
    if not reports_data:
        st.warning("No data provided.")
        return

    col1, col2 = st.columns(2)
    with col1:
        group_by_option = st.selectbox("Group By", ["Segment", "Business Classification", "Primary Strategy"], key="group_by_option3")
        section_map = {
            "Segment": "by_segment_horizon",
            "Business Classification": "by_book_attr8_horizon",
            "Primary Strategy": "by_tgroup1_horizon"
        }
        section = section_map[group_by_option]

    all_horizons = set()
    for report in reports_data:
        for group, horizons in report.get(section, {}).items():
            all_horizons.update(horizons.keys())

    if not all_horizons:
        st.warning("No horizons found in data.")
        return

    with col2:
        selected_horizon = st.selectbox("Select Horizon for Top Movers", sorted(all_horizons), key=f"{section}_horizon")

    metric_option = st.radio("Select Metric", ["Volume Baseload", "Market Value Baseload"], horizontal=True, key=f"{section}_metric")
    metric_key = "volume_bl" if metric_option == "Volume Baseload" else "market_val_bl"

    group_metric_by_date = {}

    for report in reports_data:
        report_date = report.get("report_date")
        section_data = report.get(section, {})
        for group, horizons in section_data.items():
            horizon_data = horizons.get(selected_horizon, {})
            value = horizon_data.get(metric_key, 0)
            group_metric_by_date.setdefault(group, {})[report_date] = value

    movers_data = {}
    for group, metrics in group_metric_by_date.items():
        if len(metrics) < 2:
            continue
        sorted_dates = sorted(metrics.keys())
        delta = metrics[sorted_dates[-1]] - metrics[sorted_dates[-2]]
        movers_data[group] = delta

    if not movers_data:
        st.info("Not enough data to compute deltas.")
        return

    top5 = sorted(movers_data.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    df_plot = pd.DataFrame(top5, columns=[group_by_option, 'Delta'])

    fig = px.bar(
        df_plot,
        x=group_by_option,
        y='Delta',
        color='Delta',
        color_continuous_scale=['red', 'grey', 'green'],
        title=f"Top 5 Movers by {group_by_option} - {metric_option} ({selected_horizon})"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_delta_volume_from_reports(reports_data):
    if not reports_data or len(reports_data) < 2:
        st.warning("Not enough report data to display NOP summary.")
        return

    col1, col2 = st.columns(2)

    with col1:
        group_by_option = st.selectbox("Group By", ["Segment", "Business Classification", "Primary Strategy"])
        section_map = {
            "Segment": "by_segment_horizon",
            "Business Classification": "by_book_attr8_horizon",
            "Primary Strategy": "by_tgroup1_horizon"
        }
        section = section_map.get(group_by_option)
    
    latest_report = reports_data[-1]
    categories = list(latest_report.get(section, {}).keys())
    if not categories:
        st.warning("No categories found in selected section.")
        return

    with col2:
        category = st.selectbox(f"Select {group_by_option}", categories)

    horizon_volume_by_date = {}

    for report in reports_data:
        report_date = report.get("report_date")
        section_data = report.get(section, {}).get(category, {})

        for horizon, values in section_data.items():
            vol = values.get("volume_bl", 0)
            if report_date:
                horizon_volume_by_date.setdefault(horizon, {})[report_date] = vol

    horizons = []
    deltas = []

    for horizon, volume_by_date in horizon_volume_by_date.items():
        if len(volume_by_date) < 2:
            continue
        sorted_dates = sorted(volume_by_date.keys())
        last = volume_by_date[sorted_dates[-1]]
        second_last = volume_by_date[sorted_dates[-2]]
        delta = last - second_last
        horizons.append(horizon)
        deltas.append(delta)

    if not horizons:
        st.warning("Not enough data to compute deltas.")
        return

    fig = go.Figure(data=go.Scatter(x=horizons, y=deltas, name='Delta Volume',mode='lines+markers'))
    fig.update_layout(
        title=f"📉 Delta Volume by Horizon for '{category}' in '{group_by_option}'",
        xaxis_title="Horizon",
        yaxis_title="Delta Volume (Last - Second Last)",
        template="plotly_white"
    )
    fig.update_traces(hovertemplate="Delta: %{y}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

def plot_delta_market_from_reports(reports_data):
    if not reports_data or len(reports_data) < 2:
        st.warning("Not enough report data to display NOP summary.")
        return
    col1, col2 = st.columns(2)

    with col1:
        group_by_option = st.selectbox("Group By", ["Segment", "Business Classification", "Primary Strategy"], key="group_by_option")
        section_map = {
            "Segment": "by_segment_horizon",
            "Business Classification": "by_book_attr8_horizon",
            "Primary Strategy": "by_tgroup1_horizon",
        }
        section = section_map.get(group_by_option)
    
    latest_report = reports_data[-1]
    categories = list(latest_report.get(section, {}).keys())
    if not categories:
        st.warning("No categories found in selected section.")
        return

    with col2:
        category = st.selectbox(f"Select {group_by_option}", categories, key="category")

    horizon_volume_by_date = {}

    for report in reports_data:
        report_date = report.get("report_date")
        section_data = report.get(section, {}).get(category, {})

        for horizon, values in section_data.items():
            vol = values.get("market_val_bl", 0)
            if report_date:
                horizon_volume_by_date.setdefault(horizon, {})[report_date] = vol

    horizons = []
    deltas = []

    for horizon, volume_by_date in horizon_volume_by_date.items():
        if len(volume_by_date) < 2:
            continue
        sorted_dates = sorted(volume_by_date.keys())
        last = volume_by_date[sorted_dates[-1]]
        second_last = volume_by_date[sorted_dates[-2]]
        delta = last - second_last
        horizons.append(horizon)
        deltas.append(delta)

    if not horizons:
        st.warning("Not enough data to compute deltas.")
        return

    fig = go.Figure(data=go.Scatter(x=horizons, y=deltas, name='Delta Markrt Value',mode='lines+markers'))
    fig.update_layout(
        title=f"📉 Delta Market Value by Horizon for '{category}' in '{group_by_option}'",
        xaxis_title="Horizon",
        yaxis_title="Delta Market Value (Last - Second Last)",
        template="plotly_white"
    )
    fig.update_traces(hovertemplate="Delta: %{y}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
