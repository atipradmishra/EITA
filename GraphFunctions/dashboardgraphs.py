import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import plotly.express as px

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
def show_top5_movers(data):

    col1, col2= st.columns(2)
    with col1:
        group_by_option = st.selectbox("Group By", ["Segment", "Business Classification", 'Primary Strategy'])
        if group_by_option == 'Segment':
            dimension = 'by_segment'
        elif group_by_option == 'Business Classification':
            dimension = 'by_book_attr8'
        elif group_by_option == 'Primary Strategy':
            dimension = 'by_tgroup1'
        else:
            st.warning(f"Invalid dimension '{dimension}' provided.")
            return
    with col2:
        horizons = list(next(iter(data[dimension].values())).keys())
        selected_horizon = st.selectbox("Select Horizon for Top Movers", horizons, key=f"{dimension}_horizon")

    metric_option = st.radio("Select Metric", ["Volume Baseload", "Market Value Baseload"],horizontal=True, key=f"{dimension}_metric")
    metric_key = "delta_volume_bl" if metric_option == "Volume Baseload" else "delta_market_bl"

    movers_data = {}
    for group_name, horizon_dict in data[dimension].items():
        horizon_data = horizon_dict.get(selected_horizon, [])
        if horizon_data:
            latest_delta = horizon_data[-1].get(metric_key, 0)
            movers_data[group_name] = latest_delta

    if not movers_data:
        st.info("No delta data available for the selected horizon.")
        return

    # Sort and take Top 5 by absolute value
    top5 = sorted(movers_data.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    df_plot = pd.DataFrame(top5, columns=[dimension.replace("by_", "").capitalize(), 'Delta'])

    # Generate bar chart
    fig = px.bar(
        df_plot,
        x=dimension.replace("by_", "").capitalize(),
        y='Delta',
        color='Delta',
        color_continuous_scale=['red', 'grey', 'green'],
        title=f"Top 5 Movers by {group_by_option.replace('by_', '').capitalize()} - {metric_option} ({selected_horizon})"
    )
    st.plotly_chart(fig, use_container_width=True)
