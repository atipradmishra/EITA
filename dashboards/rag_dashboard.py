# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

def rag_agents_dashboard():

    st.markdown('<div style="background-color:#0f2b46; padding:15px; border-radius:10px;">'
                   '<h1 style="color:white; text-align:center;">RAG Agents Dashboard</h1>'
                   '<p style="color:#adb5bd; text-align:center;">Monitor and evaluate retrieval-augmented generation agents</p>'
                   '</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Dummy data (replace with real DB queries)
    agent_data = pd.DataFrame({
        "agent_name": ["PowerBot", "CO2Bot", "NGInsight"],
        "status": ["Active", "Active", "Inactive"],
        "queries_today": [125, 90, 0],
        "avg_latency": [1.2, 1.5, None],
        "last_used": ["2025-04-23", "2025-04-23", "2025-04-20"]
    })

    query_logs = pd.DataFrame({
        "agent": ["PowerBot", "PowerBot", "CO2Bot"],
        "query": ["What was peak power yesterday?", "Show trend of usage", "Top emitters in March"],
        "response_snippet": ["Peak power was 452MW", "Trend plotted from Jan to Mar", "Top emitters were X, Y"],
        "timestamp": ["2025-04-23 10:02", "2025-04-23 11:13", "2025-04-23 09:45"]
    })

    # Summary KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Agents", len(agent_data))
    col2.metric("Active Agents", (agent_data["status"] == "Active").sum())
    col3.metric("Total Queries Today", agent_data["queries_today"].sum())
    col4.metric("Avg Latency (s)", round(agent_data["avg_latency"].dropna().mean(), 2))
    style_metric_cards()

    st.markdown("---")

    # Pie Chart
    fig = px.pie(agent_data, names="agent_name", values="queries_today", title="Query Volume per Agent")
    st.plotly_chart(fig, use_container_width=True)

    # Timeline Line Chart
    timeline_data = pd.DataFrame({
        "date": pd.date_range("2025-04-18", "2025-04-23"),
        "PowerBot": [80, 100, 90, 120, 110, 125],
        "CO2Bot": [40, 55, 60, 85, 80, 90],
        "NGInsight": [0, 0, 0, 10, 5, 0]
    }).melt(id_vars=["date"], var_name="Agent", value_name="Queries")

    timeline_chart = px.line(timeline_data, x="date", y="Queries", color="Agent", markers=True, title="Daily Queries per Agent")
    st.plotly_chart(timeline_chart, use_container_width=True)

    # Agent Table
    st.subheader("Agent Overview")
    st.dataframe(agent_data.style.highlight_null("red"), use_container_width=True)

    # Logs per agent
    st.subheader("Recent Query Logs")
    for agent in agent_data["agent_name"]:
        with st.expander(f"{agent} Logs"):
            logs = query_logs[query_logs["agent"] == agent]
            st.dataframe(logs, use_container_width=True)
