import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import sqlite3
from config import DB_NAME
from datetime import datetime

# DB connection
def get_connection():
    return sqlite3.connect(DB_NAME)

# Get agent + file metadata info
def get_all_agents():
    conn = get_connection()
    query = """
        SELECT 
            ra.agent_id,
            ra.name,
            ra.description,
            ra.model,
            ra.temperature,
            ra.prompt,
            ra.is_active
        FROM 
            rag_agents ra
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Get agent query logs
def get_query_logs():
    conn = get_connection()
    query = """
        SELECT 
            q.agent_id, 
            ra.name AS agent_name,
            q.query, 
            q.answer, 
            q.created_at,
            q.latency
        FROM 
            feedback_logs q
        JOIN 
            rag_agents ra ON q.agent_id = ra.agent_id
        ORDER BY 
            q.created_at DESC
    """
    df = pd.read_sql_query(query, conn)
    df["created_at"] = pd.to_datetime(df["created_at"])
    conn.close()
    return df

# Dashboard UI
def rag_agents_dashboard():
    st.markdown("""
        <div style="background-color:#0f2b46; padding:15px; border-radius:10px;">
        <h1 style="color:white; text-align:center;">RAG Agents Dashboard</h1>
        <p style="color:#adb5bd; text-align:center;">Monitor and evaluate retrieval-augmented generation agents</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load real data
    agent_df = get_all_agents()
    query_logs = get_query_logs()

    # Transform agent data
    agent_df.rename(columns={"name": "agent_name", "is_active": "status"}, inplace=True)
    agent_df["status"] = agent_df["status"].apply(lambda x: "Active" if x == 1 else "Inactive")
    agent_df["queries_today"] = 0
    agent_df["avg_latency"] = None
    agent_df["last_used"] = None

    # Today's date
    today = pd.to_datetime("today").normalize()

    # Compute metrics from query logs
    queries_today = query_logs[query_logs["created_at"].dt.normalize() == today].groupby("agent_name").size()
    avg_latency = query_logs.groupby("agent_name")["latency"].mean()
    last_used = query_logs.groupby("agent_name")["created_at"].max()

    for index, row in agent_df.iterrows():
        agent_name = row["agent_name"]
        agent_df.at[index, "queries_today"] = queries_today.get(agent_name, 0)
        agent_df.at[index, "avg_latency"] = round(avg_latency.get(agent_name, 0), 2)
        agent_df.at[index, "last_used"] = last_used.get(agent_name, None)

    # Summary KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Agents", len(agent_df))
    col2.metric("Active Agents", (agent_df["status"] == "Active").sum())
    col3.metric("Total Queries Today", agent_df["queries_today"].sum())
    col4.metric("Avg Latency (s)", round(agent_df["avg_latency"].dropna().mean(), 2))
    style_metric_cards()

    st.markdown("---")

    col1 , col2 = st.columns(2)
    with col1:
        fig = px.pie(agent_df, names="agent_name", values="queries_today", title="Query Volume per Agent")
        if agent_df.empty or agent_df['queries_today'].sum() == 0:
            st.info("No queries today")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        timeline = query_logs.copy()
        timeline["date"] = timeline["created_at"].dt.date
        daily_queries = timeline.groupby(["date", "agent_name"]).size().reset_index(name="queries")
        timeline_chart = px.line(daily_queries, x="date", y="queries", color="agent_name", markers=True, title="Daily Queries per Agent")
        st.plotly_chart(timeline_chart, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Agent Overview")
    # st.dataframe(agent_df[["agent_name", "status", "queries_today", "avg_latency", "last_used", "processed_count", "unprocessed_count"]]
    #             .style.highlight_null("red"), use_container_width=True)
    st.dataframe(agent_df[["agent_name", "status", "queries_today", "avg_latency", "last_used"]]
                .style.highlight_null("red"), use_container_width=True)

    # Query Logs Per Agent
    st.subheader("Recent Query Logs")
    for agent in agent_df["agent_name"]:
        with st.expander(f"{agent} - Recent Logs"):
            logs = query_logs[query_logs["agent_name"] == agent]
            logs = logs.sort_values(by="created_at", ascending=False).head(5)
            logs = logs[["query", "answer", "created_at"]]
            st.dataframe(logs, use_container_width=True)

