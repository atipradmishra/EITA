import streamlit as st
import sqlite3
import uuid
from datetime import datetime
from config import DB_NAME

# Database connection
def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

# Add agent
def add_agent(name, description, model,temperature,prompt, folder):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("INSERT INTO rag_agents (name, description, model,temperature,prompt, s3_folder, updated_at) VALUES (?, ?, ?, ?, ?,?,?)",
                   (name, description, model,temperature,prompt, folder, now))
    conn.commit()
    conn.close()

# Edit agent
def update_agent(name, description, model,temperature,prompt, folder):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("UPDATE rag_agents SET name = ?, description = ?,  model = ?, temperature = ?, prompt = ?, s3_folder = ?, updated_at = ? WHERE id = ?",
                   (name, description, model,temperature,prompt, folder, now, agent_id))
    conn.commit()
    conn.close()

# Delete agent
def delete_agent(agent_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM rag_agents WHERE id = ?", (agent_id,))
    conn.commit()
    conn.close()

# Fetch agents
def get_all_agents():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, description, model,temperature, prompt FROM rag_agents")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Main function for managing agents
def manage_rag_agents():
    st.title("🤖 RAG Agent Manager")
    st.markdown("Manage your Retrieval-Augmented Generation agents with status, descriptions, and file stats.")

    # --- Add New Agent ---
    with st.expander("➕ Add New Agent"):
        with st.form("add_agent_form"):
            prefix_fields = ["CO2","Natural Gas(G)","Power(PW)"]
            col1, col2 = st.columns(2)
            with col1:  
                name = st.text_input("RAG  Agent Name")
            with col2:
                prefix = st.selectbox("Select folder from Bucket (S3) to process file", prefix_fields, index=0)
            col3, col4 = st.columns(2)
            with col3:
                model = st.selectbox("Model", ["OpenAI GPT-3.5", "OpenAI GPT-4", "Llama 2", "Claude 3.5", "Claude 4", "Custom Model"])
            with col4:
                temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
            col5, col6 = st.columns(2)
            with col5:
                description = st.text_area("Agent Description", placeholder="Describe what this agent does...")
                # metadata_file = st.file_uploader("Upload Data Dictionary (CSV)", type=["csv"])
            with col6:
                # uploaded_file = st.file_uploader("Upload Transaction Log (TXT, PDF, CSV, DOCX)", type=["txt", "pdf", "csv", "docx"])
                prompt = st.text_area("📝 Provide Prompt Instructions", key='prompt')
            if  st.form_submit_button("Add Agent"):
                folder = {"CO2": "CO2", "Natural Gas": "NG", "Power": "PW"}.get(prefix, "misc")
                # process_files_from_s3_folder(VALID_BUCKET, prefix_value)
                add_agent(name, description, model, temp,prompt, folder)

    st.markdown("---")

    # --- Dummy Data (replace with DB query) ---
    dummy_agents = [
        {
            "id": "1",
            "name": "PowerBot",
            "description": "Handles power grid documents.",
            "active": True,
            "processed_files": 22,
            "unprocessed_files": 0
        },
        {
            "id": "2",
            "name": "CO2Bot",
            "description": "Handles CO2 documents.",
            "active": False,
            "processed_files": 10,
            "unprocessed_files": 5
        }
    ]

    # --- Display Agents ---
    st.subheader("📋 Existing RAG Agents")
    for agent in dummy_agents:
        with st.container():
            st.markdown(f"#### {agent['name']}")
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"📝 *{agent['description']}*")
                st.toggle("Active", value=agent["active"], key=f"active_{agent['id']}")
            with col2:
                co4, col5, col6 = st.columns(3)
                with co4:
                    st.metric("📂 Files", f"{agent['processed_files'] + agent['unprocessed_files']} files")
                with col5:
                    st.metric("✅ Processed", f"{agent['processed_files']} files")
                with col6:
                    st.metric("🕓 Unprocessed", f"{agent['unprocessed_files']} files")
            with col3:
                disabled = agent["unprocessed_files"] == 0
                st.button("⚙️ Process Files", key=f"process_{agent['id']}", disabled=disabled, use_container_width=True)
                st.button("💾 Save Changes", key=f"save_{agent['id']}", use_container_width=True)
                st.button("🗑️ Delete Agent", key=f"delete_{agent['id']}", use_container_width=True)


            st.markdown("---")