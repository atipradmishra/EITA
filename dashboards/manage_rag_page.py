import streamlit as st
import sqlite3
import uuid
from datetime import datetime
from config import DB_NAME, aws_access_key, aws_secret_key, VALID_BUCKET
import boto3

# Database connection
def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

# Add agent
def add_agent(name, description, model, temperature, prompt, folder):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    folder_prefix = {
        "CO2": "CO2/",
        "NG": "NG/",
        "PW": "PW/"
    }.get(folder, "misc/")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name='us-east-1'
    )

    try:
        cursor.execute("""
            INSERT INTO rag_agents 
                (name, description, model, temperature, prompt, s3_folder, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name, description, model, temperature, prompt, folder, now
        ))
        # log_id = cursor.lastrowid

        # objects = s3.list_objects_v2(Bucket=VALID_BUCKET, Prefix=folder_prefix).get("Contents", [])
        # for obj in objects:
        #     file_key = obj["Key"]

        #     # Skip if it's a folder or not a .csv
        #     if file_key.endswith("/") or not file_key.lower().endswith(".csv"):
        #         continue

        #     # Skip if the file is in a subfolder (more than one slash after the prefix)
        #     relative_path = file_key[len(folder_prefix):]
        #     if "/" in relative_path:
        #         continue

        #     file_name = relative_path

        #     cursor.execute("""
        #         SELECT 1 FROM data_files_metadata 
        #         WHERE agent_id = ? AND file_name = ? AND is_processed = 1
        #     """, (log_id, file_name))

        #     if cursor.fetchone():
        #         print(f"⏭️ Skipping already processed file: {file_name}")
        #         continue

        #     cursor.execute("""
        #         INSERT INTO data_files_metadata 
        #             (agent_id, file_name, s3_path)
        #         VALUES (?, ?, ?)
        #     """, (log_id, file_name, file_key))


        conn.commit()
        return st.success(f"✅ Agent '{name}' added successfully!")

    except Exception as e:
        print(f"Error adding agent: {str(e)}")
        conn.rollback()
        return False

    finally:
        conn.close()

# Edit agent
def update_agent(agent_id,name, description, model,temperature,prompt, folder):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("UPDATE rag_agents SET name = ?, description = ?,  model = ?, temperature = ?, prompt = ?, s3_folder = ?, updated_at = ? WHERE agent_id = ?",
                   (name, description, model,temperature,prompt, folder, now, agent_id))
    conn.commit()
    conn.close()

    return st.success(f"✅ Agent '{name}' edited successfully!")

# Delete agent
def delete_agent(agent_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM rag_agents WHERE agent_id = ?", (agent_id,))
    conn.commit()
    conn.close()

# Fetch agents
def get_all_agents():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            ra.agent_id,
            ra.name,
            SUM(CASE WHEN pf.is_processed = 1 THEN 1 ELSE 0 END) AS processed_count,
            SUM(CASE WHEN pf.is_processed = 0 THEN 1 ELSE 0 END) AS unprocessed_count,
            ra.description,
            ra.model,
            ra.temperature,
            ra.prompt,
            ra.s3_folder,
            ra.is_active
        FROM 
            rag_agents ra
        LEFT JOIN 
            data_files_metadata pf ON ra.s3_folder = pf.category
        GROUP BY 
            ra.agent_id
    """)
    rows = cursor.fetchall()
    conn.close()
    columns = ["id", "name", "processed_count", "unprocessed_count", "description", "model", "temperature", "prompt", 's3_folder','is_active']
    agents = []
    for row in rows:
        agents.append(dict(zip(columns, row)))
    return agents

def update_agent_status(agent_id, is_active):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE rag_agents 
           SET is_active = ? 
           WHERE agent_id = ?
        """,
        (int(is_active), agent_id)
    )
    conn.commit()
    conn.close()

# Main function for managing agents
def manage_rag_agents():
    st.title("🤖 RAG Agent Manager")
    st.markdown("Manage your Retrieval-Augmented Generation agents with status, descriptions, and file stats.")

    # --- Add New Agent ---
    with st.expander("➕ Add New Agent"):
        with st.form("add_agent_form"):
            prefix_fields = ["CO2","Natural Gas(NG)","Power(PW)"]
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
                match prefix:
                    case "CO2":
                        folder = "CO2"
                    case "Natural Gas(NG)":
                        folder = "NG"
                    case "Power(PW)":
                        folder = "PW"
                    case _:
                        folder = "misc"
                add_agent(name, description, model, temp,prompt, folder)

    st.markdown("---")

    agents = get_all_agents()

    # --- Display Agents ---
    st.subheader("📋 Existing RAG Agents")
    for agent in agents:
        with st.container():
            st.markdown(f"#### {agent['name']}")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"📝 *{agent['description']}*")
                current_status = agent["is_active"]
                new_status = st.toggle(
                    "Active",
                    value=current_status,
                    key=f"active_{agent['id']}"
                )
                
                if new_status != current_status:
                    update_agent_status(agent['id'], new_status)
                    st.success(f"Status updated for agent {agent['name']}")
            with col2:
                co4, col5, col6 = st.columns(3)
                with co4:
                    st.metric("📂 Files", f"{agent['processed_count'] + agent['unprocessed_count']} files")
                with col5:
                    st.metric("✅ Processed", f"{agent['processed_count']} files")
                with col6:
                    st.metric("🕓 Unprocessed", f"{agent['unprocessed_count']} files")
            
            with col3:
                disabled = agent["unprocessed_count"] == 0
                co7, col8 = st.columns(2)
                with co7:
                    if st.button("⚙️ Process Files", key=f"process_{agent['id']}", disabled=disabled, use_container_width=True):
                        process_files_from_s3_folder(agent['s3_folder'])
                        st.success(f"Processing files for {agent['name']}")
                with col8:
                    if st.button("🗑️ Delete Agent", key=f"delete_{agent['id']}"):
                        delete_agent(agent['id'])
                        st.rerun()
        
            with st.expander(f"✏️ Editing: {agent['name']}"):
                with st.form(f"edit_form_{agent['id']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("RAG Agent Name", value=agent["name"])
                    with col2:
                        current_prefix = agent["s3_folder"]
                        prefix_mapping = {
                            "CO2": "CO2",
                            "NG": "Natural Gas(NG)",
                            "PW": "Power(PW)",
                        }
                        selected_prefix = prefix_mapping.get(current_prefix, "CO2")
                        selected_index = prefix_fields.index(selected_prefix)
                        prefix = st.selectbox("Select folder", prefix_fields, index=selected_index)
                    col3, col4 = st.columns(2)
                    with col3:
                        model_options = [
                            "OpenAI GPT-3.5",
                            "OpenAI GPT-4",
                            "Llama 2",
                            "Claude 3.5",
                            "Claude 4",
                            "Custom Model"
                        ]
                        current_model = agent["model"]

                        if current_model in model_options:
                            selected_model_index = model_options.index(current_model)
                        else:
                            selected_model_index = 0

                        model = st.selectbox("Model", model_options, index=selected_model_index)
                    with col4:
                        temp = st.slider("Temperature", 0.0, 1.0, value=float(agent["temperature"]), step=0.1)
                    col5, col6 = st.columns(2)
                    with col5:
                        description = st.text_area("Description", value=agent["description"])
                    with col6:
                        prompt = st.text_area("Prompt", value=agent["prompt"])
                    col7, col8 = st.columns(2)
                    with col7:
                        if st.form_submit_button("💾 Save Changes"):
                            match prefix:
                                case "CO2":
                                    prefix = "CO2"
                                case "Natural Gas(NG)":
                                    prefix = "NG"
                                case "Power(PW)":
                                    prefix = "PW"
                                case _:
                                    prefix = "misc"
                            update_agent(agent["id"], name, description, model, temp, prompt, prefix)
                            st.success("Agent updated successfully.")
                            st.session_state.edit_agent_id = None
                    with col8:
                        if st.form_submit_button("❌ Cancel"):
                            st.session_state.edit_agent_id = None
                            st.rerun()

        st.markdown("---")
