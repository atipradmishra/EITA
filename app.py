import streamlit as st
import sqlite3
import pandas as pd
import time
import json
from io import BytesIO
import io
import os
import matplotlib.pyplot as plt
import re
from openai import OpenAI
from config import aws_access_key, aws_secret_key, client, DB_NAME, VALID_BUCKET, REJECTED_BUCKET
from DataProcessing.GraphDataProcess import process_and_save_from_s3, save_jsondata_to_db, process_csv_data_to_json , save_rawdata_to_db, process_and_save_byfile
from GraphFunctions.dashboardgraphs import show_top5_movers,plot_delta_market_value_by_horizon, plot_delta_market_value_by_horizon_by_tgroup1, plot_delta_volume_by_horizon, plot_delta_volume_by_horizon_by_tgroup1, plot_delta_volume_by_horizon_by_bookattr8,plot_delta_market_value_by_horizon_by_bookattr8
from GraphFunctions.heatmaptable import plot_heatmap
from GraphFunctions.dashboardcards import show_nop_cards,render_summary_card
from GraphFunctions.querygraphs import plot_combined_graph_CO2, plot_combined_graph_NG, plot_combined_graph_PW, plot_graph_based_on_prompt_all, plot_dimension_data, plot_single_date_metrics, plot_time_series
from DbUtils.models import create_db
from DbUtils.auth import register_user, authenticate_user, logout
from DataProcessing.DataProcess import process_files_from_s3_folder,process_metadata_alias
from utils import upload_to_s3,save_metadata_to_db, upload_metadatafile_to_s3, validate_against_metadata,query_sqlite_json_with_openai,create_faiss_index,prepare_training_data,save_training_data,extract_date_from_query,calculate_confidence_score,retrieve_feedback_insights
from DbUtils.DbOperations import add_feedback_log, get_feedback_logs, update_feedback_log, get_existing_metadata, load_feedback_data, load_data_for_dashboard
from dashboards.file_tracking_dashboard import main as render_file_tracking_dashboard
from dashboards.rag_dashboard import rag_agents_dashboard
from dashboards.manage_rag_page import manage_rag_agents


st.set_page_config(page_title="EITA", layout="wide")

create_db()

# --- Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "main_section" not in st.session_state:
    st.session_state.main_section = "Data Management AI Agent"
if "sub_section" not in st.session_state:
    st.session_state.sub_section = "Pipeline Dashboard"
if "feedback_logs" not in st.session_state:
    try:
        with open("energy_trading_feedback.json", "r") as f:
            st.session_state.feedback_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.feedback_logs = []
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = 0.7
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {
            "role": "system",
            "content": "You are a strict assistant. If feedback directly answers the question, always prioritize it over the data."
        }
    ]

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        
        /* Title styling */
        .card-title {
            color: #666;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        /* Value styling */
        .card-value {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        /* Delta styling */
        .card-delta {
            color: var(--secondary-color);
            font-size: 18px;
            font-weight: 500;
        }
        
        /* Update styling */
        .card-update {
            color: #aaa;
            font-size: 12px;
            margin-top: 8px;
        }
        [data-testid="stSidebar"] { box-shadow: 2px 0 10px rgba(0,0,0,0.1); }
        button { border-radius: 10px !important; }
        button:hover { background: #007BFF !important; color: white !important; border: none !important; }
        h1, h2, h3 { color: #007BFF; }
        hr { border: 1px solid #ccc; }
        .login-btn { width: 100%; background-color: #4CAF50; color: white; padding: 12px 20px; margin: 10px 0; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: 0.3s; }
        .login-btn:hover { background-color: #45a049; }
        .login-input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar and User Menu ---
def sidebar_menu():
    with st.sidebar:
        col1 , col2, col3 = st.columns(3)
        with col1:
            st.write(" ")
        with col2:
            st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95")
        with col3:
            st.write(" ")
        st.markdown("<h2 style='text-align: center;'>ETAI Energy Trading AI Platform</h2>", unsafe_allow_html=True)
        st.divider()
        main_section_options = ["Data Management AI Agent", "RAG AI Agent", "Application AI Agent"]
        main_section = st.radio("Select AI Agent", main_section_options, index=main_section_options.index(st.session_state.main_section))
        st.session_state.main_section = main_section

        sub_sections = {
            "Data Management AI Agent": ["Data File Tracking Dashboard", "Data Pipeline", "Processed Data"],
            "RAG AI Agent": ["RAG Dashboard", "Manage RAG Agents", "Fine Tuning", "Settings"],
            "Application AI Agent": ["Dashboard","Energy Trading Analysis", "Graph Query", "Deviation Analysis", "Root Cause Analysis", "Analysis History", "User Feedback"]
        }
        if st.session_state.sub_section not in sub_sections[main_section]:
            st.session_state.sub_section = sub_sections[main_section][0]
        sub_section = st.radio(f"Select {main_section} Section", sub_sections[main_section], index=sub_sections[main_section].index(st.session_state.sub_section))
        st.session_state.sub_section = sub_section
        st.divider()


def top_right_menu():
    username = st.session_state.get('username', 'Guest')
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"👤 **{username}**", unsafe_allow_html=True)
    with col2:
        if st.button("🔴 Logout", key="logout_btn", help="Logout from the platform"):
            logout()


# --- User Authentication Interface ---
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        col1_img, col2_img, col3_img = st.columns(3)
        with col2_img:
            st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95", width=150)
        st.title("🔐 Welcome to ETAI Energy Trading AI Platform")
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            st.header("🔑 Login")
            username = st.text_input("Username", key="Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="password", placeholder="Enter your password")
            if st.button("Login", use_container_width=True, key="login", help="Click to login"):
                if authenticate_user(username, password):
                    st.success("✅ Login successful!")
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
        with tab2:
            st.header("📝 Register")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Register", use_container_width=True):
                if register_user(new_username, new_password):
                    st.success("✅ User registered successfully!")
                else:
                    st.error("❌ Username already exists. Choose a different one.")
else:
    sidebar_menu()
    top_right_menu()
    # st.title(f"📊 {st.session_state.main_section}")

    if st.session_state.sub_section == "Data File Tracking Dashboard":
        render_file_tracking_dashboard()

    elif st.session_state.sub_section == "Data Pipeline":
        st.header("📊 Data Pipeline")
        st.subheader("🔧 Data Pipeline -> Energy Training")
        st.write("Upload and manage your data files efficiently.")
        st.markdown("---")

        # Section 1: CO2 Data Upload
        st.markdown("### 🗂️ Data Upload")
        folder = st.selectbox("Select Folder to Upload", ["CO2","NG", "PW"])
        col1, col2 = st.columns(2)
        with col1:
            metadata_file1 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file1")
        with col2:
            raw_files1 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files1", accept_multiple_files=True)

        metadata_df = None
        if metadata_file1:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file1.getvalue()))
                alias_mapping = process_metadata_alias(metadata_df)
                upload_metadatafile_to_s3(metadata_file1, metadata_file1.name, VALID_BUCKET)
                st.success("✅ Metadata processed successfully and uploaded.")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing uploaded metadata: {e}")
        else:
            try:
                metadata_df = get_existing_metadata(folder)
                if metadata_df is not None:
                    st.success(f"✅ Loaded existing metadata from database.")
                    alias_mapping = process_metadata_alias(metadata_df)
                    st.write("Alias Mapping (From DB):")
                    st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
                else:
                    st.warning(f"⚠️ No metadata found. Please upload the metadata file.")
            except Exception as e:
                st.error(f"❌ Error loading metadata from DB: {e}")

        if st.button("Submit", key="co2"):
            if metadata_df is not None and raw_files1:
                try:
                    for f in raw_files1:
                        file_name = f.name
                        conn = sqlite3.connect(DB_NAME)
                        process_and_save_byfile(conn, f,folder)
                        try:
                            f.seek(0)
                            df = pd.read_csv(f)
                            file_name = f.name
                            if validate_against_metadata(df, metadata_df, file_name):
                                f.seek(0)
                                success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                                st.success(f"✅ VALID: {file_name} | {msg}")
                            else:
                                f.seek(0)
                                success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                                st.error(f"❌ INVALID: {file_name} | {msg}")
                        except Exception as file_err:
                            st.error(f"❌ Error processing {file_name}: {file_err}")
                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Metadata and raw files are required to proceed.")
            
        
        st.markdown("### 📘 Business Domain Dictionary")
        business_dict_file = st.file_uploader("Upload Business Domain Dictionary (.xlsx)", type=["xlsx"], key="business_dict_file")

        if business_dict_file:
            try:
                # Get original uploaded filename
                uploaded_file_name = business_dict_file.name

                # Save uploaded file to a temporary path
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_excel_path = os.path.join(temp_dir, uploaded_file_name)

                with open(temp_excel_path, "wb") as f:
                    f.write(business_dict_file.getvalue())

                # Import and process using your function
                from business_context import load_business_context
                business_dict, _ = load_business_context(temp_excel_path)

                st.success("✅ Business Domain Dictionary processed successfully.")

                if st.button("Submit", key="bdict"):
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        for sheet_name, mappings in business_dict.items():
                            if isinstance(mappings, dict):
                                for key, val in mappings.items():
                                    cursor.execute("""
                                        INSERT INTO Business_Context (context_name, description, source_file)
                                        VALUES (?, ?, ?)
                                    """, (key, val, uploaded_file_name))
                            elif isinstance(mappings, list):
                                for record in mappings:
                                    cursor.execute("""
                                        INSERT INTO Business_Context (context_name, description, source_file)
                                        VALUES (?, ?, ?)
                                    """, (record.get("col_0", ""), record.get("col_1", ""), uploaded_file_name))
                        conn.commit()
                        conn.close()
                        st.success("✅ Data successfully inserted into Business_Context table.")
                    except Exception as db_err:
                        st.error(f"❌ Database error: {db_err}")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        else:
            st.info("ℹ️ Please upload a Business Domain Dictionary Excel (.xlsx) file.")

        # st.markdown("---")

        # st.markdown("### 🔥 Natural Gas Data Upload")
        # col3, col4 = st.columns(2)
        # with col3:
        #     metadata_file2 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file2")
        # with col4:
        #     raw_files2 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files2", accept_multiple_files=True)

        # dataset_type_ng = "NG"
        # metadata_df_ng = None

        # if metadata_file2:
        #     try:
        #         metadata_df_ng = pd.read_csv(BytesIO(metadata_file2.getvalue()))
        #         alias_mapping = process_metadata_alias(metadata_df_ng)
        #         upload_metadatafile_to_s3(metadata_file2, metadata_file2.name, VALID_BUCKET)
        #         st.success("✅ Metadata processed successfully and uploaded.")
        #         st.write("Alias Mapping:")
        #         st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
        #     except Exception as e:
        #         st.error(f"❌ Error processing uploaded metadata: {e}")
        # else:
        #     try:
        #         metadata_df_ng = get_existing_metadata(dataset_type_ng)
        #         if metadata_df_ng is not None:
        #             st.success(f"✅ Loaded existing metadata for {dataset_type_ng} from database.")
        #             alias_mapping = process_metadata_alias(metadata_df_ng)
        #             st.write("Alias Mapping (From DB):")
        #             st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
        #         else:
        #             st.warning(f"⚠️ No metadata found for {dataset_type_ng}. Please upload the metadata file.")
        #     except Exception as e:
        #         st.error(f"❌ Error loading metadata from DB: {e}")

        # # Submission
        # if st.button("Submit", key="ng"):
        #     if metadata_df_ng is not None and raw_files2:
        #         try:
        #             for f in raw_files2:
        #                 try:
        #                     f.seek(0)
        #                     df = pd.read_csv(f)
        #                     file_name = f.name
        #                     if validate_against_metadata(df, metadata_df_ng, file_name):
        #                         f.seek(0)
        #                         success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
        #                         st.success(f"✅ VALID: {file_name} | {msg}")
        #                     else:
        #                         f.seek(0)
        #                         success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
        #                         st.error(f"❌ INVALID: {file_name} | {msg}")
        #                 except Exception as file_err:
        #                     st.error(f"❌ Error processing {file_name}: {file_err}")
        #         except Exception as e:
        #             st.error(f"❌ Error during submission: {e}")
        #     else:
        #         st.warning("⚠️ Metadata and raw files are required to proceed.")

        # st.markdown("---")

        # # Section 3: Power Data Upload
        # st.markdown("### 🚀 Power Data Upload")
        # col5, col6 = st.columns(2)
        # with col5:
        #     metadata_file3 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file3")
        # with col6:
        #     raw_files3 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files3", accept_multiple_files=True)

        # dataset_type_pw = "PW"
        # metadata_df_pw = None

        # if metadata_file3:
        #     try:
        #         metadata_df_pw = pd.read_csv(BytesIO(metadata_file3.getvalue()))
        #         alias_mapping = process_metadata_alias(metadata_df_pw)
        #         upload_metadatafile_to_s3(metadata_file3, metadata_file3.name, VALID_BUCKET)
        #         st.success("✅ Metadata processed successfully and uploaded.")
        #         st.write("Alias Mapping:")
        #         st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
        #     except Exception as e:
        #         st.error(f"❌ Error processing uploaded metadata: {e}")
        # else:
        #     try:
        #         metadata_df_pw = get_existing_metadata(dataset_type_pw)
        #         if metadata_df_pw is not None:
        #             st.success(f"✅ Loaded existing metadata for {dataset_type_pw} from database.")
        #             alias_mapping = process_metadata_alias(metadata_df_pw)
        #             st.write("Alias Mapping (From DB):")
        #             st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
        #         else:
        #             st.warning(f"⚠️ No metadata found for {dataset_type_pw}. Please upload the metadata file.")
        #     except Exception as e:
        #         st.error(f"❌ Error loading metadata from DB: {e}")

        # if st.button("Submit", key="po"):
        #     if metadata_df_pw is not None and raw_files3:
        #         try:
        #             for f in raw_files3:
        #                 file_name = f.name
        #                 conn = sqlite3.connect(DB_NAME)
        #                 process_and_save_byfile(conn, f)
        #                 try:
        #                     f.seek(0)
        #                     df = pd.read_csv(f)
        #                     if validate_against_metadata(df, metadata_df_pw, file_name):
        #                         f.seek(0)
        #                         success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
        #                         st.success(f"✅ VALID: {file_name} | {msg}")
        #                     else:
        #                         f.seek(0)
        #                         success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
        #                         st.error(f"❌ INVALID: {file_name} | {msg}")
        #                 except Exception as file_err:
        #                     st.error(f"❌ Error processing {file_name}: {file_err}")
        #         except Exception as e:
        #             st.error(f"❌ Error during submission: {e}")
        #     else:
        #         st.warning("⚠️ Metadata and raw files are required to proceed.")

        # st.markdown("---")

    elif st.session_state.sub_section == "Processed Data":
        st.subheader("📊 Processed Data")
        st.write("Access and analyze the processed data records.")
    
    elif st.session_state.sub_section == "RAG Dashboard":
        rag_agents_dashboard()
    
    # elif st.session_state.sub_section == "Manage RAG Agents":
    #     # manage_rag_agents()
    
    elif st.session_state.sub_section == "Manage RAG Agents":
        prefix_fields = {
            "CO2": ['VOLUME', 'TRDVAL', 'MKTVAL', 'TRDPRC'],
            "Natural Gas": ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL'],
            "Power": ['VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK', 'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
        }
        name = st.text_input("RAG  Agent Name")
        col1, col2 = st.columns(2)
        with col1:
            bucket = st.selectbox("Select Bucket Name(S3)", ["etrm-etai-poc-chub","etrm-etai-poc", "etrm-etai-poc-ng"])
        with col2:
            prefix = st.selectbox("Select Prefix", list(prefix_fields.keys()), index=0)
        col3, col4 = st.columns(2)
        with col3:
            model = st.selectbox("Model", ["OpenAI GPT-3.5", "OpenAI GPT-4", "Llama 2", "Claude 3.5", "Claude 4", "Custom Model"])
        with col4:
            temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
        col5, col6 = st.columns(2)
        with col5:
            metadata_file = st.file_uploader("Upload Data Dictionary (CSV)", type=["csv"])
        with col6:
            uploaded_file = st.file_uploader("Upload Transaction Log (TXT, PDF, CSV, DOCX)", type=["txt", "pdf", "csv", "docx"])
        prompt = st.text_area("📝 Provide Prompt Instructions", key='prompt')
        if st.button("Submit & Process Data"):
            prefix_value = {"CO2": "CO2", "Natural Gas": "NG", "Power": "PW"}.get(prefix, "misc")
            process_files_from_s3_folder(VALID_BUCKET, prefix_value)
            # add_agent_detail(name, model, temp, prompt)

    elif st.session_state.sub_section == "Fine Tuning":
        st.subheader("📄 Fine Tuning")

    elif st.session_state.sub_section == "Settings":
        st.subheader("🚀 Settings")
        st.write("Get insights into your application's performance.")

    elif st.session_state.sub_section == "Dashboard":
        st.subheader("📚 Dashboard")
        # if st.button("🔄 Update from S3"):
        #     with sqlite3.connect(DB_NAME) as conn:
        #         process_and_save_from_s3(conn, VALID_BUCKET, aws_access_key, aws_secret_key)
        #     st.success("Data updated!")
        
        data = load_data_for_dashboard()
        segment_json = data.get("by_segment", {})
        bookattr_json = data.get("by_book_attr8", {})
        tgroup_json = data.get("by_tgroup1", {})
        heatmapdata = data.get("heatmap_table", {})
        summary = data.get("daily_summary_totals", {})

        segment_options = list(segment_json.keys())
        bookattr_options = list(bookattr_json.keys())
        tgroup1_options = list(tgroup_json.keys())

        conn = sqlite3.connect(DB_NAME)
        render_summary_card(summary,client,conn)

        show_nop_cards(data)

        selected_segment = st.selectbox("Select Segment", segment_options)

        col1, col2= st.columns(2)
        with col1:
            plot_delta_volume_by_horizon( segment_json, selected_segment)
        with col2:
            plot_delta_market_value_by_horizon(segment_json, selected_segment)
        
        selected_tgroup1 = st.selectbox("Select Primary Strategy", tgroup1_options)
        col5, col6= st.columns(2)
        with col5:
            plot_delta_volume_by_horizon_by_tgroup1( tgroup_json, selected_tgroup1)
        with col6:
            plot_delta_market_value_by_horizon_by_tgroup1(tgroup_json, selected_tgroup1)

        selected_book_attr = st.selectbox("Select Business Classification", bookattr_options)
        col3, col4= st.columns(2)
        with col3:
            plot_delta_volume_by_horizon_by_bookattr8(bookattr_json, selected_book_attr)
        with col4:
            plot_delta_market_value_by_horizon_by_bookattr8(bookattr_json, selected_book_attr)

        col5, col6= st.columns(2)
        with col5:
            row_option = st.radio("Choose row dimension", ["BUCKET", "HORIZON"], horizontal=True)
            plot_heatmap(heatmapdata, row_option)
        with col6:
            show_top5_movers(data)
        
    elif st.session_state.sub_section in ["Energy Trading Analysis", "Energy Trading Analysis"]:
        if "query_answer" not in st.session_state:
            st.session_state["query_answer"] = None
        if "feedback_submitted" not in st.session_state:
            st.session_state["feedback_submitted"] = False
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("📊 Energy Trading Analysis")
        st.write("💬 Ask Your Financial or Commodity Question")

        col1, col2 = st.columns([6, 1])
        with col1:
            category_options = ["CO2", "Natural Gas", "Power"]
            selected_category = st.selectbox("Select Data Category", category_options)

        with col2:
            st.write(" ")
            if st.button("🧹 Clear Chat"):
                st.session_state.messages.clear()

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

            if msg["role"] == "assistant":
                feedback_key = f"feedback_{i}"
                col1, col2 = st.columns([1, 6])

                with col1:
                    if not msg.get("feedback"):
                        if st.button("👍", key=f"{feedback_key}_up"):
                            st.session_state.messages[i]["feedback"] = "positive"
                            st.session_state[f"show_comment_{i}"] = False
                            st.success("Thanks for your feedback!")

                        if st.button("👎", key=f"{feedback_key}_down"):
                            st.session_state.messages[i]["feedback"] = "negative"
                            st.session_state[f"show_comment_{i}"] = True
                    else:
                        st.markdown("✅ Feedback recorded")

                if (
                    st.session_state.get(f"show_comment_{i}", False)
                    and "feedback_comment" not in msg
                ):
                    comment_key = f"comment_input_{i}"
                    comment = st.text_area("😞 What went wrong?", key=comment_key)
                    if st.button("Submit Feedback", key=f"submit_feedback_{i}"):
                        st.session_state.messages[i]["feedback_comment"] = comment
                        update_feedback_log(0, comment, st.session_state.messages[i].get("log_id", None))
                        st.success("Thanks for your feedback!")
                        st.session_state[f"show_comment_{i}"] = False

        if prompt := st.chat_input("Ask a question... e.g(What is the total Price Value on 13 Nov 2024?)"):

            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.spinner("Thinking..."):
                response = query_sqlite_json_with_openai(prompt, selected_category)
                log_id = add_feedback_log(prompt, response, selected_category)

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "log_id": log_id
            })
            st.rerun()

    elif st.session_state.sub_section == "Graph Query":
        st.subheader("📊 Graph Query")
        category_selected = st.selectbox("Select Category", ["CO2", "Natural Gas", "Power"])
        graph_query_input = st.text_input('Ask for a bar graph (e.g., "Show bar graph for TRDVAL")')
        if st.button('Generate Custom Graph'):
            plot_graph_based_on_prompt_all(graph_query_input, category_selected)
        if st.button('Combined CO2 Graph'):
            plot_combined_graph_CO2(graph_query_input)
        if st.button('Combined NG Graph'):
            plot_combined_graph_NG(graph_query_input)
        if st.button('Combined PW Graph'):
            plot_combined_graph_PW(graph_query_input)

    elif st.session_state.sub_section == "Deviation Analysis":
        st.subheader("📊 Deviation Analysis")

    elif st.session_state.sub_section == "Root Cause Analysis":
        st.subheader("⚠️ Root Cause Analysis")
        st.write("Review and troubleshoot errors.")
        st.header("📊 Analysis History & Insights")

    elif st.session_state.sub_section == "User Feedback":
        st.subheader("📝 User Feedback Dashboard")
        feedback_df = get_feedback_logs()
        if not feedback_df.empty:
            st.subheader("📋 Collected Feedback")
            display_df = feedback_df[["id", "query", "answer", "feedback_comment", "User Feedback", "Timestamp"]]
            st.dataframe(display_df, use_container_width=True)
            st.subheader("📊 Feedback Summary")
            positive_feedback_count = (feedback_df["user_feedback"] == 1).sum()
            negative_feedback_count = (feedback_df["user_feedback"] == 0).sum()
            st.write(f"✅ **Positive Feedback:** {positive_feedback_count}")
            st.write(f"❌ **Negative Feedback:** {negative_feedback_count}")
            st.subheader("📊 Feedback Trends Over Time")
            feedback_over_time = (
                feedback_df.groupby(feedback_df["Timestamp"].dt.date)["feedback_comment"]
                .value_counts()
                .unstack()
                .fillna(0)
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            feedback_over_time.plot(kind="bar", ax=ax, stacked=True)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Feedbacks")
            ax.set_title("User Feedback Trends")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            csv_feedback = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Feedback as CSV", csv_feedback, "user_feedback.csv", "text/csv", key="download-feedback")
        else:
            st.warning("No feedback logs found in the database.")