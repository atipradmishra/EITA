import pandas as pd
import sqlite3
import boto3
import streamlit as st
import faiss
import numpy as np
import json
import io
import re
import datetime
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from DbUtils.DbOperations import load_feedback_data, load_business_context
from config import aws_access_key, aws_secret_key, client, DB_NAME, VALID_BUCKET, REJECTED_BUCKET
import tiktoken

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Assume you are using GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

def upload_to_s3(file, filename, bucket):
    try:
        folder = "misc/"
        if "NOP_CO2" in filename.upper():
            folder = "CO2/"
        elif "NOP_NG" in filename.upper():
            folder = "NG/"
        elif "NOP_PW" in filename.upper():
            folder = "PW/"
        s3_key = folder + filename
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )
        s3.upload_fileobj(file, bucket, s3_key)


        return True, f"✅ Uploaded {filename} to S3 bucket '{bucket}' in folder '{folder}'"
    except Exception as e:
        return False, f"❌ Upload failed: {e}"

def save_metadata_to_db(filename, df, s3_path):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    json_data = df.to_json(orient="records")
    cursor.execute("""
        INSERT INTO metadata_files (filename, s3_path, json_data)
        VALUES (?, ?, ?)
    """, (filename, s3_path, json_data))
    conn.commit()
    conn.close()

def upload_metadatafile_to_s3(file, filename, bucket):
    try:
        folder = "misc/"
        if "CO2" in filename.upper():
            folder = "CO2/metadata/"
        elif "NG" in filename.upper():
            folder = "NG/metadata/"
        elif "PW" in filename.upper():
            folder = "PW/metadata/"

        s3_key = folder + filename

        file.seek(0)
        file_bytes = file.read()

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )
        s3.upload_fileobj(io.BytesIO(file_bytes), bucket, s3_key)

        df = pd.read_csv(io.BytesIO(file_bytes))
        save_metadata_to_db(filename, df, s3_key)

        return True, f"✅ Uploaded {filename} to S3 bucket '{bucket}' in folder '{folder}'"
    except Exception as e:
        return False, f"❌ Upload failed: {e}"

def validate_against_metadata(file_df: pd.DataFrame, metadata_df: pd.DataFrame, file_id: str) -> bool:
    file_cols = set(file_df.columns.str.strip().str.upper())
    metadata_cols = set(metadata_df["Field_Name"].str.strip().str.upper())
    st.subheader(f"🔍 Column Comparison for `{file_id}`")
    st.write("🗂 **Data File Columns**:")
    st.code("\n".join(sorted(file_cols)))
    st.write("📘 **Metadata Columns**:")
    st.code("\n".join(sorted(metadata_cols)))
    missing_in_file = metadata_cols - file_cols
    extra_in_file = file_cols - metadata_cols
    if missing_in_file:
        st.error(f"❌ Columns missing in data file: {', '.join(missing_in_file)}")
    if extra_in_file:
        st.warning(f"⚠️ Extra columns in data file: {', '.join(extra_in_file)}")
    if not missing_in_file:
        st.success("✅ Validation Passed.")
        return True
    else:
        st.error("❌ Validation Failed.")
        return False


def query_sqlite_json_with_openai(user_question, category,bot_prompt_instr):
    from DbUtils.sqlquery import run_orchestrated_agent

        # Ensure conversation_history exists in session state
    if not hasattr(st.session_state, 'conversation_history'):
        st.session_state.conversation_history = []
    
    if not hasattr(st.session_state, 'conversation'):
        st.session_state.conversation = [{"role": "system", "content": "You are a helpful assistant."}]
        
    # Step 0: Load your business glossary
    business_dict = load_business_context()

    # Step 1: Feedback / FAISS
    feedback_data = load_feedback_data()
    faiss_index, feedback_data_indexed = create_faiss_index(feedback_data)
    feedback_insights = retrieve_feedback_insights(
        user_question, faiss_index, feedback_data_indexed
    )

    # Step 2: Orchestrated SQL (single or multi-query)
    result, is_multi = run_orchestrated_agent(user_question, category, conversation_history=st.session_state.conversation_history)

    # Build all_context from the orchestrator's output
    if is_multi:
        # already a narrative summary
        all_context = result["synthesis"]
    else:
        # Handle potential error cases or missing keys in the result
        if "error" in result:
            all_context = f"Error in SQL query: {result['error']}"
        elif "columns" not in result or "rows" not in result:
            all_context = "Error: SQL query returned an unexpected structure"
        else:
            # simple table: columns + rows
            cols = result["columns"]
            rows = result["rows"]
            header = " | ".join(cols)
            body = "\n".join(" | ".join(map(str, row)) for row in rows)
            all_context = f"{header}\n{body}"

    # Check the token count of all_context
    all_context_token_count = len(enc.encode(all_context))
    if all_context_token_count > 7090:
        # Return a message to the user if the context is too large
        return "We've encountered a data volume limitation while processing your question. The information returned exceeds our current processing capacity (Token Limit). To receive a complete and accurate response, please refine your question with more specific parameters, such as:\n* Narrowing the date range\n* Adding more precise filters\n* Including aggregation terms (sum, average, total)\n* Focusing on specific metrics or dimensions"

    # We no longer extract file_names from JSON, so keep empty
    file_names = []

    # Step 3: Pull Prompt Instruction for this category
    # conn = sqlite3.connect(DB_NAME)
    # cursor = conn.cursor()
    # if category == "CO2":
    #     cursor.execute("SELECT prompt FROM agent_detail WHERE name = 'CO2 AGENT'")
    # elif category == "Natural Gas":
    #     cursor.execute("SELECT prompt FROM agent_detail WHERE name = 'NG AGENT'")
    # elif category == "Power":
    #     cursor.execute("SELECT prompt FROM agent_detail WHERE name = 'PW AGENT'")
    # else:
    #     cursor.execute("SELECT prompt FROM agent_detail WHERE name = 'PW AGENT'")
    # row = cursor.fetchone()
    # conn.close()

    # if not row:
    #     return "⚠️ No Prompt Instruction Found."
    # prompt_Instr = row[0]
    
    prompt_Instr = bot_prompt_instr
    if prompt_Instr == None:
        return "⚠️ No Prompt Instruction Found."

    # Step 4: Format business glossary
    business_context_text = "\n".join(
        f"{k}: {v}" if isinstance(v, str)
        else f"{k}: {', '.join(f'{x} = {y}' for x,y in v.items())}"
        for k, v in business_dict.items()
    )

    # Step 5: Build feedback block
    feedback_text = ""
    if feedback_insights:
        feedback_text = (
            "\n\nBased on feedback from similar queries, please be aware of these issues:\n"
            + "\n".join(f"- {insight}" for insight in feedback_insights)
        )

    # Step 6: Category context phrase
    category_context = (
        f"for the {category} category"
        if category and category != "All"
        else ""
    )

    # Step 7: Compose the system prompt for the main answer
    context_message = f"""
📌 VERIFIED FEEDBACK (Authoritative Corrections):
Use this section as the most accurate reference if it directly answers the question.
{feedback_text}

📘 BUSINESS GLOSSARY (Terms & Labels):
This glossary helps you interpret technical terms into business language when answering user questions.
reat 'Net open position' questions as the sum of VOLUME_BL for the REPORT_DATE unless the context clearly refers to monetary value, in which case use MKT_VAL_BL + {business_context_text}

🗂️ PREPROCESSED DATA (Official Trading Statistics):
This section contains structured trading data {category_context}.
{all_context} 

⚠️ INSTRUCTION (STRICTLY FOLLOW THESE STEPS — DO NOT IGNORE):
You are an expert Energy Trading and Risk Management (ETRM) analyst. Answer the user's question precisely using only the data provided above. Provide clear, concise explanations with specific numbers when available.

{prompt_Instr}
"""

    # Token count for full context
    token_count = len(enc.encode(context_message))
    st.write("🧮 Token count:", token_count)
    # Also display the all_context token count separately
    st.write("🧮 Data context token count:", all_context_token_count)

    # Step 8: Send through your session‐based conversation to get the main answer
    st.session_state.conversation.append({"role": "system", "content": context_message})
    st.session_state.conversation.append({"role": "user", "content": user_question})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=st.session_state.conversation,
        temperature=0.7,
    )
    gpt_answer = response.choices[0].message.content.strip()
    st.session_state.conversation.append({"role": "assistant", "content": gpt_answer})

    # Step 9: Generate recommended follow-up questions in a separate call
    recommendation_prompt = f"""
📌 VERIFIED FEEDBACK (Authoritative Corrections):
Use this section as the most accurate reference if it directly answers the question.
{feedback_text}

📘 BUSINESS GLOSSARY (Terms & Labels):
This glossary helps you interpret technical terms into business language when answering user questions.
reat 'Net open position' questions as the sum of VOLUME_BL for the REPORT_DATE unless the context clearly refers to monetary value, in which case use MKT_VAL_BL + {business_context_text}

🗂️ PREPROCESSED DATA (Official Trading Statistics):
This section contains structured trading data {category_context}.
This Data is CRITICAL: {all_context}

⚠️ IMPORTANT CONTEXT PRESERVATION:
When processing the data results, ALWAYS assume that any filtering conditions mentioned in the original user question (like specific books, traders, commodities, etc.) apply to ALL the data shown in the results, even if column names don't explicitly include these filters.

Original Question Context: "{user_question}"
- If the original question filtered by a specific book (e.g., GTUO), assume ALL data in the results is for that book
- If the original question filtered by commodity, trader, or other dimension, assume ALL data reflects those filters
- The SQL query engine has already applied these filters, so the data shown is the correct filtered subset


⚠️ INSTRUCTION (STRICTLY FOLLOW THESE STEPS — DO NOT IGNORE):
You are an expert Natural Language Query Parser specialized in Energy Trading and Risk Management (ETRM) queries.
CORE RESPONSIBILITIES:
1. Query Understanding
● Extract key components from natural language queries
● Identify temporal references and constraints
● Recognize trading-specific terminology
● Detect multiple intents within single queries
2. Parameter Extraction
● Identify and normalize:
○ Time periods and dates
○ Commodities and markets
○ Traders and counterparties
○ Metrics and measurements
○ Comparison requests
○ Aggregation levels
3. Query Intent Recognition Primary intents include:
● Position analysis
● Market exposure evaluation
● Risk assessment
● Performance measurement
● Compliance checking
● Strategic analysis

Based on the user's previous question: "{user_question}" and the answer that was provided, generate exactly 3 related follow-up questions that would be helpful and relevant. Format them as a numbered list (1, 2, 3).

DO NOT DEVIATE FROM THESE STEPS. ANSWERS MUST FOLLOW THIS EXACT LOGIC.

{prompt_Instr}
"""

    recommendation_messages = [
        {"role": "system", "content": recommendation_prompt},
        {"role": "user", "content": f"Previous question: {user_question}\nPrevious answer: {gpt_answer}\n\nSuggest 3 related follow-up questions try to give simple questions."}
    ]

    recommendation_response = client.chat.completions.create(
        model="gpt-4",
        messages=recommendation_messages,
        temperature=0.7,
    )
    recommended_questions = recommendation_response.choices[0].message.content.strip()

    # Keep history to last 5 messages
    if len(st.session_state.conversation) > 6:
        st.session_state.conversation = (
            [st.session_state.conversation[0]]
            + st.session_state.conversation[-4:]
        )

    # Step 10: Compute confidence and write output
    confidence_score = calculate_confidence_score(0.7, feedback_insights, file_names)
    st.session_state.confidence_score = confidence_score

    with open("query_output.txt", "w", encoding="utf-8") as f:
        f.write("🔹 Extracted Data:\n" + all_context)
        f.write("\n\n🔹 GPT Answer:\n" + gpt_answer)
        f.write("\n\n🔹 Recommended Questions:\n" + recommended_questions)

    # Store the recommended questions for display in the UI
    st.session_state.recommended_questions = recommended_questions
    
    # NEW: Log the conversation to a separate log file
    full_response = gpt_answer + "\n\n" + "You might also want to ask:\n" + recommended_questions
    
    # Return the main answer (the UI will display the recommendations separately)
    return full_response


def create_faiss_index(feedback_data):
    if not feedback_data:
        return None, []
    texts = [item.get("Query", "") for item in feedback_data if item.get("Query", "")]
    if not texts:
        return None, []
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform(texts).astype(np.float32).toarray()
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        return index, feedback_data
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None, []

def prepare_training_data():
    """
    Retrieves all training data pairs from the database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT query, incorrect_answer, correction FROM training_data")
    rows = cursor.fetchall()

    conn.close()

    # Convert to list of dictionaries
    training_pairs = []
    for row in rows:
        query, incorrect_answer, correction = row
        training_pairs.append({
            "query": query,
            "incorrect_answer": incorrect_answer,
            "correction": correction
        })

    return training_pairs

def save_training_data():
    """
    Saves the training data to a file for future fine-tuning.
    Returns the number of training pairs saved.
    """
    training_pairs = prepare_training_data()

    if not training_pairs:
        return 0

    # For file compatibility, you might still want to export to JSON
    # Or you could implement a different export method
    try:
        import json
        with open("energy_trading_training_data.json", "w") as f:
            json.dump(training_pairs, f, indent=4)
        return len(training_pairs)
    except Exception as e:
        print(f"Error saving training data: {e}")
        return 0

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

def calculate_confidence_score(base_score, feedback_insights, relevant_documents):
    confidence = base_score
    if feedback_insights:
        confidence -= 0.05 * len(feedback_insights)
    if relevant_documents:
        confidence += 0.1 * min(len(relevant_documents), 3)
    return max(0.1, min(0.99, confidence))

def retrieve_feedback_insights(query, faiss_index, feedback_data_indexed, top_k=3):
    if not faiss_index or not feedback_data_indexed:
        return []
    vectorizer = TfidfVectorizer()
    try:
        # Build the TF-IDF representation using the "Query" texts
        vectorizer.fit([item.get("Query", "") for item in feedback_data_indexed if item.get("Query", "")])
        query_vector = vectorizer.transform([query]).astype(np.float32).toarray()
        distances, indices = faiss_index.search(query_vector, top_k)
        insights = []
        for idx in indices[0]:
            if idx < len(feedback_data_indexed):
                feedback_item = feedback_data_indexed[idx]
                # Check for negative feedback by verifying if the stored value is 0.
                if feedback_item.get("User Feedback") == 0 and feedback_item.get("Feedback Comment", "").strip():
                    insights.append(f"Previous issue: {feedback_item.get('Query')} - {feedback_item.get('Feedback Comment')}")
        return insights
    except Exception as e:
        print(f"Error retrieving feedback insights: {e}")
        return []


def download_file_from_s3(s3_key, bucket_name):
    s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )
    try:
        file_obj = BytesIO()
        s3.download_fileobj(bucket_name, s3_key, file_obj)
        file_obj.seek(0)
        return file_obj
    except Exception as e:
        print(f"S3 download error: {e}")
        return None