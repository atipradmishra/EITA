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
from sklearn.feature_extraction.text import TfidfVectorizer
from DbUtils.DbOperations import load_feedback_data, load_business_context
from config import aws_access_key, aws_secret_key, client, DB_NAME, VALID_BUCKET, REJECTED_BUCKET

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

def query_sqlite_json_with_openai(user_question, category=None):
    # Load business dictionary
    business_dict = load_business_context()

    # Step 1: Load Feedback and FAISS Index
    feedback_data = load_feedback_data()
    faiss_index, feedback_data_indexed = create_faiss_index(feedback_data)
    feedback_insights = retrieve_feedback_insights(user_question, faiss_index, feedback_data_indexed)

    # Step 2: Query SQLite data
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if category:
        if category == "CO2":
            cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_CO2%'")
        elif category == "Natural Gas":
            cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND (file_name LIKE 'NOP_GAS%' OR file_name LIKE 'NOP_NG%')")
        elif category == "Power":
            cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND (file_name LIKE 'NOP_POWER%' OR file_name LIKE 'NOP_PW%')")
    else:
        cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "⚠️ No processed JSON data found for the selected category."
    
    #Prompt Instruction
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if category:
        if category == "CO2":
            cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'CO2 AGENT'")
        elif category == "Natural Gas":
            cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'NG AGENT'")
        elif category == "Power":
            cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'PW AGENT'")
    else:
        cursor.execute("SELECT prompt FROM agent_detail WHERE name= 'PW AGENT'")

    prompt_Instr = cursor.fetchall()
    print(f"Prompt Instruction, {prompt_Instr}")

    conn.close()

    if not prompt_Instr:
        return "⚠️ No Prompt Instruction Found."
    
    # Step 3: Build context from retrieved JSON
    all_context = ""
    file_names = []
    for file_name, json_text in rows:
        file_names.append(file_name)
        try:
            json_data = json.loads(json_text)
            summary = json.dumps(json_data, separators=(',', ':'))[:6000]
            all_context += f"\n---\n📄 File: {file_name}\n{summary}"
        except Exception as e:
            all_context += f"\n---\n📄 File: {file_name}\n⚠️ Error reading JSON: {e}"

    # Step 4: Prepare business glossary
    business_context_text = "\n".join(
        f"{key}: {value}" if isinstance(value, str)
        else f"{key}: {', '.join(f'{k} = {v}' for k, v in value.items())}"
        for key, value in business_dict.items()
    )

    # Step 5: Compose system prompt
    feedback_text = ""
    
    if feedback_insights:
        feedback_text = "\n\nBased on feedback from similar queries, please be aware of these issues:\n"
        feedback_text += "\n".join(f"- {insight}" for insight in feedback_insights)
        print(feedback_text)
    category_context = f"for the {category} category" if category and category != "All" else ""

    context_message = f"""
    📌 VERIFIED FEEDBACK (Authoritative Corrections):
    Use this section as the most accurate reference if it directly answers the question.
    {feedback_text}

    📘 BUSINESS GLOSSARY (Terms & Labels):
    This glossary helps you interpret technical terms into business language when answering user questions.
    {business_context_text}

    🗂️ PREPROCESSED DATA (Official Trading Statistics):
    This section contains structured trading data for multiple files {category_context}.
    {all_context}

    ⚠️ INSTRUCTION (STRICTLY FOLLOW THESE STEPS — DO NOT IGNORE):
    1. If the VERIFIED FEEDBACK section directly answers the question, use it as the ONLY source.
    2. If the feedback is not sufficient, use PREPROCESSED DATA and BUSINESS GLOSSARY to answer.
    3. Always translate technical codes (e.g., ELCE, NG, TGROUP1) into business-friendly terms using the glossary.
    4. Provide a clear, professional, and concise answer suitable for business users.
    5. End your answer with 2–3 relevant follow-up questions based on the question and data.

    DO NOT DEVIATE FROM THESE STEPS. ANSWERS MUST FOLLOW THIS EXACT LOGIC.
    {prompt_Instr}
    """
    
    # Step 6: Construct the conversation and call OpenAI
    st.session_state.conversation.append({"role": "system", "content": context_message})
    st.session_state.conversation.append({"role": "user", "content": user_question})

    for idx, message in enumerate(st.session_state.conversation):
        print(f"\n🔹 Message {idx+1} ({message['role']}):\n{message['content']}\n{'-'*60}")

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=st.session_state.conversation,
        temperature=0.7
    )

    gpt_answer = response.choices[0].message.content.strip()
    st.session_state.conversation.append({"role": "assistant", "content": gpt_answer})

    if len(st.session_state.conversation) > 6:
        st.session_state.conversation = [st.session_state.conversation[0]] + st.session_state.conversation[-4:]

    confidence_score = calculate_confidence_score(0.7, feedback_insights, file_names)
    st.session_state.confidence_score = confidence_score

    with open("query_output.txt", "w", encoding="utf-8") as f:
        f.write("🔹 Extracted Data:\n" + all_context)
        f.write("\n\n🔹 GPT Answer:\n" + gpt_answer)

    return gpt_answer

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


