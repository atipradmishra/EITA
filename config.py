from openai import OpenAI
import os
import streamlit as st

DB_NAME = "vector_chunks.db"
VALID_BUCKET = "etrm-etai-poc-chub"
REJECTED_BUCKET = "etai-rejected-files"

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")

client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])