import sqlite3
import streamlit as st
import bcrypt
from config import DB_NAME


def register_user(username, password):
    """Registers a new user with hashed password."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticates user with hashed password verification."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_password = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()
