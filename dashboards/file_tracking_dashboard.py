import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import io
from datetime import datetime
from config import DB_NAME


# Custom CSS for better UI
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #0f2b46;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #0f2b46;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f1f3f5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f2b46;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Database connection function
def get_data_from_db():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_NAME)
        
        # Query to fetch all records from file_tracking table
        query = "SELECT file_id, file_name, report_date, uploaded_at, is_processed FROM data_files_metadata"
        
        # Load the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return pd.DataFrame()

# Function to get metadata from database
def get_existing_metadata(dataset_type: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT json_data FROM metadata_files 
        WHERE filename LIKE ? 
        ORDER BY uploaded_at DESC LIMIT 1
    """, (f"%{dataset_type.upper()}%",))
    row = cursor.fetchone()
    conn.close()
    if row:
        json_data = row[0]
        df = pd.read_json(io.StringIO(json_data), orient="records")
        return df
    return None

# Create a custom metric component
def metric_card(title, value, icon, color):
    html = f"""
    <div class="metric-card">
        <div style="font-size: 24px; color: {color};">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """
    return html

# Create a styled section
def styled_section(title):
    return f"""
    <div class="section-card">
        <h3>{title}</h3>
    """

# Main function
def render_file_tracking_dashboard():
    # Apply custom CSS
    apply_custom_css()
    
    # Sidebar branding
    with st.sidebar:
        # Only showing Dashboard navigation option
        page = "Dashboard"  # Fixed to always show Dashboard
    
    # Load data
    df = get_data_from_db()
    
    if df.empty:
        st.warning("No data available. Please check your database connection.")
    
    # Extract file categories
    df['category'] = df['file_name'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'OTHER')
    
    # Main dashboard page
    if page == "Dashboard":
        # Header with background
        st.markdown('<div style="background-color:#0f2b46; padding:15px; border-radius:10px;">'
                   '<h1 style="color:white; text-align:center;">Data File Tracking Dashboard</h1>'
                   '<p style="color:#adb5bd; text-align:center;">Real-time monitoring of processed files</p>'
                   '</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # First row - Summary metrics
        st.markdown("<h2>Dataset Overview</h2>", unsafe_allow_html=True)
        
        # Total datasets
        total_datasets = len(df)
        
        # Get category counts
        categories = df['category'].value_counts().to_dict()
        
        # Ensure all categories are represented (with zero counts if none)
        for cat in ['CO2', 'PW', 'NG']:
            if cat not in categories:
                categories[cat] = 0
        
        # Display metrics in a more appealing way
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(metric_card("Total Datasets", total_datasets, "📊", "#1e88e5"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(metric_card("CO2 Files", categories.get('CO2', 0), "🔵", "#43a047"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(metric_card("PW Files", categories.get('PW', 0), "🟣", "#7b1fa2"), unsafe_allow_html=True)
        
        with col4:
            st.markdown(metric_card("NG Files", categories.get('NG', 0), "🟠", "#ff9800"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-card">
                <h3>File Distribution by Category</h3>
            """, unsafe_allow_html=True)
            
            # Category distribution chart
            colors = {'CO2': '#43a047', 'PW': '#7b1fa2', 'NG': '#ff9800', 'OTHER': '#607d8b'}
            category_colors = [colors.get(cat, '#607d8b') for cat in categories.keys()]
            
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                color_discrete_sequence=category_colors,
                hole=0.6
            )
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation='h', yanchor='bottom', y=-0.3),
                annotations=[dict(text='Files by<br>Category', x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="section-card">
                <h3>Processing Status</h3>
            """, unsafe_allow_html=True)
            
            # Processing status chart
            processed = df['is_processed'].sum()
            not_processed = len(df) - processed
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = processed / len(df) * 100 if len(df) > 0 else 0,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Processed Files (%)", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#0f2b46"},
                    'steps': [
                        {'range': [0, 50], 'color': '#ff9800'},
                        {'range': [50, 75], 'color': '#ffeb3b'},
                        {'range': [75, 100], 'color': '#4caf50'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # NEW SECTION: Metadata Files below Dataset Overview
        st.markdown("<h2>Metadata Files</h2>", unsafe_allow_html=True)
        
        # Create tabs for different dataset types
        meta_tabs = st.tabs(["CO2", "PW", "NG"])
        
        # Tab for CO2 metadata
        with meta_tabs[0]:
            co2_metadata = get_existing_metadata("CO2")
            if co2_metadata is not None:
                st.write("### CO2 Metadata Schema")
                st.dataframe(co2_metadata, use_container_width=True)
            else:
                st.info("No metadata available for CO2 datasets.")
        
        # Tab for PW metadata
        with meta_tabs[1]:
            pw_metadata = get_existing_metadata("PW")
            if pw_metadata is not None:
                st.write("### PW Metadata Schema")
                st.dataframe(pw_metadata, use_container_width=True)
            else:
                st.info("No metadata available for PW datasets.")
        
        # Tab for NG metadata
        with meta_tabs[2]:
            ng_metadata = get_existing_metadata("NG")
            if ng_metadata is not None:
                st.write("### NG Metadata Schema")
                st.dataframe(ng_metadata, use_container_width=True)
            else:
                st.info("No metadata available for NG datasets.")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # File details section with tabs
        st.markdown("""
        <div class="section-card">
            <h3>File Details by Category</h3>
        """, unsafe_allow_html=True)
        
        # Category tabs
        tabs = st.tabs(["All"] + list(categories.keys()))
        
        # All tab
        with tabs[0]:
            display_df = df[['file_name', 'report_date', 'uploaded_at', 'is_processed']].copy()
            display_df['uploaded_at'] = pd.to_datetime(display_df['uploaded_at'], errors='coerce')
            display_df['uploaded_at'] = display_df['uploaded_at'].dt.strftime('%Y-%m-%d')
            display_df['is_processed'] = display_df['is_processed'].apply(lambda x: "✅ Yes" if x == 1 else "❌ No")
            display_df.columns = ['File Name','Report Date', 'Date Uploaded', 'Processed']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Category specific tabs
        for i, category in enumerate(categories.keys(), 1):
            with tabs[i]:
                category_df = df[df['category'] == category].copy()
                if not category_df.empty:
                    # after
                    display_df = category_df[['file_name', 'report_date' , 'uploaded_at', 'is_processed']].copy()
                    display_df['uploaded_at'] = pd.to_datetime(display_df['uploaded_at'], errors='coerce')
                    display_df['uploaded_at'] = display_df['uploaded_at'].dt.strftime('%Y-%m-%d')
                    display_df['is_processed'] = display_df['is_processed'].apply(lambda x: "✅ Yes" if x == 1 else "❌ No")
                    display_df.columns = ['File Name', 'Report Date', 'Date Uploaded', 'Processed']
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info(f"No files found for category: {category}")
        
        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================
        # Business Domain Context Files Section
        # ==========================
        st.markdown("""
        <div class="section-card">
            <h3>Business Domain Dictionary Files</h3>
        """, unsafe_allow_html=True)

        # Connect and fetch distinct files from Business_Context
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT source_file FROM Business_Context ORDER BY source_file
            """)
            files = cursor.fetchall()
            conn.close()

            if files:
                for file in files:
                    st.markdown(f"📘 **{file[0]}**")
            else:
                st.info("No Business Domain Dictionary files have been added yet.")
        except Exception as e:
            st.error(f"Error fetching business context files: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Moved: Processing Timeline as the last section
        st.markdown("""
        <div class="section-card">
            <h3>Processing Timeline</h3>
        """, unsafe_allow_html=True)
        
        # Convert uploaded_at to datetime
        df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])
        
        # Group by date and category, count files
        timeline_df = df.groupby([pd.Grouper(key='uploaded_at', freq='D'), 'category']).size().reset_index(name='count')
        
        # Create timeline chart
        fig = px.line(
            timeline_df, 
            x='uploaded_at', 
            y='count', 
            color='category',
            markers=True,
            color_discrete_map={'CO2': '#43a047', 'PW': '#7b1fa2', 'NG': '#ff9800', 'OTHER': '#607d8b'}
        )
        fig.update_layout(
            xaxis_title="Processing Date", 
            yaxis_title="Number of Files",
            legend_title="Category",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=20, b=40, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
