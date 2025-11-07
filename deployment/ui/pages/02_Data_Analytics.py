import os
import json
import pandas as pd
import altair as alt
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Data Analytics", page_icon="", layout="wide")

# --- CUSTOM DARK THEME ---
st.markdown("""
<style>
/* Global background */
.main {
    background-color: #121212;
    color: #FFFFFF;
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    padding: 20px;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #FFFFFF;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #000000;
    color: white;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    color: #1DB954 !important;
}

/* Charts background */
.vega-embed {
    background-color: #181818 !important;
    border-radius: 10px;
}

/* Divider */
hr {
    border: 1px solid #333;
}

/* Text & caption */
.stText, .stCaption, .stMarkdown {
    color: #B3B3B3 !important;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---

st.title("Data Analytics")

st.caption("Statistical analysis and visualization of email data")

# --- LOAD DATA ---
def load_local_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_dir = os.path.dirname(current_dir)
        data_path = os.path.join(os.path.dirname(ui_dir), "data", "email_features.json")
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load local data: {e}")
        return []

data = load_local_data()
if not data:
    st.warning("No data available")
    st.stop()

df = pd.DataFrame(data)

# --- METRICS ---
st.subheader("Basic Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric(" Total Emails", len(df))
col2.metric(" Category", df["category"].nunique(dropna=True))
col3.metric(" Urgency Levels", df["urgency_level"].nunique(dropna=True))
col4.metric(" Action Types", df["action_required"].nunique(dropna=True))

st.markdown("<hr>", unsafe_allow_html=True)

# --- DISTRIBUTION CHARTS ---
st.subheader("Data Distribution")
colA, colB = st.columns(2)

with colA:
    if "category" in df.columns:
        s = df["category"].fillna("(none)").value_counts().reset_index()
        s.columns = ["Category", "Count"]
        chart = alt.Chart(s).mark_bar().encode(
            x="Count:Q",
            y=alt.Y("Category:N", sort="-x"),
            color=alt.Color("Category:N")  # warna asli kategori
        ).properties(title="Event Type Distribution", height=320)
        st.altair_chart(chart, use_container_width=True)

with colB:
    if "urgency_level" in df.columns:
        s = df["urgency_level"].fillna("(none)").value_counts().reset_index()
        s.columns = ["Urgency Level", "Count"]
        chart = alt.Chart(s).mark_bar().encode(
            x="Count:Q",
            y=alt.Y("Urgency Level:N", sort="-x"),
            color=alt.Color("Urgency Level:N")  # warna asli kategori
        ).properties(title="Urgency Level Distribution", height=320)
        st.altair_chart(chart, use_container_width=True)
