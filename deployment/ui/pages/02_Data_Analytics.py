import os
import json
import pandas as pd
import altair as alt
import streamlit as st


st.set_page_config(page_title="Data Analytics", page_icon="📊", layout="wide")
st.title("📊 Data Analytics")
st.caption("Statistical analysis and visualization of email data")


def load_local_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
        ui_dir = os.path.dirname(current_dir)  # .../ui
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

st.subheader("Basic Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Emails", len(df))
with col2:
    st.metric("Event Types", df["event_type"].nunique(dropna=True))
with col3:
    st.metric("Urgency Levels", df["urgency_level"].nunique(dropna=True))
with col4:
    st.metric("Action Types", df["action_required"].nunique(dropna=True))

st.divider()
st.subheader("Data Distribution")

colA, colB = st.columns(2)
with colA:
    if "event_type" in df.columns:
        s = df["event_type"].fillna("(none)").value_counts().reset_index()
        s.columns = ["Event Type", "Count"]
        chart = alt.Chart(s).mark_bar().encode(
            x="Count:Q",
            y=alt.Y("Event Type:N", sort="-x"),
            color=alt.Color("Event Type:N", legend=None)
        ).properties(title="Event Type Distribution", height=320)
        st.altair_chart(chart, use_container_width=True)

with colB:
    if "urgency_level" in df.columns:
        s = df["urgency_level"].fillna("(none)").value_counts().reset_index()
        s.columns = ["Urgency Level", "Count"]
        chart = alt.Chart(s).mark_bar().encode(
            x="Count:Q",
            y=alt.Y("Urgency Level:N", sort="-x"),
            color=alt.Color("Urgency Level:N", legend=None)
        ).properties(title="Urgency Level Distribution", height=320)
        st.altair_chart(chart, use_container_width=True)


