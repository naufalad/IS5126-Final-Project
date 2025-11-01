import os
import json
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Data Browser", page_icon="📋", layout="wide")
st.title("📋 Data Browser")
st.caption("Browse and analyze email feature data")


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


st.sidebar.header("Filters")
search_term = st.sidebar.text_input("Search", placeholder="Search email content...")
event_filter = st.sidebar.text_input("Event type equals", value="")
page_size = st.sidebar.selectbox("Page size", options=[10, 20, 50, 100], index=1)

local_data = load_local_data()

if local_data:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(local_data))
    with col2:
        has_datetime = sum(1 for i in local_data if i.get("scheduled_datetime"))
        st.metric("Has Complete Datetime", has_datetime)
    with col3:
        has_location = sum(1 for i in local_data if i.get("location"))
        st.metric("Has Location", has_location)
    with col4:
        has_action = sum(1 for i in local_data if i.get("action_required") and i.get("action_required") != "none")
        st.metric("Action Required", has_action)

    st.divider()

    filtered = local_data
    if search_term:
        filtered = [i for i in filtered if search_term.lower() in str(i.get("email_text", "")).lower()]
    if event_filter:
        filtered = [i for i in filtered if i.get("event_type") == event_filter]

    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_data = filtered[start_idx:end_idx]

    st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_filtered)} / Total {total_filtered} records")

    if page_data:
        df = pd.DataFrame(page_data)
        if "email_text" in df.columns:
            df["email_text"] = df["email_text"].astype(str).str.slice(0, 120) + "..."
        st.dataframe(df, use_container_width=True, height=420)

        st.divider()
        st.subheader("🔍 Detail View")
        detail_idx = st.number_input("Select Index (global)", min_value=0, max_value=len(local_data)-1, value=0)
        if st.button("View Details"):
            item = local_data[int(detail_idx)]
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Email Content**")
                st.text_area("", item.get("email_text", ""), height=240, disabled=True)
            with c2:
                st.write("**Extracted Features**")
                st.json({k: v for k, v in item.items() if k != "email_text"})
    else:
        st.info("No matching records found")
else:
    st.warning("Local data file does not exist or is empty")


