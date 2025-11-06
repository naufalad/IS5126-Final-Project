import os
import json
import pandas as pd
import streamlit as st

# --- CONFIG PAGE ---
st.set_page_config(page_title="Data Browser", page_icon="", layout="wide")

# --- CUSTOM SPOTIFY STYLE ---
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

        /* Buttons */
        button[kind="primary"] {
            background-color: #1DB954 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            transition: all 0.3s ease-in-out;
        }
        button[kind="primary"]:hover {
            background-color: #1ed760 !important;
            transform: scale(1.02);
        }

        /* Inputs */
        input, select, textarea {
            background-color: #1e1e1e !important;
            color: white !important;
            border-radius: 6px !important;
        }

        /* DataFrame container */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            background-color: #181818;
        }

        /* Metrics */
        .metric-container {
            margin-top: 15px;
            margin-bottom: 20px;
        }
        div[data-testid="stMetricValue"] {
            color: #1DB954 !important;
        }

        /* Title SVG and layout */
        .svg-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }

        .stCaption, .stMarkdown, .stText, .stTextInput {
            color: #B3B3B3 !important;
        }
    </style>
""", unsafe_allow_html=True)

# HEADER  ICON SVG (Spotify Green)
st.title("Data Browser")

#  LOAD LOCAL DATA 
def load_local_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_dir = os.path.dirname(current_dir)
        data_path = os.path.join(os.path.dirname(ui_dir), "data", "email_features.json")
        st.title(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f" Failed to load local data: {e}")
        return []

# --- SIDEBAR FILTER ---
st.sidebar.write("### 🔍 Filter")
search_term = st.sidebar.text_input("Search text", placeholder="Type keyword...")
event_filter = st.sidebar.text_input("Event type equals", placeholder="e.g. meeting")
page_size = st.sidebar.selectbox("Page size", options=[10, 20, 50, 100], index=1)

local_data = load_local_data()

# --- DASHBOARD ---
if local_data:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(local_data))
    col2.metric("Has Datetime", sum(1 for i in local_data if i.get("scheduled_datetime")))
    col3.metric("Has Location", sum(1 for i in local_data if i.get("location")))
    col4.metric("Action Required", sum(1 for i in local_data if i.get("action_required") and i.get("action_required") != "none"))

    st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

   
# --- FILTER DATA ---
filtered = local_data

# 🔍 Search filter (subject + body)
if search_term:
    filtered = [
        i for i in filtered
        if search_term.lower() in str(i.get("subject", "")).lower()
        or search_term.lower() in str(i.get("body", "")).lower()
    ]

#  Event type dropdown filter (sorted ascending)
event_types = list({
    (i.get("event_type") if i.get("event_type") is not None else "unknown")
    for i in local_data
})
event_types = sorted(event_types)  # sort A-Z

event_filter = st.sidebar.selectbox(
    "Category: Event Type",
    ["All"] + event_types,
    index=0,
    key="event_filter"
)

if event_filter != "All":
    filtered = [
        i for i in filtered
        if (i.get("event_type") if i.get("event_type") is not None else "unknown") == event_filter
    ]

# ⚡ Urgency level dropdown filter
urgency_levels = list({
    (i.get("urgency_level") if i.get("urgency_level") is not None else "unknown")
    for i in local_data
})
urgency_levels = sorted(urgency_levels)

urgency_filter = st.sidebar.selectbox(
    "Urgency Level",
    ["All"] + urgency_levels,
    index=0,
    key="urgency_filter"
)

if urgency_filter != "All":
    filtered = [
        i for i in filtered
        if (i.get("urgency_level") if i.get("urgency_level") is not None else "unknown") == urgency_filter
    ]


# 📄 Pagination setup
total_filtered = len(filtered)
total_pages = max(1, (total_filtered + page_size - 1) // page_size)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, label_visibility="collapsed", key="page_number_input")

start_idx, end_idx = (page - 1) * page_size, (page - 1) * page_size + page_size
page_data = filtered[start_idx:end_idx]

# 🧾 Summary caption
st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_filtered)} of {total_filtered} filtered records")

if page_data:
    df = pd.DataFrame(page_data)

    # Potong teks panjang untuk preview
    if "email_text" in df.columns:
        df["email_text"] = df["email_text"].astype(str).str.slice(0, 120) + "..."

    # Tambahkan index kolom yang lebih informatif
    df.index = [f"{i}" for i in range(len(df))]

    # --- Custom Styling ---
    st.markdown("""
        <style>
            /* Custom dark Spotify-like table */
            .stDataFrame {
                background-color: #121212;
                color: #EAEAEA;
                border-radius: 12px;
                border: 1px solid #333;
                font-family: 'Inter', sans-serif;
            }
            .stDataFrame th {
                background-color: #1DB954 !important;
                color: black !important;
                font-weight: bold !important;
            }
            .stDataFrame td {
                background-color: #181818;
                border-bottom: 1px solid #333 !important;
            }
            .stDataFrame tr:hover td {
                background-color: #282828 !important;
                color: #FFFFFF !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Tampilkan tabel dengan fitur baru Streamlit ---
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "subject": st.column_config.TextColumn("Subject ", help="Email subject line"),
            "sender": st.column_config.TextColumn("From ", help="Email sender"),
            "event_type": st.column_config.TextColumn("Event Type ", help="Predicted event type"),
            "email_text": st.column_config.TextColumn("Email Preview ", help="First 120 chars of email body"),
        },
        hide_index=False,
    )

    st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

    # --- DETAIL VIEW ---
    with st.expander("📄 View Details", expanded=False):
        detail_idx = st.number_input("Select Index (global)", min_value=0, max_value=len(local_data)-1, value=0)
        if st.button("🔍 View Details"):
            item = local_data[int(detail_idx)]
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Email Content**")
                st.text_area("", item.get("email_text", ""), height=220, disabled=True)
            with c2:
                st.write("**Extracted Features**")
                st.json({k: v for k, v in item.items() if k != "email_text"})
else:
    st.info("No matching records found.")



