import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re
import requests
from dotenv import load_dotenv


st.set_page_config(page_title="Email Prediction and Extraction", page_icon="📥", layout="wide")
st.title("Email Prediction and Extraction")
st.caption("Show prepared email, or fill in. Predict category and extract features using backend API.")
# --- DARK THEME ---
st.markdown("""
<style>Email Prediction & Extraction

/* Background & fonts */
.main { background-color: #121212; color: #FFFFFF; font-family: 'Inter', sans-serif; padding: 20px; }
h1,h2,h3,h4,h5 { color:#FFFFFF; font-weight:600; }
.stText, .stCaption, .stMarkdown { color: #B3B3B3 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #000000; color:white; }

/* Buttons */
button[kind="primary"] { background-color:#1DB954 !important; color:white !important; border-radius:6px !important; border:none !important; }
button[kind="primary"]:hover { background-color:#1ed760 !important; }

/* Textareas & inputs */
textarea, input, select { background-color: #1e1e1e !important; color: white !important; border-radius:6px !important; }

/* Divider */
hr { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# Backend API configuration
load_dotenv()  # Load .env file if it exists

BACKEND_URL = os.getenv("BACKEND_API", "http://127.0.0.1:8000")

def load_local_data(show_debug=False):
    """Load email cases from JSON file - always reloads fresh data"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
        ui_dir = os.path.dirname(current_dir)  # .../ui
        data_path = os.path.join(os.path.dirname(ui_dir), "data", "email_cases.json")
        
        # Debug information
        if show_debug:
            st.write("🔍 **Debug Info:**")
            st.write(f"- Current file: `{__file__}`")
            st.write(f"- Current dir: `{current_dir}`")
            st.write(f"- UI dir: `{ui_dir}`")
            st.write(f"- Data path: `{data_path}`")
            st.write(f"- File exists: {os.path.exists(data_path)}")
            if os.path.exists(data_path):
                st.write(f"- File size: {os.path.getsize(data_path)} bytes")
                st.write(f"- Last modified: {os.path.getmtime(data_path)}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if show_debug:
            st.write(f"- Loaded {len(data)} email(s)")
            st.write(f"- Email subjects: {[e.get('subject', 'N/A')[:50] for e in data]}")
        
        return data
    except Exception as e:
        error_msg = f"Failed to load local data: {e}"
        st.error(error_msg)
        import traceback
        if show_debug:
            st.code(traceback.format_exc())
        return []


def calendar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
    ui_dir = os.path.dirname(current_dir)  # .../ui
    return os.path.join(os.path.dirname(ui_dir), "data", "calendar", "events.json")

def create_openai_client():
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        return None
    try:
        return OpenAI(api_key=api)
    except Exception:
        return None

def display_spotify_results(func_result: dict):
    """Display Spotify results in a beautiful UI.
    Supports two modes:
    - Single Agent: Direct links to Spotify (with play buttons)
    - Multi Agent: "Guess You Like" recommendations (options only, no links)
    """
    data = func_result.get("data", {})
    songs = data.get("songs", [])
    mode = data.get("mode", "direct_links")  # "direct_links" or "recommendations"
    
    if not songs:
        st.warning("⚠️ No Spotify songs found")
        return
    
    # Different header based on mode
    if mode == "recommendations":
        st.success(f"🎵 猜你喜欢 - {len(songs)} 首推荐歌曲")
        st.caption("💡 基于您的邮件内容，为您推荐以下歌曲")
    else:
        st.success(f"🎵 Found {len(songs)} Spotify song(s)!")
    
    # Spotify-themed styling
    st.markdown("""
    <style>
    .spotify-card {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .spotify-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(29, 185, 84, 0.4);
    }
    .spotify-song-title {
        color: #121212;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .spotify-artist {
        color: #1a1a1a;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .recommendation-reason {
        color: #1a1a1a;
        font-size: 13px;
        font-style: italic;
        margin-top: 5px;
        padding: 8px;
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display each song
    for i, song in enumerate(songs, 1):
        song_name = song.get("song", "Unknown Song")
        artist = song.get("artist", "Unknown Artist")
        spotify_url = song.get("spotify_url", "")
        album = song.get("album", "")
        release_date = song.get("release_date", "")
        reason = song.get("reason", "")  # For recommendations mode
        
        # Create card
        with st.container():
            if mode == "recommendations":
                # Multi Agent: Recommendations mode (no links, just options)
                st.markdown(f"### 🎵 {song_name}")
                st.markdown(f"**👤 Artist:** {artist}")
                if reason:
                    st.markdown(f'<div class="recommendation-reason">💡 {reason}</div>', unsafe_allow_html=True)
            else:
                # Single Agent: Direct links mode
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 🎵 {song_name}")
                    st.markdown(f"**👤 Artist:** {artist}")
                    
                    if album:
                        st.markdown(f"**💿 Album:** {album}")
                    if release_date:
                        st.markdown(f"**📅 Released:** {release_date}")
                
                with col2:
                    if spotify_url:
                        # Use link button for better UX
                        st.link_button(
                            "🎵 Play on Spotify",
                            spotify_url,
                            use_container_width=True,
                            type="primary"
                        )
        
        st.divider()


def display_attraction_results(func_result_or_data: dict, auto_open: bool = False):
    """Display attraction discovery results with Google Maps links.
    Supports two categories: direct_match and recommendations.
    Accepts either the full function_result dict (with data field) or a data dict.
    """
    # Normalize input
    data = func_result_or_data.get("data", func_result_or_data)
    
    # Support both old format (attractions) and new format (direct_match + recommendations)
    if "attractions" in data:
        # Old format: single list
        direct_match = data.get("attractions", [])
        recommendations = []
    else:
        # New format: categorized
        direct_match = data.get("direct_match", [])
        recommendations = data.get("recommendations", [])

    total_count = len(direct_match) + len(recommendations)
    if total_count == 0:
        st.warning("⚠️ No attractions found")
        return

    st.success(f"🎭 Found {total_count} attraction(s)!")

    # Styling similar to Spotify cards
    st.markdown(
        """
        <style>
        .attraction-card {
            background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        .attraction-card:hover { transform: translateY(-2px); }
        .attraction-title { color: #121212; font-size: 18px; font-weight: bold; margin-bottom: 5px; }
        .attraction-meta { color: #1a1a1a; font-size: 14px; margin-bottom: 10px; }
        .section-header { 
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            margin: 20px 0 10px 0;
            font-weight: bold;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    opened = False
    
    def render_attraction(attr: dict, auto_open_first: bool = False):
        """Helper function to render a single attraction"""
        nonlocal opened
        name = attr.get("name", "Unknown Attraction")
        location = attr.get("location", "")
        a_type = attr.get("type", "general")
        desc = attr.get("description", "")
        map_link = attr.get("map_link", "")
        fun_fact = attr.get("fun_fact", "")

        with st.container():
            st.markdown(f"<div class='attraction-card'><div class='attraction-title'>📍 {name}</div></div>", unsafe_allow_html=True)
            if location:
                st.markdown(f"**📌 Location:** {location}")
            if a_type:
                st.markdown(f"**🏷️ Type:** {a_type}")
            if desc:
                st.markdown(f"**📝 Description:** {desc}")
            if fun_fact:
                st.markdown(f"**✨ Fun fact:** {fun_fact}")

            if map_link:
                st.link_button("🗺️ Open in Google Maps", map_link, type="primary")
                if auto_open_first and auto_open and not opened:
                    # Open only the first link automatically to avoid popup blocks
                    st.markdown(
                        f"<script>window.open('{map_link}', '_blank');</script>",
                        unsafe_allow_html=True,
                    )
                    opened = True
        st.divider()

    # Display direct match attractions (mentioned location)
    if direct_match:
        st.markdown("<div class='section-header'>📍 Direct Match - Mentioned Location</div>", unsafe_allow_html=True)
        for i, attr in enumerate(direct_match):
            render_attraction(attr, auto_open_first=(i == 0))
    
    # Display recommendations (if requested)
    if recommendations:
        st.markdown("<div class='section-header'>🌟 Recommended Attractions</div>", unsafe_allow_html=True)
        for i, attr in enumerate(recommendations):
            render_attraction(attr, auto_open_first=(i == 0 and len(direct_match) == 0))


def call_function_call_api(email_data: dict) -> dict:
    """Call backend API /function_call endpoint to process email and create calendar event"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/function_call",
            json=email_data,
            timeout=60  # Increased timeout for Single Agent processing
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend may be processing - please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Failed to call API: {str(e)}")
        return None

def call_multi_agent_api(email_data: dict) -> dict:
    """Call backend API /create endpoint to process email and create calendar event"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/create",
            json=email_data,
            timeout=30  # Longer timeout for LLM processing
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend may be processing - please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Failed to call API: {str(e)}")
        return None

def call_predict_api(email_data: dict) -> dict:
    """Call backend API /predict endpoint to process email and create calendar event"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json=email_data,
            timeout=30  # Longer timeout for LLM processing
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend may be processing - please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Failed to call API: {str(e)}")
        return None

# Add refresh button and debug toggle
col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
with col_header1:
    st.empty()  # Spacer
with col_header2:
    show_debug = st.checkbox("🐛 Debug Mode", help="Show detailed loading information")
with col_header3:
    if st.button("🔄 Refresh", help="Reload email cases from file"):
        st.rerun()

# Load data with optional debug info
data = load_local_data(show_debug=show_debug)
if not data:
    st.warning("No data available")
    st.stop()

# Display data count for debugging
st.caption(f"📧 Loaded {len(data)} email(s) from file")

# Optional: auto-open Google Maps links when attractions are displayed
auto_open_maps = st.checkbox("🚀 Auto-open Google Maps links", value=False, help="Open the first attraction map link automatically in a new tab")
st.session_state['auto_open_maps'] = auto_open_maps

colA, colB = st.columns([1, 3])
with colA:
    idx_display = st.number_input("Email No.", min_value=0, max_value=len(data)-1, value=0, step=1)
    idx = idx_display
    model_choice = st.radio("Choose classification model:", ['BERT', 'XGBoost', 'RF'], horizontal=True)
    model_choice = ['BERT', 'XGBoost', 'RF'].index(model_choice) + 1
    if st.button("Predict Email Category", type="primary"):
        st.session_state['received_email_index'] = idx_display  # Store 1-based index
        subject = st.session_state.get("subject_area", "")
        body = st.session_state.get("body_area", "")

        # Prepare data for backend API
        email_payload = {
            "subject": subject,
            "body": body,
            "model": model_choice
        }
    
        with st.spinner("Processing email..."):
            # Call backend API
            api_response = call_predict_api(email_payload)
            
            if api_response:
                if api_response.get("success"):
                    st.success("Received email prediction")
                    category = api_response.get("prediction")[0]
                    st.info(f"📧 Predicted Category: **{category}**")
                    st.session_state['predicted_category'] = category
                    explanation = api_response.get("explanation", "No explanation provided")
                    st.markdown("**Explanation:**")
                    st.info(explanation)
                else:
                    st.warning(f"Email received but classification failed: {api_response.get('message', 'Unknown error')}")
            else:
                st.error("Failed to process email via backend API")
    model_choice = st.radio("Choose model agent:", ['Single Agent', 'Multi Agent'], horizontal=True)
    st.info("""ℹ️ Single Agent uses one LLM agent to extract features and create calendar event. It uses function calling to extract features then whether create the event, show spotify links, or give tourist attractions.

• **For Spotify**: Simple extraction with direct links to Spotify (click to play)
• **For locations**: Only extracts attractions for the explicitly mentioned location.""")
    st.info("""ℹ️ Multi Agent uses multiple specialized agents for better accuracy. For calendar events, it uses specialized agents for classification, scheduling, and formatting.

• **For Spotify**: Uses two agents to analyze email content and generate "Guess You Like" personalized song recommendations (options only, no direct links)
• **For locations**: It uses two agents: one for extraction and one for analyzing recommendation needs, enabling intelligent attraction discovery with recommendations.""")

    if st.button("Extract and Manage Email Features", type="primary"):
        st.session_state['received_email_index'] = idx_display
        subject = st.session_state.get("subject_area", "")
        body = st.session_state.get("body_area", "")
        
        with st.spinner("Processing email..."):
            # Prepare data for backend API
            email_payload = {
                "subject": subject,
                "body": body,
                "category": st.session_state.get('predicted_category', 'general')
            }
            
            # Call backend API
            if(model_choice == 'Single Agent'):
                api_response = call_function_call_api(email_payload)
            else:
                api_response = call_multi_agent_api(email_payload)
            
            # Handle API response
            if api_response:
                if api_response.get("success"):
                    # Handle Multi Agent response (different structure)
                    if model_choice == "Multi Agent":
                        data = api_response.get("data", {})
                        st.success("✅ Multi-Agent processing completed")
                        
                        # Display calendar event if available
                        if data.get("calendar_event"):
                            calendar_data = data["calendar_event"]
                            if calendar_data and calendar_data.get("calendar_event"):
                                st.success("📅 Calendar event created successfully")
                                calendar_event = calendar_data.get("calendar_event", {})
                                if calendar_event.get("title"):
                                    st.info(f"📅 Event: {calendar_event.get('title')}")
                        
                        # Display Spotify links if available
                        if data.get("spotify_links"):
                            spotify_data = data["spotify_links"]
                            if spotify_data.get("success"):
                                display_spotify_results({
                                    "function_name": "spotify_link_discovery",
                                    "data": spotify_data.get("data", {})
                                })
                        
                        # Display attractions if available
                        if data.get("attractions"):
                            attractions_data = data["attractions"]
                            if attractions_data.get("success"):
                                display_attraction_results(attractions_data, auto_open=st.session_state.get('auto_open_maps', False))
                        
                        # Display features if available
                        if data.get("features"):
                            st.session_state['email_features'] = data["features"]
                    
                    # Handle Single Agent response
                    else:
                        st.success("✅ Email features extracted successfully")
                        
                        email_features = api_response.get("features")
                        function_result = api_response.get("function_result", [])
                        
                        # Display features
                        if email_features:
                            st.session_state['email_features'] = email_features
                        
                        # Display function results
                        if function_result:
                            st.info(f"📊 {len(function_result)} function(s) executed")
                            if isinstance(function_result, dict) and "response" in function_result:
                                st.markdown("**Function Responses:**")
                                st.info(f"ℹ️ No functions were executed: {function_result.get('response')}")
                            else:
                                # Process all function results (can be multiple functions)
                                if isinstance(function_result, list):
                                    for func_result in function_result:
                                        function_name = func_result.get("function_name", "")
                                        
                                        # Handle calendar event creation
                                        if function_name == "create_event":
                                            ics_file_path = func_result.get("data", {}).get("ics_file_path")
                                            if ics_file_path:
                                                st.success(f"📄 ICS file created: {ics_file_path}")
                                                try:
                                                    with open(ics_file_path, "rb") as f:
                                                        ics_data = f.read()
                                                    st.download_button(
                                                        label="📥 Download ICS Calendar File",
                                                        data=ics_data,
                                                        file_name=ics_file_path,
                                                        mime="text/calendar",
                                                        type="primary"
                                                    )
                                                except Exception as e:
                                                    st.error(f"Failed to download ICS file: {str(e)}")
                                            else:
                                                st.info("ℹ️ No calendar file generated (no time-based event detected)")
                                        
                                        # Handle Spotify link discovery
                                        elif function_name == "spotify_link_discovery":
                                            display_spotify_results(func_result)
                                        
                                        # Handle attraction discovery
                                        elif function_name == "attraction_discovery":
                                            display_attraction_results(func_result, auto_open=st.session_state.get('auto_open_maps', False))
                                else:
                                    st.info("ℹ️ No functions were executed")
                        else:
                            st.info("ℹ️ No functions were executed")
                else:
                    st.warning(f"⚠️ Email processing failed: {api_response.get('message', 'Unknown error')}")
            else:
                st.error("❌ Failed to process email via backend API")

with colB:
    st.subheader("Email Content")
    # Load email from data array
    if 0 <= idx < len(data):
        item = data[idx]
        subject = item.get("subject", "")
        body = item.get("body", "")
    else:
        # Fallback to session state if index is out of range
        subject = st.session_state.get("subject_area", "")
        body = st.session_state.get("body_area", "")
    
    st.markdown("**Subject:**")
    st.text_area("", subject, key="subject_area", height=100)
    
    st.markdown("**Body:**")
    st.text_area("", body, key="body_area", height=300)

st.divider()
st.subheader("Extracted Features")

if 'email_features' in st.session_state:
    st.json(st.session_state['email_features'])
else:
    st.info("No features extracted yet. Click 'Extract and Manage Email Features' to process an email.")
