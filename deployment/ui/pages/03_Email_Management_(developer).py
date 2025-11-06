import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re
import requests
from dotenv import load_dotenv
import urllib.parse as _urlparse
import time


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

# ---------------------------------------------
# Helper: extract explicit location mention
# 优先使用已提取的 features.location；否则用简单规则从正文中抓取“专有名词, 专有名词”样式的地点，例如
# "Marina Bay Sands, Singapore"、"Changi Airport, Singapore"
# ---------------------------------------------
def extract_explicit_location(subject: str, body: str, features: dict | None) -> str | None:
    try:
        if features and isinstance(features, dict):
            loc = features.get("location") or features.get("venue")
            if isinstance(loc, str) and loc.strip():
                return loc.strip()
    except Exception:
        pass
    text = f"{subject or ''}\n{body or ''}"
    # 简单启发式：匹配形如 "Word Word, Word" 的短语（首字母大写的词组 + 逗号 + 首字母大写的词组）
    # 尽量不贪婪，避免抓太长
    pattern = r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*, [A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)"
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    return None

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
        st.success(f"🎵 You Might Like - {len(songs)} recommendations")
        st.caption("💡 Based on your email content, here are suggested songs")
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


def display_attraction_results(func_result_or_data: dict, auto_open: bool = False,
                               direct_header: str | None = None,
                               rec_header: str | None = None,
                               show_count: bool = True):
    """Display attraction discovery results with Google Maps links.
    Supports two categories: direct_match and recommendations.
    Accepts either the full function_result dict (with data field) or a data dict.
    
    Args:
        show_count: If True, display the total count message. Default True.
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

    # Only show count if requested
    if show_count:
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
        header_txt = direct_header or "📍 Direct Match - Mentioned Location"
        st.markdown(f"<div class='section-header'>{header_txt}</div>", unsafe_allow_html=True)
        for i, attr in enumerate(direct_match):
            render_attraction(attr, auto_open_first=(i == 0))
    
    # Display recommendations (if requested)
    if recommendations:
        header_txt = rec_header or "🌟 Recommended Attractions"
        st.markdown(f"<div class='section-header'>{header_txt}</div>", unsafe_allow_html=True)
        for i, attr in enumerate(recommendations):
            render_attraction(attr, auto_open_first=(i == 0 and len(direct_match) == 0))


def call_function_call_api(email_data: dict) -> dict:
    """Call backend API /function_call endpoint to process email and create calendar event"""
    url = f"{BACKEND_URL}/function_call"
    try:
        # Increased timeout to 120 seconds for Single Agent processing (includes LLM calls)
        response = requests.post(
            url,
            json=email_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out after 120 seconds. The backend may still be processing.")
        st.info("💡 **Tips:**\n- Single Agent processing can take 60-120 seconds\n- Try using Multi Agent mode for faster processing\n- Check backend logs for more details")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        st.info(f"Backend URL: {BACKEND_URL}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to call API: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def call_multi_agent_api(email_data: dict) -> dict:
    """Call backend API /create endpoint to process email and create calendar event with longer timeout and retry."""
    url = f"{BACKEND_URL}/create"
    max_retries = 2
    backoff_seconds = 2
    for attempt in range(1, max_retries + 1):
        start_ts = time.time()
        try:
            response = requests.post(
                url,
                json=email_data,
                timeout=120  # extended timeout for multi-agent LLM processing
            )
            response.raise_for_status()
            data = response.json()
            data["_elapsed_seconds"] = round(time.time() - start_ts, 2)
            return data
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries:
                time.sleep(backoff_seconds)
                continue
            st.error("⏱️ Request timed out. The backend may still be processing. Please try again.")
            st.caption(f"Endpoint: {url}")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend server. Please ensure the backend is running on port 8000.")
            st.caption(f"Tried: {url}")
            return None
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(backoff_seconds)
                continue
            st.error("Request timed out. The backend may be processing - please try again.")
            st.caption(f"Endpoint: {url}")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            st.caption(f"Endpoint: {url}")
            return None
        except Exception as e:
            st.error(f"Failed to call API: {str(e)}")
            st.caption(f"Endpoint: {url}")
            return None

def call_predict_api(email_data: dict) -> dict:
    """Call backend API /predict endpoint to process email and create calendar event"""
    url = f"{BACKEND_URL}/predict"
    try:
        # Increased timeout to 60 seconds for model prediction
        response = requests.post(
            url,
            json=email_data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out after 60 seconds. Model prediction may be taking longer than expected.")
        st.info("💡 **Tips:**\n- Large emails may take longer to process\n- Check backend logs for more details")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        st.info(f"Backend URL: {BACKEND_URL}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to call API: {str(e)}")
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
        # Clear previous extraction results immediately and forcefully
        # Use a flag to track if we're processing a new extraction
        st.session_state['_extraction_in_progress'] = True
        
        # Completely remove old results
        keys_to_clear = ['spotify_result', 'attraction_display_data', 'attraction_reco_data', 'calendar_event_created']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
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
                        response_data = api_response.get("data", {})
                        st.success("✅ Multi-Agent processing completed")
                        if "_elapsed_seconds" in api_response:
                            st.caption(f"Backend elapsed: {api_response['_elapsed_seconds']}s")
                        
                        # Display calendar event if available
                        if response_data.get("calendar_event"):
                            calendar_data = response_data["calendar_event"]
                            if calendar_data and calendar_data.get("calendar_event"):
                                st.success("📅 Calendar event created successfully")
                                calendar_event = calendar_data.get("calendar_event", {})
                                if calendar_event.get("title"):
                                    st.info(f"📅 Event: {calendar_event.get('title')}")
                        
                        # Display Spotify links if available
                        if response_data.get("spotify_links"):
                            spotify_data = response_data["spotify_links"]
                            if spotify_data.get("success"):
                                display_spotify_results({
                                    "function_name": "spotify_link_discovery",
                                    "data": spotify_data.get("data", {})
                                })
                        
                        # Display attractions if available (move to right side; split direct vs 猜你喜欢)
                        if response_data.get("attractions"):
                            attractions_data = response_data["attractions"]
                            if attractions_data.get("success"):
                                # Ensure previous data is cleared (should already be cleared, but double-check)
                                st.session_state.pop('attraction_display_data', None)
                                st.session_state.pop('attraction_reco_data', None)
                                
                                dat = attractions_data.get("data", {})
                                # Multi-Agent mode: directly use backend's direct_match (from Single Agent code)
                                direct_list = dat.get("direct_match", [])
                                rec_list = dat.get("recommendations", [])

                                # Set mentioned location data (direct_match from backend - already processed by Single Agent)
                                if direct_list:
                                    st.session_state['attraction_display_data'] = {
                                        "direct_match": direct_list,  # Use backend's direct_match directly
                                        "recommendations": []
                                    }
                                
                                # Set recommendation data (限制为3个)
                                if rec_list:
                                    limited_recos = rec_list[:3]  # Limit to 3 recommendations
                                    st.session_state['attraction_reco_data'] = {
                                        "direct_match": [],  # 推荐部分不使用 direct_match
                                        "recommendations": limited_recos  # 推荐放在 recommendations 中
                                    }
                                # If no recommendations, don't set attraction_reco_data at all
                        
                        # Display features if available
                        if response_data.get("features"):
                            st.session_state['email_features'] = response_data["features"]
                    
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
                                    # Single Agent: Process each function result appropriately
                                    for func_result in function_result:
                                        function_name = func_result.get("function_name", "")

                                        if function_name == "spotify_link_discovery":
                                            # Clear previous Spotify result before setting new one
                                            if 'spotify_result' in st.session_state:
                                                del st.session_state['spotify_result']
                                            # Display Spotify links
                                            if func_result.get("success") and func_result.get("data", {}).get("songs"):
                                                st.session_state['spotify_result'] = func_result
                                                st.success("🎵 Spotify links found!")
                                        
                                        elif function_name == "attraction_discovery":
                                            # Clear previous attraction data before setting new data
                                            if 'attraction_display_data' in st.session_state:
                                                del st.session_state['attraction_display_data']
                                            if 'attraction_reco_data' in st.session_state:
                                                del st.session_state['attraction_reco_data']
                                            
                                            # 从邮件中抽取直接提及的地点；仅返回该地点，并给出 Google Maps 搜索链接
                                            subj = st.session_state.get("subject_area", "")
                                            bod = st.session_state.get("body_area", "")
                                            feats = st.session_state.get('email_features')
                                            mention = extract_explicit_location(subj, bod, feats)

                                            render_data = {"direct_match": [], "recommendations": []}
                                            if mention:
                                                q = _urlparse.quote(mention)
                                                gm_url = f"https://www.google.com/maps/search/?api=1&query={q}"
                                                render_data["direct_match"] = [{
                                                    "name": mention,
                                                    "location": mention,
                                                    "type": "place",
                                                    "description": "Directly mentioned in the email",
                                                    "map_link": gm_url
                                                }]
                                            # 如果没有解析到明确地点，则清空（Single Agent 不展示其他）
                                            st.session_state['attraction_display_data'] = render_data
                                        
                                        elif function_name == "create_event":
                                            # Calendar event created - can show success message
                                            if func_result.get("success"):
                                                st.session_state['calendar_event_created'] = True
                                else:
                                    st.info("ℹ️ No functions were executed")
                        else:
                            st.info("ℹ️ No functions were executed")
                else:
                    st.warning(f"⚠️ Email processing failed: {api_response.get('message', 'Unknown error')}")
            else:
                st.error("❌ Failed to process email via backend API")
        
        # Mark extraction as complete
        st.session_state['_extraction_in_progress'] = False

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

    # 在 Email 内容下方展示结果（右栏更宽）
    # Display Spotify results first (if available and not None)
    # Only display if extraction is not in progress (to avoid showing old data)
    if not st.session_state.get('_extraction_in_progress', False):
        spotify_result = st.session_state.get('spotify_result')
        if spotify_result and spotify_result is not None:
            st.markdown("---")
            st.subheader("🎵 Spotify Links")
            display_spotify_results(spotify_result)
    
    # Display attraction results (only if not None and has data)
    # Only display if extraction is not in progress (to avoid showing old data)
    if not st.session_state.get('_extraction_in_progress', False):
        attraction_display = st.session_state.get('attraction_display_data')
        attraction_reco = st.session_state.get('attraction_reco_data')
        
        # Calculate total count for display
        direct_count = len(attraction_display.get('direct_match', [])) if attraction_display else 0
        reco_count = len(attraction_reco.get('recommendations', [])) if attraction_reco else 0
        total_attractions = direct_count + reco_count
        
        # Show total count only once if we have any attractions
        if total_attractions > 0:
            st.success(f"🎭 Found {total_attractions} attraction(s)!")
        
        # Display direct match attractions
        if attraction_display and attraction_display is not None:
            direct_match = attraction_display.get('direct_match', [])
            if direct_match:
                st.markdown("---")
                display_attraction_results(
                    attraction_display,
                    auto_open=st.session_state.get('auto_open_maps', False),
                    direct_header="📍 Mentioned Location",
                    show_count=False  # Don't show count again, we already showed it above
                )
        
        # Display recommended attractions
        if attraction_reco and attraction_reco is not None:
            reco_direct = attraction_reco.get('direct_match', [])
            reco_recs = attraction_reco.get('recommendations', [])
            # 只显示推荐部分（recommendations），限制为3个
            if reco_recs:
                # 确保只显示3个推荐
                limited_reco_recs = reco_recs[:3]
                if limited_reco_recs:
                    st.markdown("---")
                    # 创建一个只包含推荐的数据结构
                    reco_data_to_display = {
                        "direct_match": [],
                        "recommendations": limited_reco_recs
                    }
                    display_attraction_results(
                        reco_data_to_display,
                        auto_open=st.session_state.get('auto_open_maps', False),
                        rec_header="🌟 You Might Like",
                        show_count=False  # Don't show count again, we already showed it above
                    )

st.divider()
st.subheader("Extracted Features")

if 'email_features' in st.session_state:
    st.json(st.session_state['email_features'])
else:
    st.info("No features extracted yet. Click 'Extract and Manage Email Features' to process an email.")
