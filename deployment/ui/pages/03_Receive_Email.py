import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re
import requests
from dotenv import load_dotenv


st.set_page_config(page_title="Email Prediction and Extraction", page_icon="", layout="wide")
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

def load_local_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
        ui_dir = os.path.dirname(current_dir)  # .../ui
        data_path = os.path.join(os.path.dirname(ui_dir), "data", "email_cases.json")
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load local data: {e}")
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

def call_function_call_api(email_data: dict) -> dict:
    """Call backend API /function_call endpoint to process email and create calendar event"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/function_call",
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
    

# -- Spotify Discovery UI Logic --
def call_spotify_discovery_api(email_data: dict) -> dict:
    """Call backend API /ui/spotify-discovery endpoint"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/ui/spotify-discovery",
            json=email_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Failed to call Spotify discovery API: {str(e)}")
        return None
def display_spotify_results(spotify_data):
    """Display Spotify discovery results in a beautiful card layout"""
    if not spotify_data:
        return
    
    success = spotify_data.get('success', False)
    message = spotify_data.get('message', '')
    artists_count = spotify_data.get('artists_count', 0)
    concerts_matched = spotify_data.get('concerts_matched', 0)
    data = spotify_data.get('data', [])
    
    # Header
    st.markdown("---")
    st.markdown("## 🎵 Spotify Artist Discovery")
    
    if success and artists_count > 0:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎤 Artists Found", artists_count)
        with col2:
            st.metric("🎫 Concerts Matched", concerts_matched)
        with col3:
            st.metric("Status", "✅ Success")
        
        st.markdown("---")
        
        # Display each artist
        for artist in data:
            if artist.get('error'):
                st.warning(f"⚠️ {artist.get('searched_term', 'Unknown')}: {artist.get('error')}")
                continue
            
            # Artist info
            artist_name = artist.get('name', 'Unknown Artist')
            spotify_url = artist.get('spotify_url', '#')
            followers = artist.get('followers', 0)
            popularity = artist.get('popularity', 0)
            genres = artist.get('genres', [])
            image_url = artist.get('image_url')
            has_concert = artist.get('has_concert_data', False)
            concert_info = artist.get('concert_info', {})
            
            # Create columns for image and info
            col_img, col_info = st.columns([1, 3])
            
            with col_img:
                if image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.markdown("### 🎤")
            
            with col_info:
                st.markdown(f"""
                <div class="spotify-card">
                    <div class="artist-name">{artist_name}</div>
                    <div class="artist-stats">
                        <div class="stat-item">
                            <span class="stat-label">Followers</span>
                            <span class="stat-value">{followers:,}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Popularity</span>
                            <span class="stat-value">{popularity}/100</span>
                        </div>
                    </div>
                    <div>
                        {''.join([f'<span class="genre-badge">{genre}</span>' for genre in genres[:5]])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Spotify link button
                st.markdown(f"[🎧 Open in Spotify]({spotify_url})", unsafe_allow_html=True)
                
                # Concert information if available
                if has_concert and concert_info:
                    st.markdown(f"""
                    <div class="concert-info">
                        <strong>🎫 Concert Information</strong><br><br>
                        <strong>Event:</strong> {concert_info.get('name', 'N/A')}<br>
                        <strong>Venue:</strong> {concert_info.get('venue', 'N/A')}<br>
                        <strong>Date:</strong> {concert_info.get('date', 'N/A')}<br>
                        <strong>Location:</strong> {concert_info.get('location', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    elif success and artists_count == 0:
        st.info(f"ℹ️ {message}")
    else:
        st.error(f"❌ {message}")
    
    # Initialize session state
    if 'spotify_results' not in st.session_state:
        st.session_state['spotify_results'] = None
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None

data = load_local_data()
if not data:
    st.warning("No data available")
    st.stop()


# colA, colB = st.columns([1, 3])
# with colA:
#     idx_display = st.number_input("Email No.", min_value=0, max_value=len(data), value=0, step=1)
#     idx = idx_display
#     if st.button("Predict Email Category", type="primary"):
#         st.session_state['received_email_index'] = idx_display  # Store 1-based index
#         received_item = data[idx]
#         subject = received_item.get("subject", "")
#         body = received_item.get("body", "")
#         st.session_state['received_email_item'] = received_item

#         # Auto-create calendar event for final-type emails via backend API
#         etype = (received_item.get("event_type") or "").strip().lower()
        
#         with st.spinner("Processing email..."):
#             # add button to choose which model to be used, between 1 2 or 3
#             model_choice = st.radio("Choose model:", (1, 2, 3))
#             if etype == "final":
#                 # Prepare data for backend API
#                 email_payload = {
#                     "subject": subject,
#                     "body": body,
#                     "model": model_choice
#                 }
                
#                 # Call backend API
#                 api_response = call_predict_api(email_payload)
                
#                 if api_response:
#                     # Check if event was created
#                     if api_response.get("status") == "success":
#                         created_event = api_response.get("event")
#                         if created_event:
#                             # Save to local calendar as well for UI display
#                             events = load_events()
#                             events.append(created_event)
#                             save_events(events)
#                             st.success(f"Received email #{idx_display}")
#                             st.info(f"Calendar event created: '{created_event.get('title', 'Untitled')}'")
#                         else:
#                             st.success(f"Received email #{idx_display}")
#                             st.info("No calendar event created (missing time or requirements)")
#                     else:
#                         st.info(api_response)
#                         st.warning(f"Email received but event creation failed: {api_response.get('message', 'Unknown error')}")
#                 else:
#                     st.error("Failed to process email via backend API")
#             else:
#                 # Non-final emails: just acknowledge receipt
#                 st.success(f"Received email #{idx_display} (Type: {etype})")
#     elif st.button("Extract and Manage Email Features", type="primary"):
#         st.session_state['received_email_index'] = idx_display  # Store 1-based index
#         received_item = data[idx]
#         subject = received_item.get("subject", "")
#         body = received_item.get("body", "")
#         st.session_state['received_email_item'] = received_item
        
#         # Auto-create calendar event for final-type emails via backend API
#         etype = (received_item.get("event_type") or "").strip().lower()
        
#         with st.spinner("Processing email..."):
#             if etype == "final":
#                 # Prepare data for backend API
#                 email_payload = {
#                     "subject": subject,
#                     "body": body,
#                 }
                
#                 # Call backend API
#                 api_response = call_function_call_api(email_payload)
                
#                 if api_response:
#                     # Check if event was created
#                     if api_response.get("status") == "success":
#                         created_event = api_response.get("event")
#                         if created_event:
#                             # Save to local calendar as well for UI display
#                             events = load_events()
#                             events.append(created_event)
#                             save_events(events)
#                             st.success(f"Received email #{idx_display}")
#                             st.info(f"Calendar event created: '{created_event.get('title', 'Untitled')}'")
#                         else:
#                             st.success(f"Received email #{idx_display}")
#                             st.info("No calendar event created (missing information)")
#                     else:
#                         st.info(api_response)
#                         st.warning(f"Email received but event creation failed: {api_response.get('message', 'Unknown error')}")
#                 else:
#                     st.error("Failed to process email via backend API")
#             else:
#                 # Non-final emails: just acknowledge receipt
#                 st.success(f"Received email #{idx_display} (Type: {etype})")
# Main UI
colA, colB = st.columns([1, 3])

with colA:
    idx_display = st.number_input("Email No.", min_value=0, max_value=len(data)-1, value=0, step=1)
    idx = idx_display
    model_choice = st.radio("Choose classification model:", [1, 2, 3], horizontal=True)
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
                # Check if event was created
                if api_response.get("status") == "success":
                    st.success("Received email prediction")
                    created_event = api_response.get("event")
                    if created_event:
                        # Save to local calendar as well for UI display
                        st.success(f"Received email #{idx_display}")
                        st.info(f"Calendar event created: '{created_event.get('title', 'Untitled')}'")
                    else:
                        st.success(f"Received email #{idx_display}")
                        st.info("No calendar event created (missing time or requirements)")
                else:
                    st.info(api_response)
                    st.warning(f"Email received but event creation failed: {api_response.get('message', 'Unknown error')}")
            else:
                st.error("Failed to process email via backend API")
    if st.button("Extract and Manage Email Features", type="primary"):
        st.session_state['received_email_index'] = idx_display
        subject = st.session_state.get("subject_area", "")
        body = st.session_state.get("body_area", "")
        
        with st.spinner("Processing email..."):
            # Prepare data for backend API
            email_payload = {
                "subject": subject,
                "body": body,
            }
            
            # Call backend API
            api_response = call_function_call_api(email_payload)
            
            if api_response:
                # Check if event was managed
                if api_response.get("success"):
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
                            # Handle ICS file download if event created
                            if function_result and function_result[0].get("function_name")=="create_event":
                                ics_file_path = function_result[0].get("data", {}).get("ics_file_path")
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
                    else:
                        st.info("ℹ️ No functions were executed")
                else:
                    st.warning(f"⚠️ Email processing failed: {api_response.get('message', 'Unknown error')}")
            else:
                st.error("❌ Failed to process email via backend API")

    
    # Model selection
    model_choice = st.radio("Choose model:", (1, 2, 3), key="model_radio")
    
    # Button 1: Predict Email Category
    if st.button("🔮 Predict Email Category", type="primary", use_container_width=True):
        st.session_state['received_email_index'] = idx_display
        received_item = data[idx]
        subject = received_item.get("subject", "")
        body = received_item.get("body", "")
        st.session_state['received_email_item'] = received_item

        etype = (received_item.get("event_type") or "").strip().lower()
        
        with st.spinner("🔄 Processing email..."):
            email_payload = {
                "subject": subject,
                "body": body,
                "model": model_choice
            }
            
            # Call predict API
            api_response = call_predict_api(email_payload)
            
            if api_response and api_response.get("success"):
                prediction = api_response.get("prediction", "Unknown")
                st.success(f"✅ Email #{idx_display} predicted successfully!")
                st.info(f"📧 Category: **{prediction}**")
                
                # Store prediction result
                st.session_state['prediction_result'] = api_response
                
                # Auto-trigger Spotify discovery for concert emails
                if "concert" in prediction.lower() or "promotion" in prediction.lower():
                    with st.spinner("🎵 Discovering Spotify artists..."):
                        spotify_response = call_spotify_discovery_api(email_payload)
                        st.session_state['spotify_results'] = spotify_response
            else:
                st.error("❌ Failed to predict email category")
    
    st.markdown("---")
    
    # Button 2: Extract and Manage Features
    if st.button("📝 Extract and Manage Features", type="primary", use_container_width=True):
        st.session_state['received_email_index'] = idx_display
        received_item = data[idx]
        subject = received_item.get("subject", "")
        body = received_item.get("body", "")
        st.session_state['received_email_item'] = received_item
        
        etype = (received_item.get("event_type") or "").strip().lower()
        
        with st.spinner("🔄 Processing email..."):
            email_payload = {
                "subject": subject,
                "body": body,
            }
            
            api_response = call_function_call_api(email_payload)
            
            if api_response and api_response.get("success"):
                st.success(f"✅ Email #{idx_display} processed successfully!")
            else:
                st.warning(f"⚠️ {api_response.get('message', 'Unknown error') if api_response else 'Failed to process'}")
    
    st.markdown("---")
    
    # Button 3: Discover Spotify Artists (Manual)
    if st.button("🎵 Discover Spotify Artists", use_container_width=True):
        received_item = data[idx]
        subject = received_item.get("subject", "")
        body = received_item.get("body", "")
        
        with st.spinner("🎸 Searching for artists on Spotify..."):
            email_payload = {
                "subject": subject,
                "body": body,
            }
            
            spotify_response = call_spotify_discovery_api(email_payload)
            st.session_state['spotify_results'] = spotify_response
            
            if spotify_response and spotify_response.get('success'):
                st.success(f"✅ Found {spotify_response.get('artists_count', 0)} artist(s)!")


# ----- end colA spotify----
with colB:
    st.subheader("Email Content")
    if(idx>4):
        subject = st.session_state.get("subject_area", "")
        body = st.session_state.get("body_area", "")
    else:
        item = data[idx]
        subject = item.get("subject", "")
        body = item.get("body", "")
    
    st.markdown("**Subject:**")
    st.text_area("", subject, key="subject_area", height=100)
    
    st.markdown("**Body:**")
    st.text_area("", body, key="body_area", height=300)

#Display prediction results if available
if st.session_state.get('prediction_result'):
    result = st.session_state['prediction_result']
    
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📧 Predicted Category", result.get('prediction', 'Unknown'))
    with col2:
        probabilities = result.get('probabilities', [])
        if probabilities:
            max_prob = max(probabilities)
            st.metric("✅ Confidence", f"{max_prob*100:.2f}%")
    
    # Display explanation
    if result.get('explanation'):
        st.markdown("**🧠 Explanation:**")
        st.info(result['explanation'])


if 'email_features' in st.session_state:
    st.json(st.session_state['email_features'])

# # Display Spotify Discovery Results
# if st.session_state.get('spotify_results'):
#     display_spotify_results(st.session_state['spotify_results'])

# st.divider()

# # Display extracted features
# st.subheader("🔍 Extracted Features")
# if features:
#     st.json(features)
# else:
#     st.info("No features extracted yet. Click 'Extract and Manage Email Features' to process an email.")
