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

data = load_local_data()
if not data:
    st.warning("No data available")
    st.stop()


colA, colB = st.columns([1, 3])
with colA:
    idx_display = st.number_input("Email No.", min_value=0, max_value=len(data), value=0, step=1)
    idx = idx_display
    model_choice = 1
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
                    explanation = api_response.get("explanation", "No explanation provided")
                    st.markdown("**Explanation:**")
                    st.info(explanation)
                else:
                    st.warning(f"Email received but classification failed: {api_response.get('message', 'Unknown error')}")
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

st.divider()
st.subheader("Extracted Features")

if 'email_features' in st.session_state:
    st.json(st.session_state['email_features'])
else:
    st.info("No features extracted yet. Click 'Extract and Manage Email Features' to process an email.")
