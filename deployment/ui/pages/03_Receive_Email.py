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
st.caption("Select one email record, review details, then click 'Receive' to proceed.")

# Backend API configuration
load_dotenv()  # Load .env file if it exists

BACKEND_URL = os.getenv("BACKEND_API", "http://127.0.0.1:8000")

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


def calendar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
    ui_dir = os.path.dirname(current_dir)  # .../ui
    return os.path.join(os.path.dirname(ui_dir), "data", "calendar", "events.json")


def load_events():
    path = calendar_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_events(events):
    path = calendar_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save calendar: {e}")
        return False


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
    if st.button("Predict Email Category", type="primary"):
        st.session_state['received_email_index'] = idx_display  # Store 1-based index
        received_item = data[idx]
        subject = received_item.get("subject", "")
        body = received_item.get("body", "")
        st.session_state['received_email_item'] = received_item

        # Auto-create calendar event for final-type emails via backend API
        etype = (received_item.get("event_type") or "").strip().lower()
        
        with st.spinner("Processing email..."):
            # add button to choose which model to be used, between 1 2 or 3
            model_choice = st.radio("Choose model:", (1, 2, 3))
            if etype == "final":
                # Prepare data for backend API
                email_payload = {
                    "subject": subject,
                    "body": body,
                    "model": model_choice
                }
                
                # Call backend API
                api_response = call_predict_api(email_payload)
                
                if api_response:
                    # Check if event was created
                    if api_response.get("status") == "success":
                        created_event = api_response.get("event")
                        if created_event:
                            # Save to local calendar as well for UI display
                            events = load_events()
                            events.append(created_event)
                            save_events(events)
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
            else:
                # Non-final emails: just acknowledge receipt
                st.success(f"Received email #{idx_display} (Type: {etype})")
    elif st.button("Extract and Manage Email Features", type="primary"):
        st.session_state['received_email_index'] = idx_display  # Store 1-based index
        received_item = data[idx]
        subject = received_item.get("subject", "")
        body = received_item.get("body", "")
        st.session_state['received_email_item'] = received_item
        
        # Auto-create calendar event for final-type emails via backend API
        etype = (received_item.get("event_type") or "").strip().lower()
        
        with st.spinner("Processing email..."):
            if etype == "final":
                # Prepare data for backend API
                email_payload = {
                    "subject": subject,
                    "body": body,
                }
                
                # Call backend API
                api_response = call_function_call_api(email_payload)
                
                if api_response:
                    # Check if event was created
                    if api_response.get("status") == "success":
                        created_event = api_response.get("event")
                        if created_event:
                            # Save to local calendar as well for UI display
                            events = load_events()
                            events.append(created_event)
                            save_events(events)
                            st.success(f"Received email #{idx_display}")
                            st.info(f"Calendar event created: '{created_event.get('title', 'Untitled')}'")
                        else:
                            st.success(f"Received email #{idx_display}")
                            st.info("No calendar event created (missing information)")
                    else:
                        st.info(api_response)
                        st.warning(f"Email received but event creation failed: {api_response.get('message', 'Unknown error')}")
                else:
                    st.error("Failed to process email via backend API")
            else:
                # Non-final emails: just acknowledge receipt
                st.success(f"Received email #{idx_display} (Type: {etype})")

with colB:
    st.subheader("Email Content")
    if(idx>4):
        subject = ""
        body = ""
    else:
        item = data[idx]
        subject = item.get("subject", "")
        body = item.get("body", "")
        features = {k: v for k, v in item.items() if k != 'email_text'}
    
    st.markdown("**Subject:**")
    st.text_area("", subject, key="subject_area", height=100)
    
    st.markdown("**Body:**")
    st.text_area("", body, key="body_area", height=300)

st.divider()
st.subheader("Extracted Features")

if 'features' in locals():
    st.json(features)
else:
    st.info("No features available for this record")