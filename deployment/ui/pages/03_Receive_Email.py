import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re


st.set_page_config(page_title="Receive Email", page_icon="📥", layout="wide")
st.title("📥 Receive Email")
st.caption("Select one email record, review details, then click 'Receive' to proceed.")


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


def generate_calendar_event_via_llm(email_text: str) -> dict:
    """Ask GPT to create a calendar event object matching our calendar schema."""
    client = create_openai_client()
    if not client or not isinstance(email_text, str) or not email_text.strip():
        return {}

    system_prompt = (
        "You generate calendar events from emails. Return valid JSON with keys: "
        "title (string), start (ISO datetime), end (ISO datetime), description (string), "
        "location (string), label (one of: meeting, appointment, deadline, reminder, other). "
        "Keep it concise; if any field is unknown, use empty string. If end is unknown, copy start."
    )
    user_prompt = f"Email to analyze and summarize into a calendar event:\n\n{email_text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        # Basic normalization
        event = {
            "title": str(data.get("title", "")).strip(),
            "start": str(data.get("start", "")).strip(),
            "end": str(data.get("end", "")).strip(),
            "description": str(data.get("description", "")).strip(),
            "location": str(data.get("location", "")).strip(),
            "label": (str(data.get("label", "other")).strip() or "other"),
        }
        # sanity: ensure label is allowed
        if event["label"] not in {"meeting", "appointment", "deadline", "reminder", "other"}:
            event["label"] = "other"
        # simple ISO check (very light)
        for k in ("start", "end"):
            if event[k] and not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", event[k]):
                # leave as-is; frontend will coerce or user can edit later
                pass
        if not event["end"] and event["start"]:
            event["end"] = event["start"]
        return event
    except Exception as e:
        st.warning(f"Event generation failed: {e}")
        return {}


data = load_local_data()
if not data:
    st.warning("No data available")
    st.stop()


colA, colB = st.columns([1, 3])
with colA:
    idx_display = st.number_input("Index", min_value=1, max_value=len(data), value=1, step=1)
    idx = idx_display - 1  # Convert to 0-based for array access
    preview_len = st.slider("Preview length", 40, 400, 160)
    if st.button("Receive Email", type="primary"):
        st.session_state['received_email_index'] = idx_display  # Store 1-based index
        st.session_state['received_email_item'] = data[idx]
        st.success(f"Received email #{idx_display}")
        # Auto-create calendar event for final-type emails
        received_item = data[idx]
        etype = (received_item.get("event_type") or "").strip().lower()
        if etype == "final":
            evt = generate_calendar_event_via_llm(received_item.get("email_text", ""))
            # 仅当包含有效起始时间（ISO 格式且含时间部分）时才创建
            if evt and isinstance(evt.get("start"), str) and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", evt.get("start")):
                events = load_events()
                events.append(evt)
                if save_events(events):
                    st.info("Auto-created calendar event from FINAL email")
            else:
                st.info("Skipped creating calendar event (missing time)")

with colB:
    item = data[idx]
    st.subheader("Email Content")
    full_text = item.get("email_text", "")
    etype = item.get("event_type") or "(none)"
    st.markdown(f"**Event type:** `{etype}`")
    st.text_area("", full_text[:preview_len] + ("..." if len(full_text) > preview_len else ""), height=240, disabled=True)

st.divider()
st.subheader("Extracted Features")
features = {k: v for k, v in item.items() if k != 'email_text'}
if features:
    st.json(features)
else:
    st.info("No features available for this record")


