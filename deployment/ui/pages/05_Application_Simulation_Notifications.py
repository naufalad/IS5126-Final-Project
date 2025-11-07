import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
import re


st.set_page_config(page_title="Notifications", page_icon="🔔", layout="wide")
st.title("Notifications")


# Paths
def data_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
    ui_dir = os.path.dirname(current_dir)  # .../ui
    dev_dir = os.path.dirname(ui_dir)  # .../development
    return os.path.join(dev_dir, "data")


def notif_path():
    # New location: development/data/notifications/events.json (mirror calendar structure)
    return os.path.join(data_dir(), "notifications", "events.json")


def load_notifications():
    try:
        path = notif_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Backward compatibility/migration from old paths
        old_flat = os.path.join(data_dir(), "notifications.json")
        old_entries = os.path.join(data_dir(), "notifications", "entries.json")
        for old_path in (old_flat, old_entries):
            if os.path.exists(old_path):
                with open(old_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as wf:
                    json.dump(data, wf, ensure_ascii=False, indent=2)
                return data
    except Exception as e:
        st.error(f"Failed to load notifications: {e}")
    return []


def save_notifications(items):
    try:
        os.makedirs(os.path.dirname(notif_path()), exist_ok=True)
        with open(notif_path(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save notifications: {e}")
        return False


# Ingest from session when user clicked Receive Email
notifs = load_notifications()
received = st.session_state.get("received_email_item")
if received and received.get("event_type") == "notification":
    # build a concise notification record
    text = str(received.get("email_text", ""))
    # keep full text; we will parse Subject/Body when rendering
    title = text
    ts = datetime.utcnow().isoformat() + "Z"
    new_item = {
        "id": hash(text) % (10**12),
        "title": title,
        "preview": text[:160],
        "timestamp": ts,
        "pinned": False,
        "meta": {
            "event_type": received.get("event_type"),
            "urgency": received.get("urgency_level"),
            "contains_links": received.get("contains_links")
        }
    }
    # de-duplicate by id
    if all(x.get("id") != new_item["id"] for x in notifs):
        notifs.insert(0, new_item)
        save_notifications(notifs)


# Header stats
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Notifications", len(notifs))
with col2:
    link_count = sum(1 for n in notifs if n.get("meta", {}).get("contains_links"))
    st.metric("With Links", link_count)

st.divider()


def extract_subject_body(text: str):
    if not isinstance(text, str):
        return "", ""
    if text.startswith("Subject:") and "\n\nBody:" in text:
        try:
            head, tail = text.split("\n\nBody:", 1)
            subject = head.replace("Subject:", "").strip()
            body = tail.strip()
            return subject, body
        except Exception:
            pass
    # Fallback: first line as subject
    parts = text.splitlines()
    return (parts[0] if parts else "", "\n".join(parts[1:]) if len(parts) > 1 else "")


def linkify(text: str) -> str:
    # convert bare URLs to markdown links
    url_re = re.compile(r"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)")
    return url_re.sub(r"<\1>", text)


def render_card(n, idx):
    with st.container():
        c1, c2, c3 = st.columns([8, 1, 1])
        with c1:
            full = n.get('title', 'Notification')
            subj, body = extract_subject_body(full)
            # Subject in bold markdown
            st.markdown(f"**Subject: {subj or '(no subject)'}**")
            # Body as markdown with linkified URLs
            if body:
                st.markdown(linkify(body))
            st.caption(n.get("timestamp", ""))
        with c2:
            pin_label = "📌 Unpin" if n.get("pinned") else "📌 Pin"
            if st.button(pin_label, key=f"pin_{n.get('id')}"):
                n["pinned"] = not n.get("pinned", False)
                save_notifications(notifs)
                st.rerun()
        with c3:
            if st.button("🗑️ Delete", key=f"del_{n.get('id')}"):
                # remove by id
                nid = n.get("id")
                notifs[:] = [x for x in notifs if x.get("id") != nid]
                save_notifications(notifs)
                st.rerun()
        st.divider()


if notifs:
    # Pinned first
    ordered = sorted(notifs, key=lambda x: (not x.get("pinned", False), x.get("timestamp", "")), reverse=False)
    for i, n in enumerate(ordered):
        render_card(n, i)
else:
    st.info("No notifications yet. Go to Receive Email and pick a record with event_type = 'notification'.")



