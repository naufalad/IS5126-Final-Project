import os
import json
import re
import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="App for Entering Verification", page_icon="✅", layout="wide")
st.title("App for Entering Verification")
st.caption("Enter your verification code")


def create_openai_client():
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        return None
    try:
        return OpenAI(api_key=api)
    except Exception:
        return None


def extract_code_via_llm(text: str) -> str:
    """Use OpenAI to extract a 4-8 digit verification code from email text."""
    client = create_openai_client()
    if not client or not isinstance(text, str):
        return ""
    system_prompt = (
        "You extract verification codes from short emails. "
        "Return a JSON object with a single key 'code' containing the numeric code. "
        "If no code is found, return {'code': ''}."
    )
    user_prompt = f"Email:\n{text}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        code = str(data.get("code", "")).strip()
        # Sanity check: 4-8 digits
        return code if re.fullmatch(r"\d{4,8}", code or "") else ""
    except Exception:
        return ""


prefill = ""
received = st.session_state.get("received_email_item")
if received:
    prefill = extract_code_via_llm(received.get("email_text", ""))

code = st.text_input("Verification code (4-8 digits)", value=prefill, max_chars=8)

if st.button("Submit", type="primary"):
    if not code or not re.fullmatch(r"\d{4,8}", code):
        st.error("Please enter a 4-8 digit numeric code.")
    else:
        st.session_state["verification_code"] = code
        st.success("Verification submitted.")



