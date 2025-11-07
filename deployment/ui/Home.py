import streamlit as st




st.set_page_config(page_title="Email Management Assistant", page_icon="📊", layout="wide")

st.title("Email Management Assistant")
st.caption("Use the Pages in the left sidebar to navigate.")

st.markdown(
    """
### Available Pages
- Data Browser
- Data Analytics
- Calendar
- Email Management (user)
- Email Management (developer)

Tip: If pages do not appear, stop and re-run the app to reload the `pages/` directory.
"""
)

st.caption("Email Classification Dashboard v1.0 | Powered by FastAPI + Streamlit + OpenAI")


