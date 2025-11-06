import streamlit as st


st.markdown("""
<style>
/* Sidebar main container */
section[data-testid="stSidebar"] {
  background-color: #1E1E1E !important;
  color: #FFFFFF !important;
}

/* Sidebar navigation items / headers etc */
section[data-testid="stSidebar"] div.css-1aumxhk {
  background-color: #2E2E2E !important;
  color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Email Dashboard", page_icon="📊", layout="wide")

st.title("Email Dashboard")
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


