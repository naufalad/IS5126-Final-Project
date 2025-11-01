## Development - Run Guides (Windows PowerShell)

Structure:

- Backend (FastAPI): `development/app/main.py`
- Frontend (Streamlit): `development/ui/streamlit_app.py`
- Data (local): `development/data/email_features.json`

Prerequisites (once at project root):

```
pip install -r requirements.txt
$env:OPENAI_API_KEY="<YOUR_API_KEY>"
```

### Backend only (FastAPI)

```
uvicorn development.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open docs: http://127.0.0.1:8000/docs

### Frontend only (Streamlit)

Keep Backend API URL as http://127.0.0.1:8000 (or your actual one) in the sidebar.

```
streamlit run development/ui/streamlit_app.py
```

Optional Streamlit secret `.streamlit/secrets.toml`:

```
API_URL = "http://127.0.0.1:8000"
```

### Run backend and frontend (recommended, two terminals)

```
# Terminal 1 (backend)![1761934812856](image/README/1761934812856.png)
uvicorn development.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 (frontend)
![1761934483779](image/README/1761934483779.png)
```
