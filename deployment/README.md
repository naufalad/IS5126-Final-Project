## Development - Run Guides (Windows PowerShell)

Structure:

- Backend (FastAPI): `deployment/app/main.py`
- Frontend (Streamlit): `deployment/ui/Home.py`
- Data (local): `deployment/data/email_features.json`

Prerequisites (once at project root):

```
pip install -r requirements.txt
$env:OPENAI_API_KEY="<YOUR_API_KEY>"
```

### Run backend and frontend

**Terminal 1 (Backend)** - Navigate to deployment directory first:
```powershell
cd deployment
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend)** - Navigate to deployment/ui directory:
```powershell
cd deployment/ui
streamlit run Home.py
```

**Alternative (from project root):**
```powershell
# Backend
cd deployment; uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend  
cd deployment/ui; streamlit run Home.py
```