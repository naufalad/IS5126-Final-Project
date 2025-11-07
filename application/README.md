## Development - Run Guides (Windows PowerShell)

Structure:

- Backend (FastAPI): `application/FastAPI/main.py`
- Frontend (Streamlit): `application/ui/Home.py`
- Data (local): `application/data/email_features.json`

Prerequisites (once at project root):

```
pip install -r requirements.txt
$env:OPENAI_API_KEY="<YOUR_API_KEY>"
```

### Run backend and frontend

**Terminal 1 (Backend)** - Navigate to application directory first:
```powershell
cd application
uvicorn FastAPI.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend)** - Navigate to application/ui directory:
```powershell
cd application/ui
streamlit run Home.py
```

**Alternative (from project root):**
```powershell
# Backend
cd application; uvicorn FastAPI.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend  
cd application/ui; streamlit run Home.py
```