import os
import sys
import json
from typing import Any, Optional

from fastapi import FastAPI, Query, HTTPException
import joblib
from pydantic import BaseModel as PBaseModel, Field
from dotenv import load_dotenv
from email_manager.calendar_code import process_email_to_calendar

# Add parent directory to path to import from deployment root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import prompts from root deployment folder
from prompts import (
    EMAIL_EXTRACTION_SYSTEM_PROMPT,
    EMAIL_EXPLANATION_SYSTEM_PROMPT,
    FUNCTION_CALLING_SYSTEM_PROMPT,
    FUNCTION_CALLING_USER_PROMPT_TEMPLATE,
    EMAIL_EXPLANATION_USER_PROMPT_TEMPLATE,
    EMAIL_EXTRACTION_USER_PROMPT_TEMPLATE,
    format_prompt
)

# Import data models from data folder
from classes.EmailFeatures import EmailFeatures

# Import function calling utilities
from classes.FunctionCall import (
    FunctionCall,
    create_event_schema,
    spotify_link_schema,
    attraction_discovery_schema,
    create_openai_client,
    call_llm
)
# Load environment variables from .env file
load_dotenv()

# Constant Variables
OPENAI_MODEL_NAME = "gpt-4o-mini"
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_DIR = os.path.dirname(_APP_DIR)
DATA_PATH = os.path.join(_DEV_DIR, "data", "email_features.json")
CALENDAR_PATH = os.path.join(_DEV_DIR, "data", "calendar", "events.json")


# ============================================================================
# PYDANTIC REQUEST MODELS
# ============================================================================

class EmailRequest(PBaseModel):
    subject: Optional[str] = None
    body: str
    category: Optional[str] = None

class PredictRequest(EmailRequest):
    model: int = Field(..., ge=1, le=3, description="Model selection: 1=BERT, 2=MPNET+XGBoost, 3=CNN")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_email_features(email_text: str) -> EmailFeatures:
    """Extract structured features from email text using LLM."""
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    
    try:
        schema = {
            "name": "email_features",
            "schema": EmailFeatures.model_json_schema(),
            "strict": False
        }
        
        # Use imported prompts
        user_prompt = format_prompt(EMAIL_EXTRACTION_USER_PROMPT_TEMPLATE, email_text=email_text)
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": EMAIL_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema
            }
        )
        
        email_features = EmailFeatures.model_validate_json(response.choices[0].message.content)
        return email_features
    except Exception as e:
        raise ValueError(f"Failed to parse EmailFeatures from LLM response: {e}")


def explain_email_categories(email_text: str, category: str = "") -> str:
    """Generate explanation for why an email belongs to a specific category."""
    user_prompt = format_prompt(
        EMAIL_EXPLANATION_USER_PROMPT_TEMPLATE,
        category=category,
        email_text=email_text
    )
    response = call_llm(EMAIL_EXPLANATION_SYSTEM_PROMPT, user_prompt)
    return response


def function_calling(email_features: EmailFeatures, email_text: str = "") -> Any:
    """Determine and execute appropriate function based on email features."""
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    
    function_call = FunctionCall(email_features, email_text).function_call
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": FUNCTION_CALLING_SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(FUNCTION_CALLING_USER_PROMPT_TEMPLATE, email_features=email_features, email_text=email_text)}
        ],
        tools=[create_event_schema, spotify_link_schema, attraction_discovery_schema],
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Check if the model wants to call a function
    results = []
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"🔧 LLM decided to call: {function_name}({function_args})")

            # Execute the function
            result = function_call(function_name, **function_args)
            if result is None:
                print(f"❌ Function returned nothing")
                return None
                
            if isinstance(result, dict) and 'error' in result:
                print(f"❌ {result['error']}")
                return result

            try:
                match function_name:
                    case "create_event":
                        event = result.get("data", {}).get("event", {})
                        print(f"✅ Event Created: {event.get('title')} on {event.get('start')}")
                    case "spotify_link_discovery":
                        if isinstance(result, (list, tuple)) and result:
                            print(f"🎵 Found {len(result)} Song(s):")
                            result = {"songs": result}
                            for r in result["songs"]:
                                print(f"🎵 Song Name: {r.get('name')}")
                                print(f"👤 Artist: {r.get('artist')}")
                                print(f"🗓️ Release Date: {r.get('release_date')}")
                                print(f"💽 Album: {r.get('album')}")
                                print(f"🔗 Spotify Link: {r.get('spotify_url')}")
                        else:
                            print(f"🎵 Song Name: {result.get('name')}")
                            print(f"👤 Artist: {result.get('artist')}")
                            print(f"🗓️ Release Date: {result.get('release_date')}")
                            print(f"💽 Album: {result.get('album')}")
                            print(f"🔗 Spotify Link: {result.get('spotify_url')}")
                    case "attraction_discovery":
                        result = {"attractions": result}
                        for r in result["attractions"]:
                            print(f"🎭 Attraction Name: {r.get('name')}")
                            print(f"📍 Map Link: {r.get('map_link')}")
                            print(f"📝 Description: {r.get('description')}")
                            print(f"🤩 Fun Fact: {r.get('fun_fact')}")
            except Exception as e:
                print(f"❌ Result structuring failed: {e}")
            result['function_name'] = function_name
            results.append(result)
        return results
    else:
        # Model didn't call a function - gave direct response
        print(f"💬 Direct response: {message.content}")
        return {"response": message.content}


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Email Feature Extraction API",
    version="1.0.0",
    description="API for extracting features from emails and performing intelligent actions"
)


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "ok",
        "message": "Email Feature Extraction API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "code": 200}


@app.post("/predict")
async def predict(req: PredictRequest):
    """Predict email category using selected model"""
    try:
        MODEL_DIRECTORY = os.path.join(_DEV_DIR, "models")
        
        # Load selected model
        match req.model:
            case 1:
                # BERT + Transformers
                model_data = joblib.load('./models/bert.joblib')
            case 2:
                # MPNET + XGBoost
                model_data = joblib.load('./models/rf_mpnet_model.joblib')
            case 3:
                # CNN
                model_data = joblib.load('./models/xgb_mpnet_model.joblib')
            case _:
                raise ValueError(f"Invalid model selection: {req.model}")
        
        # Combine subject and body
        input_data = f"{req.subject} {req.body}" if req.subject else req.body
        
        # Make prediction
        prediction, probabilities = model_data.predict(input_data)
        
        return {
            "success": True,
            "prediction": prediction,
            # "probabilities": probabilities.tolist(),
            "explanation": explain_email_categories(input_data, category=prediction),
            "model_used": req.model
        } 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract(req: EmailRequest):
    """Extract structured features from email"""
    try:
        full_text = f"Subject: {req.subject}\n\nBody: {req.body}"
        features = extract_email_features(full_text)
        return {
            "success": True,
            "data": features.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create")
async def create(req: EmailRequest):
    """Create calendar event from email"""
    try:
        full_text = f"Subject: {req.subject}\n\nBody: {req.body}"
        features = extract_email_features(full_text)
        features.category = req.category
        response = process_email_to_calendar(features)
        
        if response is None:
            raise HTTPException(status_code=400, detail="Failed to create event from email")
            
        return {
            "success": True,
            "data": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/function_call")
async def function_call_endpoint(req: EmailRequest):
    """Process email and execute appropriate function"""
    try:
        full_text = f"Subject: {req.subject}\n\nBody: {req.body}" if req.subject else req.body
        features = extract_email_features(full_text)
        response = function_calling(features, full_text)
        return {
            "success": True,
            "features": features.model_dump(),
            "function_result": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/emails")
async def list_emails(
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of items to return")
):
    """List emails with pagination"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        
        total = len(items) if isinstance(items, list) else 0
        sliced = items[offset: offset + limit] if isinstance(items, list) else []
        
        return {
            "success": True,
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": sliced
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/emails/{idx}")
async def get_email(idx: int):
    """Get specific email by index"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        
        if not isinstance(items, list) or idx < 0 or idx >= len(items):
            raise HTTPException(status_code=404, detail="Index out of range")
        
        return {
            "success": True,
            "item": items[idx],
            "index": idx
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_PATH}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)