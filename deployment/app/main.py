import os
import sys
import json
from typing import Any, Optional

from fastapi import FastAPI, Query, HTTPException
import joblib
from pydantic import BaseModel as PBaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path to import from deployment root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add app directory to path for relative imports
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_DIR)

from email_manager.calendar_code import process_email_to_calendar

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
# Constant Variables
OPENAI_MODEL_NAME = "gpt-4o-mini"
_DEV_DIR = os.path.dirname(_APP_DIR)

# Load environment variables from .env file
load_dotenv()  # Load .env if exists
# Also try loading spotify.env (it will override .env values)
spotify_env_path = os.path.join(_DEV_DIR, "spotify.env")
if os.path.exists(spotify_env_path):
    load_dotenv(spotify_env_path, override=True)
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
                        for r in result.get("data", {}).get('songs', []):
                            print(f"🎵 Song Name: {r.get('song')}")
                            print(f"👤 Artist: {r.get('artist')}")
                            print(f"🔗 Spotify Link: {r.get('spotify_url')}")
                    case "attraction_discovery":
                        data = result.get("data", {})
                        # Support both old format (attractions) and new format (direct_match + recommendations)
                        direct_match = data.get('direct_match', [])
                        recommendations = data.get('recommendations', [])
                        old_attractions = data.get('attractions', [])
                        
                        if direct_match:
                            print(f"📍 Direct Match Attractions ({len(direct_match)}):")
                            for r in direct_match:
                                print(f"  🎭 {r.get('name')} - {r.get('location')}")
                        
                        if recommendations:
                            print(f"🌟 Recommended Attractions ({len(recommendations)}):")
                            for r in recommendations:
                                print(f"  🎭 {r.get('name')} - {r.get('location')}")
                        
                        # Old format support
                        if old_attractions:
                            print(f"🎭 Attractions ({len(old_attractions)}):")
                            for r in old_attractions:
                                print(f"  🎭 {r.get('name')} - {r.get('location')}")
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
    """Create calendar event from email and handle Spotify/attractions if needed (Multi-Agent)"""
    import time
    start_time = time.time()
    
    try:
        print(f"📧 Processing email via create endpoint (Multi-Agent)...")
        full_text = f"Subject: {req.subject}\n\nBody: {req.body}" if req.subject else req.body
        print(f"⏱️ Step 1: Extracting email features...")
        features = extract_email_features(full_text)
        features.category = req.category or features.category
        elapsed = time.time() - start_time
        print(f"✅ Features extracted in {elapsed:.2f}s")
        
        # Initialize response structure
        result = {
            "calendar_event": None,
            "spotify_links": None,
            "attractions": None,
            "features": features.model_dump()
        }
        
        # Check if email contains music/concert information
        email_text_lower = full_text.lower()
        music_keywords = ["music", "song", "artist", "concert", "spotify", "band", "album", "track"]
        has_music_content = any(keyword in email_text_lower for keyword in music_keywords)
        
        # Check if email contains travel/tourism information
        travel_keywords = ["travel", "tourism", "attraction", "visit", "tour", "sightseeing", "landmark"]
        has_travel_content = any(keyword in email_text_lower for keyword in travel_keywords)
        
        # Process calendar event (multi-agent)
        print(f"⏱️ Step 2: Processing calendar event (Multi-Agent)...")
        calendar_start = time.time()
        calendar_response = process_email_to_calendar(features)
        calendar_elapsed = time.time() - calendar_start
        print(f"✅ Calendar processing completed in {calendar_elapsed:.2f}s")
        if calendar_response and calendar_response.get("calendar_event"):
            result["calendar_event"] = calendar_response
        
        # Process Spotify if music content detected (Multi-Agent mode)
        if has_music_content:
            try:
                from classes.FunctionCall import FunctionCall
                function_call = FunctionCall(features, full_text)
                # Use Multi-Agent mode for Spotify recommendations
                spotify_result = function_call.spotify_link_discovery(use_multi_agent=True)
                if spotify_result.get("success") and spotify_result.get("data", {}).get("songs"):
                    result["spotify_links"] = spotify_result
            except Exception as e:
                print(f"⚠️ Spotify discovery failed: {e}")
                import traceback
                traceback.print_exc()
                result["spotify_links"] = {
                    "success": False,
                    "message": f"Spotify discovery failed: {str(e)}"
                }
        
        # Process attractions if travel content detected (Multi-Agent mode)
        if has_travel_content:
            try:
                from classes.FunctionCall import FunctionCall
                function_call = FunctionCall(features, full_text)
                # Use Multi-Agent mode for attractions
                attractions_result = function_call.attraction_discovery(use_multi_agent=True)
                # Check if we have any attractions (direct_match or recommendations)
                if attractions_result.get("success"):
                    data = attractions_result.get("data", {})
                    direct_match = data.get("direct_match", [])
                    recommendations = data.get("recommendations", [])
                    # Also support old format for backward compatibility
                    old_attractions = data.get("attractions", [])
                    if direct_match or recommendations or old_attractions:
                        result["attractions"] = attractions_result
            except Exception as e:
                print(f"⚠️ Attraction discovery failed: {e}")
                import traceback
                traceback.print_exc()
                result["attractions"] = {
                    "success": False,
                    "message": f"Attraction discovery failed: {str(e)}"
                }
        
        total_elapsed = time.time() - start_time
        print(f"✅ Multi-Agent processing completed in {total_elapsed:.2f}s")
        return {
            "success": True,
            "data": result,
            "_processing_time": round(total_elapsed, 2)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error after {elapsed:.2f}s: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/function_call")
async def function_call_endpoint(req: EmailRequest):
    """Process email and execute appropriate function"""
    import time
    start_time = time.time()
    
    try:
        print(f"📧 Processing email via function_call endpoint...")
        full_text = f"Subject: {req.subject}\n\nBody: {req.body}" if req.subject else req.body
        print(f"⏱️ Step 1: Extracting email features...")
        features = extract_email_features(full_text)
        elapsed = time.time() - start_time
        print(f"✅ Features extracted in {elapsed:.2f}s")
        
        print(f"⏱️ Step 2: Calling functions...")
        response = function_calling(features, full_text)
        total_elapsed = time.time() - start_time
        print(f"✅ Function calling completed in {total_elapsed:.2f}s")
        
        return {
            "success": True,
            "features": features.model_dump(),
            "function_result": response,
            "_processing_time": round(total_elapsed, 2)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error after {elapsed:.2f}s: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


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