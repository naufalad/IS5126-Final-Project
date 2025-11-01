import os
import json
from typing import Optional, List
from enum import Enum
from datetime import datetime, date, time

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel as PBaseModel, Field, field_validator
from openai import OpenAI


OPENAI_MODEL_NAME = "gpt-4o-mini"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecurrencePattern(str, Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class EventType(str, Enum):
    APPOINTMENT = "appointment"
    MEETING = "meeting"
    DEADLINE = "deadline"
    MAINTENANCE = "maintenance"
    PAYMENT = "payment"
    VERIFICATION = "verification"
    NOTIFICATION = "notification"
    REMINDER = "reminder"
    FINAL = "final"
    OTHER = "other"


class ActionRequirement(str, Enum):
    CONFIRM = "confirm"
    REPLY = "reply"
    PAY = "pay"
    VERIFY = "verify"
    CLICK = "click"
    DOWNLOAD = "download"
    COMPLETE = "complete"
    REVIEW = "review"
    NONE = "none"


class LocationType(str, Enum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    HYBRID = "hybrid"
    NONE = "none"


class EmailFeatures(PBaseModel):
    email_text: Optional[str] = Field(None)
    scheduled_datetime: Optional[datetime] = Field(None)
    date_text: Optional[str] = Field(None)
    date_from: Optional[date] = Field(None)
    date_to: Optional[date] = Field(None)
    time_from: Optional[time] = Field(None)
    time_to: Optional[time] = Field(None)
    has_complete_datetime: bool = Field(False)
    location: Optional[str] = Field(None)
    meeting_url: Optional[str] = Field(None)
    maps_url: Optional[str] = Field(None)
    coordinates: Optional[str] = Field(None)
    location_type: Optional[LocationType] = Field(None)
    event_type: Optional[EventType] = Field(None)
    event_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    urgency_level: Optional[UrgencyLevel] = Field(None)
    urgency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    urgency_indicators: List[str] = Field(default_factory=list)
    recurrence_pattern: Optional[RecurrencePattern] = Field(None)
    recurrence_text: Optional[str] = Field(None)
    action_required: Optional[ActionRequirement] = Field(None)
    action_deadline: Optional[datetime] = Field(None)
    action_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    action_phrases: List[str] = Field(default_factory=list)
    contains_links: bool = Field(False)
    contains_attachments: bool = Field(False)
    financial_amount: Optional[str] = Field(None)

    @field_validator('location_type', mode='before')
    @classmethod
    def set_location_type_default(cls, v):
        return v if v is not None else LocationType.NONE

    @field_validator('event_type', mode='before')
    @classmethod
    def set_event_type_default(cls, v):
        return v if v is not None else EventType.OTHER

    @field_validator('urgency_level', mode='before')
    @classmethod
    def set_urgency_level_default(cls, v):
        return v if v is not None else UrgencyLevel.LOW

    @field_validator('recurrence_pattern', mode='before')
    @classmethod
    def set_recurrence_pattern_default(cls, v):
        return v if v is not None else RecurrencePattern.NONE

    @field_validator('action_required', mode='before')
    @classmethod
    def set_action_required_default(cls, v):
        if v is True:
            return ActionRequirement.NONE
        return v if v is not None else ActionRequirement.NONE

    @field_validator('event_confidence', 'urgency_score', 'action_confidence', mode='before')
    @classmethod
    def set_score_default(cls, v):
        return v if v is not None else 0.0

    class Config:
        json_encoders = { datetime: lambda v: v.isoformat() if v else None }
        use_enum_values = True


def create_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = (
    "You are an expert email analyzer. Extract structured information in the specified JSON format."
)


def extract_email_features(email_text: str, subject: str = "") -> EmailFeatures:
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    full_text = f"Subject: {subject}\n\nBody: {email_text}" if subject else email_text
    user_prompt = f"Analyze this email and extract structured features:\n\n{full_text}"

    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
        max_tokens=1200,
        response_format={"type": "json_object"}
    )
    return EmailFeatures(**json.loads(response.choices[0].message.content))


app = FastAPI(title="Email Feature Extraction API", version="1.0.0")


class PredictRequest(PBaseModel):
    subject: Optional[str] = None
    body: str


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        features = extract_email_features(req.body, req.subject or "")
        return {"success": True, "data": features.model_dump()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Data browsing endpoints (read-only) — resolve relative to this file
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_DIR = os.path.dirname(_APP_DIR)
DATA_PATH = os.path.join(_DEV_DIR, "data", "email_features.json")


@app.get("/data/emails")
async def list_emails(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        total = len(items) if isinstance(items, list) else 0
        sliced = items[offset: offset + limit] if isinstance(items, list) else []
        return {"success": True, "total": total, "offset": offset, "limit": limit, "items": sliced}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/emails/{idx}")
async def get_email(idx: int):
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list) or idx < 0 or idx >= len(items):
            raise HTTPException(status_code=404, detail="Index out of range")
        return {"success": True, "item": items[idx], "index": idx}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_PATH}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


