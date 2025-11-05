import os
import sys
import json
from typing import Any, Dict, Optional, List
from enum import Enum
from datetime import datetime, date, time
from fastapi import FastAPI, Query, HTTPException
import joblib
from pydantic import BaseModel as PBaseModel, Field, field_validator
from dotenv import load_dotenv
from openai import OpenAI

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
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# Load environment variables from .env file
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Constant Variables
OPENAI_MODEL_NAME = "gpt-4o-mini"
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_DIR = os.path.dirname(_APP_DIR)
DATA_PATH = os.path.join(_DEV_DIR, "data", "email_features.json")
CALENDAR_PATH = os.path.join(_DEV_DIR, "data", "calendar", "events.json")


# ============================================================================
# PYDANTIC REQUEST MODELS
# ============================================================================
CALENDAR_PATH = os.path.join(_DEV_DIR, "data", "calendar.json")
CONCERT_PATH = os.path.join(_DEV_DIR, "data", "concerts.json")



# Class Definitions for Email Features Extraction
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

class FunctionCall():
    def __init__(self, email_features: EmailFeatures):
        self.email_features = email_features
    # function to create calendar event
    def create_calendar_event(self) -> bool:
        path = CALENDAR_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            event = {
                "title": f"Event: {self.email_features.event_type}",
                "start_date": self.email_features.date_from.isoformat() if self.email_features.date_from else None,
                "end_date": self.email_features.date_to.isoformat() if self.email_features.date_to else None,
                "start_time": self.email_features.time_from.isoformat() if self.email_features.time_from else None,
                "end_time": self.email_features.time_to.isoformat() if self.email_features.time_to else None,
                "location": self.email_features.location,
                "meeting_url": self.email_features.meeting_url,
                "notes": self.email_features.email_text,
                "urgency_level": self.email_features.urgency_level,
                "description": f"Action Required: {self.email_features.action_required}",
                "label": self.email_features.category if hasattr(self.email_features, 'category') else "general"
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save calendar: {e}")
            return False
    #----- Spotify discovery end----   
    def spotify_link_discovery(self) -> List[Dict[str, Any]]:
       
        """
    Discover Spotify links for artists mentioned in the email.
    Reads concert data from concerts.json and searches artists on Spotify.
    Returns a list of artist information including Spotify links.
    """
        try:
           # Step 1: Read concert data from JSON file
           concert_data = []
           if os.path.exists(CONCERT_PATH):
            try:
                with open(CONCERT_PATH, "r", encoding="utf-8") as f:
                    concert_data = json.load(f)
                    if not isinstance(concert_data, list):
                        concert_data = []
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {CONCERT_PATH}")
                concert_data = []
            else:
                print(f"Warning: Concert data is not a list in {CONCERT_PATH}")
            print(f"Warning: Concert data file not found at {CONCERT_PATH}")
            
            # Step 2: Extract artist names using LLM from email
            SYSTEM_PROMPT = """You are an expert at identifying musician and artist names from text.
            Extract all musician, band, or artist names mentioned in the email.
            Return ONLY a JSON array of artist names, nothing else.
            Example output format:
            ["Atif Aslam", "Coldplay", "Taylor Swift"]
            If no artists are found, return an empty array: []"""
            USER_PROMPT = f"""Extract all musician/artist names from this email:
            {self.email_features.email_text}
            Return only a JSON array of artist names."""
            # Call LLM to extract artist names
            client = create_openai_client()
            if not client:
                return [{"error": "OpenAI client not initialized"}]
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.1,
                max_tokens=200
            )
            llm_response = response.choices[0].message.content.strip()
            try:
                artist_names = json.loads(llm_response)
                if not isinstance(artist_names, list):
                    artist_names = []
            except json.JSONDecodeError:
                artist_names = []
            
            if not artist_names:
                return [{"message": "No artists found in email"}]
        
            # Step 3: Check if artists exist in concert data
            matched_concerts=[]
            for artist_name in artist_names:
            # Search in concert data (case-insensitive matching)
                artist_lower = artist_name.lower()
                for concert in concert_data:
                    concert_artist = concert.get("artist", "").lower()
                    concert_name = concert.get("name", "").lower()
                    
                    # Match if artist name is in concert artist or concert name
                    if artist_lower in concert_artist or artist_lower in concert_name:
                        matched_concerts.append({
                            "artist_name": artist_name,
                            "concert_info": concert,
                            "matched": True
                        })
            
            # Step 4: Search for each artist on Spotify
            spotify_token = os.getenv('SPOTIFY_ACCESS_TOKEN')
            if not spotify_token:
                return [{"error": "Spotify access token not found. Set SPOTIFY_ACCESS_TOKEN in .env"}]
            results = []
            for artist_name in artist_names:
                try:
                    # Search Spotify API
                    import requests
                    headers = {
                        "Authorization": f"Bearer {spotify_token}"
                    }
                    params = {
                        "q": artist_name,
                        "type": "artist",
                        "limit": 1
                    }
                    spotify_response = requests.get(
                        "https://api.spotify.com/v1/search",
                        headers=headers,
                        params=params,
                        timeout=10
                    )
                    if spotify_response.status_code == 200:
                        data = spotify_response.json()
                        if data.get("artists", {}).get("items"):
                            artist = data["artists"]["items"][0]
                            # Find matching concert info
                            concert_info = None
                            for matched in matched_concerts:
                                if matched["artist_name"].lower() == artist_name.lower():
                                    concert_info = matched["concert_info"]
                                    break
                            result = {
                                "name": artist.get("name"),
                                "spotify_url": artist.get("external_urls", {}).get("spotify"),
                                "spotify_id": artist.get("id"),
                                "followers": artist.get("followers", {}).get("total"),
                                "popularity": artist.get("popularity"),
                                "genres": artist.get("genres", []),
                                "image_url": artist.get("images", [{}])[0].get("url") if artist.get("images") else None,
                                "searched_term": artist_name,
                                "has_concert_data": concert_info is not None
                            }
                            # Add concert information if found
                            if concert_info:
                                result["concert_info"] = concert_info
                            
                            results.append(result)
                        else:
                                results.append({
                                "searched_term": artist_name,
                                "error": "Artist not found on Spotify",
                                "has_concert_data": False
                            })
                    elif spotify_response.status_code == 401:
                        return [{"error": "Spotify token expired or invalid. Please refresh your token."}]
                    else:
                        results.append({
                            "searched_term": artist_name,
                            "error": f"Spotify API error: {spotify_response.status_code}",
                            "has_concert_data": False
                        })
                except requests.exceptions.RequestException as e:
                    results.append({
                        "searched_term": artist_name,
                        "error": f"Request failed: {str(e)}",
                        "has_concert_data": False
                    })
                except Exception as e:
                    results.append({
                        "searched_term": artist_name,
                        "error": f"Error processing artist: {str(e)}",
                        "has_concert_data": False
                    })
            # Step 5: Save results back to concerts.json (optional - update with Spotify data)
            try:
                updated_concert_data = concert_data.copy()
                for result in results:
                    if result.get("spotify_url"):
                        # Update or add Spotify info to concert data
                        artist_name = result["name"]
                        for concert in updated_concert_data:
                            concert_artist = concert.get("artist", "")
                            if artist_name.lower() in concert_artist.lower():
                                concert["spotify_url"] = result["spotify_url"]
                                concert["spotify_id"] = result["spotify_id"]
                                concert["spotify_data"] = {
                                    "followers": result["followers"],
                                    "popularity": result["popularity"],
                                    "genres": result["genres"]
                                }
                # Write back to file
                with open(CONCERT_PATH, "w", encoding="utf-8") as f:
                    json.dump(updated_concert_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to update concert data: {e}")
            return results
        except Exception as e:
            return [{"error": f"Failed to discover Spotify links: {str(e)}"}]

    #----- Spotify discovery end----
    def attraction_discovery(self) -> List[Dict[str, Any]]:
        pass

    # function wrapper for function calling
    def function_call(self, function:str, **kwargs):
        if function == "create_calendar_event":
            result = self.create_calendar_event(**kwargs)
        elif function == "spotify_link_discovery":
            result = self.spotify_link_discovery(**kwargs)
        elif function == "attraction_discovery":
            result = self.attraction_discovery(**kwargs)
        return result

# Function Schema
create_event_schema = {
    "type": "function",
    "function": {
        "name": "create_event_schema",
        "description": "Create a calendar event based on extracted email features",
    }
}
spotify_link_schema = {
    "type": "function",
    "function": {
        "name": "spotify_link_discovery_schema",
        "description": "Discover Spotify links based on extracted email features",
    }
}
attraction_discovery_schema = {
    "type": "function",
    "function": {
        "name": "attraction_discovery_schema",
        "description": "Discover scientific attractions based on extracted email features",
    }
}

def create_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def extract_email_features(email_text: str, subject: str = "") -> EmailFeatures:
    SYSTEM_PROMPT = (
        """You are an expert email analyzer. Extract structured information from emails and return it in the specified JSON format.

            Focus on identifying:
            1. Scheduled dates/times (appointments, deadlines, events) - extract date ranges and time ranges
            2. Urgency indicators (urgent, asap, now, today, deadline, final notice, etc.)
            3. Event types (meetings, payments, verifications, etc.)
            4. Required actions (confirm, reply, pay, verify, etc.)
            5. Recurrence patterns (daily, weekly, monthly, etc.)
            6. Financial amounts and deadlines
            7. Location information (physical addresses, venue names, virtual meeting URLs, coordinates)

            For dates and times:
            - Extract start and end dates separately (date_from and date_to in YYYY-MM-DD format)
            - Extract start and end times separately (time_from and time_to in HH:MM:SS 24-hour format)
            - If only one date mentioned, use same value for both date_from and date_to
            - If only one time mentioned, use same value for both time_from and time_to
            - Convert 12-hour format to 24-hour (1 PM = 13:00:00, 2:30 PM = 14:30:00, etc.)
            - Set has_complete_datetime to true only if BOTH date AND time are present

        Return valid JSON matching the EmailFeatures schema exactly."""
    )
    full_text = f"Subject: {subject}\n\nBody: {email_text}" if subject else email_text
    user_prompt = f"""Analyze this email and extract structured features:
        {full_text}

        Return a JSON object with these fields:

        DATE AND TIME FIELDS (NEW - IMPORTANT):
        - date_from: start date in YYYY-MM-DD format (e.g., "2025-11-15"), null if no date
        - date_to: end date in YYYY-MM-DD format (same as date_from if single date), null if no date
        - time_from: start time in HH:MM:SS 24-hour format (e.g., "13:00:00" for 1 PM), null if no time
        - time_to: end time in HH:MM:SS 24-hour format (same as time_from if single time), null if no time
        - has_complete_datetime: boolean - true ONLY if both date and time are present, false otherwise

        LEGACY DATE/TIME FIELDS:
        - scheduled_datetime: ISO datetime string if specific date/time mentioned, null otherwise
        - date_text: raw text containing date/time info, null if none

        URGENCY:
        - urgency_level: one of [low, medium, high, critical]
        - urgency_score: float 0.0-1.0
        - urgency_indicators: array of urgency phrases found

        LOCATION:
        - location: meeting location, address, or venue name, null if none
        - meeting_url: virtual meeting URL (Zoom, Teams, etc.), null if none
        - maps_url: Google Maps or other map service URL, null if none
        - coordinates: geographic coordinates (latitude, longitude), null if none
        - location_type: one of [physical, virtual, hybrid, none]

        EVENT:
        - event_type: one of [appointment, meeting, deadline, maintenance, payment, verification, notification, reminder, final, other]
        - event_confidence: float 0.0-1.0

        RECURRENCE:
        - recurrence_pattern: one of [none, daily, weekly, monthly, yearly, custom]
        - recurrence_text: raw recurrence text, null if none

        ACTION:
        - action_required: one of [confirm, reply, pay, verify, click, download, complete, review, none]
        - action_deadline: ISO datetime for action deadline, null if none
        - action_confidence: float 0.0-1.0
        - action_phrases: array of action-indicating phrases

        METADATA:
        - contains_links: boolean
        - contains_attachments: boolean
        - financial_amount: string of any monetary amounts, null if none

        EXAMPLES OF TIME CONVERSION:
        - "1 PM" or "13" → "13:00:00"
        - "2:30 PM" → "14:30:00"
        - "9 AM" → "09:00:00"
        - "midnight" → "00:00:00"
        - "noon" → "12:00:00"

        EXAMPLES OF DATE EXTRACTION:
        - "Meeting on Nov 15, 2025" → date_from: "2025-11-15", date_to: "2025-11-15"
        - "Conference from Dec 1-3" → date_from: "2025-12-01", date_to: "2025-12-03"
        - "this week in 2021" → extract specific date if possible, otherwise null

        EXAMPLES OF has_complete_datetime:
        - Has date "Nov 15" and time "2 PM" → has_complete_datetime: true
        - Has only date "Nov 15" → has_complete_datetime: false
        - Has only time "2 PM" → has_complete_datetime: false
        - No date or time → has_complete_datetime: false"""

    response = call_llm(SYSTEM_PROMPT, user_prompt)
    return EmailFeatures(**json.loads(response))

def explain_email_categories(email_text: str, category: str = "") -> str:
    SYSTEM_PROMPT = (
        """You are an expert email classifier. Given an email, explain why it belongs to a specific category.
        Here are the list of possible categories:
        - Promotions: Marketing emails promoting products, services, or events.
        - Spam: Unsolicited bulk emails, often promotional or fraudulent.
        - Social Media: Emails between individuals for personal communication.
        - Forums: Notifications from online communities or discussion groups.
        - Updates: Notifications about account activity, order status, or service changes.
        - Verify Code: Emails containing verification codes for account access or security.
        - Flight Booking: Emails related to flight reservations, itineraries, or travel updates.
        - Concert Promotions: Emails promoting concerts, music events, or ticket sales.

        Focus on identifying key phrases, context, and indicators in the email text that justify the classification.

        Provide a clear, concise explanation highlighting the main reasons for the category assignment."""
    )
    user_prompt = f"""Analyze this email and explain why it belongs to the category '{category}':
        {email_text}

        Provide a detailed explanation with references to specific parts of the email text."""
    response = call_llm(SYSTEM_PROMPT, user_prompt)
    return response

def function_calling(email_features: EmailFeatures):
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    
    function_call = FunctionCall(email_features).function_call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert assistant that can call functions to perform tasks based on user queries."
            },
            {
                "role": "user",
                "content": "Based on the extracted email features, decide which function to call to handle the email appropriately."
            }
        ],
        tools=[create_event_schema, spotify_link_schema, attraction_discovery_schema],  # Available functions
        tool_choice="auto"  # Let the model decide when to call functions
    )

    message = response.choices[0].message

    # Step 2: Check if the model wants to call a function
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"🔧 LLM decided to call: {function_name}({function_args})")

        # Step 3: Execute the function
        result = function_call(function_name, **function_args)
        if 'error' in result:
            print(f"❌ {result['error']}")
        elif result == None:
            print(f"❌ function return nothing")
        try:
            match function_name:
                case "create_event":
                    print(f"Event Created: {result.event_title} on {result.start_date} at {result.start_time}")
                case "spotify_link_discovery":
                    for r in result:
                        print(f"Song Name: {r.name}")
                        print(f"Artist: {r.name}")
                        print(f"Spotify Link: {r.applications}")
                case "attraction_discovery":
                    for r in result:
                        print(f"Attraction Name: {r.name}")
                        print(f"Location: {r.location}")

        except Exception as e:
            print(f"❌ Structuring failed: {e}")
            attempt+=1
    else:
        # Model didn't call a function - gave direct response
        print(f"💬 Direct response: {message.content}")

    return result

app = FastAPI(title="Email Feature Extraction API", version="1.0.0")

class EmailRequest(PBaseModel):
    subject: Optional[str] = None
    body: str

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
                        for r in result.get("data", []).get('songs', []):
                            print(f"🎵 Song Name: {r.get('song')}")
                            print(f"👤 Artist: {r.get('artist')}")
                            print(f"🔗 Spotify Link: {r.get('spotify_url')}")
                    case "attraction_discovery":
                        for r in result.get("data", []).get('attractions', []):
                            print(f"🎭 Attraction Name: {r.get('name')}")
                            print(f"📍 Location: {r.get('location')}")
                            print(f"🏷 Type: {r.get('type')}")
                            print(f"📝 Description: {r.get('description')}")
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
                model_data = joblib.load(os.path.join(MODEL_DIRECTORY, 'chocka.joblib'))
                # model = model_data["model"]
            case 2:
                # MPNET + XGBoost
                model_data = joblib.load(os.path.join(MODEL_DIRECTORY, 'habibi.joblib'))
            case 3:
                # CNN
                model_data = joblib.load(os.path.join(MODEL_DIRECTORY, 'source.joblib'))
            case _:
                raise ValueError(f"Invalid model selection: {req.model}")
        
        # Combine subject and body
        input_data = f"{req.subject} {req.body}" if req.subject else req.body
        
        # Make prediction
        prediction, probabilities = model_data.predict([input_data])[0]
        
        return {
            "success": True,
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
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
        response = function_calling(features, full_text)
        
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

@app.post("/ui/spotify-discovery")
async def ui_spotify_discovery(req: EmailRequest):
    """
    Main POST endpoint for UI to discover Spotify artists from email.
    This endpoint will:
    1. Extract artist names from email using LLM
    2. Search artists on Spotify API
    3. Match with concert data from concerts.json
    4. Update concerts.json with Spotify enrichment
    
    Request body:
    {
        "subject": "Concert Alert",  (optional)
        "body": "See Atif Aslam live in concert..."
    }
    
    Returns:
    {
        "success": true,
        "message": "Found 2 artists with Spotify data",
        "artists_count": 2,
        "concerts_matched": 1,
        "data": [...]
    }
    """
    try:
        # Validate input
        if not req.body or req.body.strip() == "":
            return {
                "success": False,
                "message": "Email body is required",
                "artists_count": 0,
                "concerts_matched": 0,
                "data": []
            }
        
        # Extract features from email
        features = extract_email_features(req.body, req.subject or "")
        
        # Call spotify discovery function
        function_caller = FunctionCall(features)
        results = function_caller.spotify_link_discovery()
        
        # Count statistics
        successful_artists = [r for r in results if r.get("spotify_url") and not r.get("error")]
        artists_count = len(successful_artists)
        concerts_matched = len([r for r in results if r.get("has_concert_data")])
        
        # Check if any errors
        errors = [r for r in results if r.get("error")]
        has_critical_error = any("token" in r.get("error", "").lower() or "client not initialized" in r.get("error", "").lower() for r in errors)
        
        if has_critical_error:
            return {
                "success": False,
                "message": errors[0].get("error", "Unknown error"),
                "artists_count": 0,
                "concerts_matched": 0,
                "data": results
            }
        
        # Success message
        if artists_count == 0:
            message = "No artists found in email"
        elif artists_count == 1:
            message = f"Found 1 artist with Spotify data"
        else:
            message = f"Found {artists_count} artists with Spotify data"
        
        if concerts_matched > 0:
            message += f" ({concerts_matched} matched with concert data)"
        
        return {
            "success": True,
            "message": message,
            "artists_count": artists_count,
            "concerts_matched": concerts_matched,
            "data": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Server error: {str(e)}",
            "artists_count": 0,
            "concerts_matched": 0,
            "data": []
        }


