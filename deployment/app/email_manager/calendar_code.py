import os
import json
from datetime import datetime, timedelta
import pytz
import dateparser
from icalendar import Calendar, Event, vText
from openai import OpenAI
import sys
import json
from datetime import datetime, date, time
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Constants
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEV_DIR = os.path.dirname(_APP_DIR)
CALENDAR_PATH = os.path.join(_DEV_DIR, "data", "calendar", "events.json")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in temp.env or environment variables.")

client = OpenAI(api_key=api_key)

DEFAULT_TZ = "UTC"


def parse_event_with_llm(text):
    """
    Extract structured calendar info using OpenAI.
    """
    system_prompt = (
    "You are an assistant that extracts calendar event data from text. "
    "Respond ONLY with a valid JSON object (no markdown, no explanations). "
    "Include these fields: title, start_time, end_time, location, description. "
    "Use ISO 8601 datetime format. "
    "If end_time is missing, set it one hour after start_time."
)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Model output was not valid JSON:\n{content}")

class CalendarFunction():
    def __init__(self, email_features):
        self.email_features = email_features
        # Create event object
        self.event = {
            "title": email_features.title or "Untitled Event",
            "start": None,  # Will be set below
            "end": None,    # Will be set below
            "description": email_features.email_text or "",
            "location": email_features.location or "",
            "label": self._get_event_label(),
            "meeting_url": email_features.meeting_url or "",
            "urgency_level": str(email_features.urgency_level) if email_features.urgency_level else "low",
            "action_required": str(email_features.action_required) if email_features.action_required else "none"
        }
        
        # Combine date and time for ISO format
        if email_features.date_from and email_features.time_from:
            start_datetime = datetime.combine(
                email_features.date_from,
                email_features.time_from
            )
            self.event["start"] = start_datetime.isoformat()
        elif email_features.date_from:
            # If only date, set time to midnight
            self.event["start"] = datetime.combine(
                email_features.date_from,
                time(0, 0, 0)
            ).isoformat()

        if email_features.date_to and email_features.time_to:
            end_datetime = datetime.combine(
                email_features.date_to,
                email_features.time_to
            )
            self.event["end"] = end_datetime.isoformat()
        elif email_features.date_to:
            # If only date, set time to midnight
            self.event["end"] = datetime.combine(
                email_features.date_to,
                time(23, 59, 59)
            ).isoformat()
        elif self.event["start"]:
            # If no end date, copy start date
            self.event["end"] = self.event["start"]
        

    def create_ics(self, output_path="event.ics"):
        """
        Build and save an .ics file.
        """
        cal = Calendar()
        cal.add('prodid', '-//OpenAI Calendar Agent//')
        cal.add('version', '2.0')

        event = Event()
        event.add('summary', self.event['title'])
        event.add('description', self.event.get('description', ''))
        if self.event.get('location'):
            event.add('location', vText(self.event['location']))

        tz = pytz.timezone(DEFAULT_TZ)
        start = dateparser.parse(self.event['start'])
        if not start:
            raise ValueError("Unable to parse start_time")
        end = dateparser.parse(self.event.get('end'))
        if not end:
            end = start + timedelta(hours=1)

        event.add('dtstart', start.astimezone(tz))
        event.add('dtend', end.astimezone(tz))
        event.add('dtstamp', datetime.now(tz))
        event['uid'] = f"{datetime.now().timestamp()}@llm-agent"

        cal.add_component(event)

        with open(output_path, "wb") as f:
            f.write(cal.to_ical())

        return output_path

    def save_calendar(self)  -> Dict[str, Any]:
        path = CALENDAR_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Load existing events or create new list
            events = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                        if not isinstance(events, list):
                            events = [events]
                except:
                    events = []
            
            # Append new event
            events.append(self.event)
            
            # Save to file
            with open(path, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Calendar event created: {self.event['title']}")
            return self.event
            
        except Exception as e:
            error_msg = f"Failed to create calendar event: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    def _get_event_label(self) -> str:
        """Map event type to calendar label."""
        if not self.email_features.event_type:
            return "other"

        event_type = str(self.email_features.event_type).lower()

        # Map event types to calendar labels
        label_mapping = {
            "meeting": "meeting",
            "appointment": "appointment",
            "deadline": "deadline",
            "reminder": "reminder",
            "payment": "deadline",
            "verification": "reminder",
            "notification": "reminder",
            "maintenance": "appointment",
        }
        
        return label_mapping.get(event_type, "other")