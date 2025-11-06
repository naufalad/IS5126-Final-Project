import os
import json
from datetime import datetime, timedelta
import pytz
import dateparser
from icalendar import Calendar, Event, vText
from openai import OpenAI
from crewai import Agent, Task, Crew, LLM
import json
from datetime import datetime
import time
from dotenv import load_dotenv
from typing import Dict, Any
OPENAI_MODEL_NAME = "gpt-4o-mini"
from crewai import Agent, Task, Crew, LLM

from classes.EmailFeatures import EmailFeatures
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

crewai_llm = LLM(
    model=f"openai/{OPENAI_MODEL_NAME}",
    temperature=0.1,
    max_tokens=1500
)

DEFAULT_TZ = "UTC"

class CalendarFunction():
    def __init__(self, email_features, calendar_event=None):
        self.email_features = email_features
        if calendar_event:
            self.event = calendar_event
        else:
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
        try:
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

            return str(os.path.abspath(output_path))
        except Exception as e:
            print(f"❌ Failed to create ICS file: {e}")
            return None

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


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

classifier_agent = Agent(
    role="Email Classifier",
    goal="Decide if email should be added to the calendar",
    backstory="Expert at identifying calendar-worthy events.",
    verbose=False,
    allow_delegation=False,
    llm=crewai_llm,
)

scheduler_agent = Agent(
    role="DateTime Specialist",
    goal="Validate datetime information for events",
    backstory="Expert at handling date/time logic.",
    verbose=False,
    allow_delegation=False,
    llm=crewai_llm,
)

formatter_agent = Agent(
    role="Event Formatter",
    goal="Create properly formatted calendar event JSON",
    backstory="Expert at building structured calendar entries.",
    verbose=False,
    allow_delegation=False,
    llm=crewai_llm,
)


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_classification_task(email_features: EmailFeatures) -> Task:
    category = email_features.category
    date_from = email_features.date_from

    return Task(
        description=f"""
        Classify email: category={category}, date={date_from}

        If date exists AND (category is concert_promotion OR flight_booking OR event_type is appointment/meeting/deadline):
            Return: {{"should_add": true, "reasoning": "Has valid date", "priority": "high", "confidence": 0.9}}
        Else:
            Return: {{"should_add": false, "reasoning": "Not calendar-worthy", "priority": "low", "confidence": 0.9}}
        """,
        agent=classifier_agent,
        expected_output="JSON with should_add boolean"
    )


def create_scheduling_task(email_features: EmailFeatures) -> Task:
    date_from = email_features.date_from
    return Task(
        description=f"""
        Set scheduling for event: date={date_from}
        Return exactly:
        {{"date_from": "{date_from}", "date_to": "{date_from}", "time_from": "19:00:00", "time_to": "22:00:00", "all_day": true}}
        """,
        agent=scheduler_agent,
        expected_output="JSON with date/time fields"
    )


def create_formatting_task(email_features: EmailFeatures) -> Task:
    text = email_features.email_text[:100]
    location = email_features.location
    return Task(
        description=f"""
        Create title from: {text} at {location}
        Return exactly:
        {{"calendar_title": "Event at Location", "calendar_description": "Description",
          "calendar_color": "#9370DB", "calendar_reminder_minutes": 1440}}
        """,
        agent=formatter_agent,
        expected_output="JSON with 4 fields"
    )

# ============================================================================
# CROSS-PLATFORM TIMEOUT-SAFE PROCESSOR
# ============================================================================

class TimeoutException(Exception):
    pass


def process_email_to_calendar(email_features: EmailFeatures, timeout_seconds: int = 120) -> dict:
    """Processes an email through three CrewAI agents with soft timeout."""
    start_time = time.time()  # Set start time once at the beginning
    
    def check_timeout():
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutException(f"Processing timed out after {timeout_seconds}s (elapsed: {elapsed:.1f}s)")

    try:
        # Task 1 — Classification
        check_timeout()
        classification_task = create_classification_task(email_features)
        classification_result_raw = Crew(agents=[classifier_agent], tasks=[classification_task], verbose=False).kickoff()
        result_str = str(classification_result_raw)
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0].strip()
        classification_result = json.loads(result_str)

        if not classification_result.get("should_add", False):
            return {
                "calendar_event": None,
                "decision": classification_result,
                "processed": True,
                "skipped": True
            }

        # Task 2 — Scheduling
        check_timeout()
        scheduling_task = create_scheduling_task(email_features)
        scheduling_result_raw = Crew(agents=[scheduler_agent], tasks=[scheduling_task], verbose=False).kickoff()
        result_str = str(scheduling_result_raw)
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0].strip()
        scheduling_result = json.loads(result_str)

        # Task 3 — Formatting
        check_timeout()
        formatting_task = create_formatting_task(email_features)
        formatting_result_raw = Crew(agents=[formatter_agent], tasks=[formatting_task], verbose=False).kickoff()
        result_str = str(formatting_result_raw)
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0].strip()
        calendar_fields = json.loads(result_str)
        
        # Merge
        calendar_event = {
            "title": email_features.title,
            "location": email_features.location,
            "start": email_features.date_from,
            "end": email_features.date_to,
            **scheduling_result,
            **calendar_fields
        }
        return {
            "calendar_event": calendar_event,
            "decision": classification_result,
            "scheduling": scheduling_result,
            "processed": True,
            "skipped": False
        }

    except TimeoutException as e:
        print(str(e))
        return {
            "calendar_event": None,
            "decision": {"should_add": False, "reasoning": str(e)},
            "processed": False,
            "skipped": True
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "calendar_event": None,
            "decision": {"should_add": False, "reasoning": str(e)},
            "processed": False,
            "skipped": True
        }