from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime, date, time
from enum import Enum

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

class EmailFeatures(BaseModel):
    """Pydantic model for extracting structured features from email content."""

    email_text: Optional[str] = Field(None, description="Full email text")

    # Date/Time fields
    scheduled_datetime: Optional[datetime] = Field(None, description="Extracted date and time")
    date_text: Optional[str] = Field(None, description="Raw date/time text")
    date_from: Optional[date] = Field(None, description="Start date (YYYY-MM-DD)")
    date_to: Optional[date] = Field(None, description="End date (YYYY-MM-DD)")
    time_from: Optional[time] = Field(None, description="Start time (HH:MM:SS 24-hour)")
    time_to: Optional[time] = Field(None, description="End time (HH:MM:SS 24-hour)")
    has_complete_datetime: bool = Field(False, description="True if both date and time present")

    # Location
    location: Optional[str] = Field(None, description="Meeting location")
    meeting_url: Optional[str] = Field(None, description="Virtual meeting URL")
    maps_url: Optional[str] = Field(None, description="Maps URL")
    coordinates: Optional[str] = Field(None, description="Coordinates")
    location_type: Optional[LocationType] = Field(None, description="Location type")  # ← Changed to Optional

    # Event
    title: Optional[str] = Field(None, description="Event title")
    event_type: Optional[EventType] = Field(None, description="Event type")  # ← Changed to Optional
    event_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Event confidence")  # ← Changed to Optional

    # Urgency
    urgency_level: Optional[UrgencyLevel] = Field(None, description="Urgency level")  # ← Changed to Optional
    urgency_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Urgency score")  # ← Changed to Optional
    urgency_indicators: List[str] = Field(default_factory=list, description="Urgency phrases")

    # Recurrence
    recurrence_pattern: Optional[RecurrencePattern] = Field(None, description="Recurrence")  # ← Changed to Optional
    recurrence_text: Optional[str] = Field(None, description="Recurrence text")

    # Action
    action_required: Optional[ActionRequirement] = Field(None, description="Action required")  # ← Changed to Optional
    action_deadline: Optional[datetime] = Field(None, description="Action deadline")
    action_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Action confidence")  # ← Changed to Optional
    action_phrases: List[str] = Field(default_factory=list, description="Action phrases")

    # Metadata
    contains_links: bool = Field(False, description="Contains links")
    contains_attachments: bool = Field(False, description="Contains attachments")
    financial_amount: Optional[str] = Field(None, description="Financial amounts")

    # Validators to set defaults when None is provided
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
        # Handle boolean True being passed (OpenAI bug)
        if v is True:
            return ActionRequirement.NONE
        return v if v is not None else ActionRequirement.NONE

    @field_validator('event_confidence', 'urgency_score', 'action_confidence', mode='before')
    @classmethod
    def set_score_default(cls, v):
        return v if v is not None else 0.0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            date: lambda v: v.isoformat() if v else None,
            time: lambda v: v.isoformat() if v else None
        }
        use_enum_values = True