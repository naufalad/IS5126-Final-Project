from unicodedata import category


EMAIL_EXTRACTION_SYSTEM_PROMPT = """You are an expert email analyzer. Extract structured information from emails and return it in the specified JSON format.

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

EMAIL_EXPLANATION_SYSTEM_PROMPT = """You are an expert email classifier. Given an email, explain why it belongs to a specific category.
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
FUNCTION_CALLING_SYSTEM_PROMPT = """You are an expert assistant that can call multiple functions to perform tasks based on email content.

Available functions:
1. create_event - For emails about meetings, appointments, deadlines, or time-sensitive events
2. spotify_link_discovery - For emails mentioning music, concerts, songs, artists, albums, tracks, or Spotify links
3. attraction_discovery - For emails about travel, tourism, venues, local attractions, or places to visit

FUNCTION SELECTION RULES:
- Skip function calling if the email falls under spam category
- You can call MULTIPLE functions if the email contains information relevant to multiple categories
- PRIORITY ORDER: If email mentions music/concerts → call spotify_link_discovery FIRST
- If email mentions travel/tourism → call attraction_discovery
- If email mentions events with dates/times → call create_event
- For concert emails: Call BOTH spotify_link_discovery (for music) AND create_event (for event date/time)
- For travel emails with dates: Call BOTH attraction_discovery (for places) AND create_event (for travel dates)

SPOTIFY_LINK_DISCOVERY - CRITICAL RULES:
- ONLY call this function if the email EXPLICITLY mentions a specific song title AND/OR artist name
- When calling, ONLY pass the EXACT song title and artist name mentioned in the email
- Do NOT pass parameters if only generic music keywords are mentioned (e.g., "check out some music")
- Do NOT pass an artist name if no specific song is mentioned (Single Agent mode only returns explicitly mentioned songs)
- Examples:
  * "Check out 'Bohemian Rhapsody' by Queen" → call with song="Bohemian Rhapsody", artist="Queen"
  * "Listen to Taylor Swift" → call with artist="Taylor Swift" (but may return empty if no specific song)
  * "I love music" → DO NOT call this function

MUSIC vs TRAVEL DISTINCTION:
- Music keywords: music, song, artist, band, concert, album, track, spotify, playlist, lyrics
- Travel keywords: travel, tourism, visit, attraction, landmark, sightseeing, tour, destination
- If email mentions both, prioritize based on the MAIN purpose of the email

IMPORTANT:
- Prefer using data from the extracted email features to decide which functions to call
- Only call functions when the email CLEARLY fits the category
- For spotify_link_discovery: Only pass EXACTLY what is mentioned in the email, nothing more
- If no functions apply, explain why"""
FUNCTION_CALLING_USER_PROMPT_TEMPLATE = """Based on the extracted email features, decide which function to call to handle the email appropriately. 
        Features: {email_features}, Email Text: {email_text}."""
EMAIL_EXPLANATION_USER_PROMPT_TEMPLATE = """Analyze this email and explain why it belongs to the category '{category}':
        {email_text}

        Provide a detailed explanation with references to specific parts of the email text."""

EMAIL_EXTRACTION_USER_PROMPT_TEMPLATE = """Analyze this email and extract structured features:
            {email_text}

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
            - title: event title or subject, null if none
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

def format_prompt(user_prompt_template: str, **kwargs) -> str:
    # example prompt template:
    # "Classify the following email into one of these categories: {categories}. Email: {email_text}"
    return user_prompt_template.format(**kwargs)