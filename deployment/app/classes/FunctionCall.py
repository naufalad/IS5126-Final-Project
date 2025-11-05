import os, sys
from typing import Any, Dict, List, Optional

# Add parent directories to path for imports
from classes.EmailFeatures import EmailFeatures

# Import OpenAI client
from openai import OpenAI
import email_manager.calendar_code as calendar
import email_manager.spotify_code as spotify
import email_manager.flights_code as flights

def create_openai_client() -> Optional[OpenAI]:
    """Create and return OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found in environment variables")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to create OpenAI client: {e}")
        return None


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM with system and user prompts."""
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")


class FunctionCall():
    """Handler for executing functions based on email features."""
    
    def __init__(self, email_features: EmailFeatures, email_text: str = ""):
        self.email_features = email_features
        self.email_text = email_text
    
    def create_calendar_event(self) -> Dict[str, Any]:
        """Create a calendar event from email features."""
        try:
            print("📅 Creating calendar event...")
            calendar_function = calendar.CalendarFunction(self.email_features)
            event = calendar_function.save_calendar()
            ics = calendar_function.create_ics()
            return {
                "message": "Calendar event created successfully",
                "success": True,
                "data": {
                    "event": event,
                    "ics_file_path": ics
                }
            }
        except Exception as e:
            error_msg = f"❌ Failed to create calendar event: {e}"
            print(error_msg)
            return {
                "message": error_msg,
                "success": False
            }
    def spotify_link_discovery(self) -> List[Dict[str, Any]]:
        """Discover Spotify links based on email features."""
        # TODO: Implement Spotify API integration
        print("🎵 Spotify link discovery called")
        
        parsed_input = spotify.parse_song_input(self.email_text)
        artist = artist or parsed_input.get("artist")
        song = song or parsed_input.get("title")
        print(f"🔍 Searching for Artist: {artist}, Song: {song}")

        # Call Spotify API to search for the song
        if song:
            track = spotify.search_spotify_song(song, artist)
            if track:
                # Add descriptions
                track["artist_description"] = spotify.get_artist_description(track["artist_id"])
                track["song_description"] = spotify.get_song_description(track["name"], track["artist"])

                print("\n✅ Track info with descriptions:")
                return track
            else:
                print("⚠️ Track not found on Spotify.")
        else:
            print(f"\nℹ️ No song specified. Showing latest songs by {artist}:")
            latest = spotify.latest_songs_by_artist(artist, limit=5)
            return latest
    
    def attraction_discovery(self) -> List[Dict[str, Any]]:
        """Discover attractions based on email features."""
        print("🎭 Attraction discovery called")
        
        # Use location from email features if not provided
        location = flights.parse_destination_input(self.email_text)
        print(f"🔍 Searching for attractions in Location: {location}")
        # Call Flights API to search for attractions
        attractions = flights.get_attractions_with_maps(location)
        if attractions:
            print("\n✅ Attractions found:")
            return attractions
        else:
            print("⚠️ No attractions found.")
            return []
    
    def function_call(self, function: str, **kwargs) -> Any:
        """Wrapper function for calling specific functions."""
        result = None
        
        print(f"🔧 Calling function: {function}")
        
        if function == "create_calendar_event" or function == "create_event":
            result = self.create_calendar_event(**kwargs)
        elif function == "spotify_link_discovery":
            result = self.spotify_link_discovery(**kwargs)
        elif function == "attraction_discovery":
            result = self.attraction_discovery(**kwargs)
        else:
            error_msg = f"Unknown function: {function}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        return result


# ============================================================================
# FUNCTION SCHEMAS FOR OPENAI FUNCTION CALLING
# ============================================================================

create_event_schema = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event based on extracted email features. Use this when an email contains information about meetings, appointments, deadlines, or any time-sensitive events.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

spotify_link_schema = {
    "type": "function",
    "function": {
        "name": "spotify_link_discovery",
        "description": "Discover Spotify links for songs or artists mentioned in the email. Use this when the email mentions music, concerts, or artists.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

attraction_discovery_schema = {
    "type": "function",
    "function": {
        "name": "attraction_discovery",
        "description": "Discover attractions, venues, or points of interest based on location mentioned in the email. Use this when the email mentions travel, tourism, or local attractions.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}
