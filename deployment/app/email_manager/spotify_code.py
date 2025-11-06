import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from openai import OpenAI

# Load environment variables
# Try loading .env first, then spotify.env (spotify.env will override .env values)
_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()  # Load .env if exists
spotify_env_path = os.path.join(_APP_DIR, "spotify.env")
if os.path.exists(spotify_env_path):
    load_dotenv(spotify_env_path, override=True)  # Override with spotify.env values

# Constants
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Initialize OpenAI client
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Spotify client
spotify = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        spotify = spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        print(f"⚠️ Failed to initialize Spotify client: {e}")


def parse_song_input(text: str) -> Dict[str, Optional[str]]:
    """
    Extract song title and artist from text using OpenAI.
    Returns dict with 'title' and 'artist' fields.
    title=None if no song mentioned.
    """
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    
    prompt = f"""
    Extract the song title and artist from this text.
    Respond ONLY as JSON with fields "title" and "artist".
    If no song mentioned, set title to null.
    If no artist mentioned, set artist to null.

    Text: "{text}"
    """
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if any
        if content.startswith("```"):
            content = content.strip("`").replace("json", "").strip()
            if content.endswith("```"):
                content = content[:-3].strip()

        return json.loads(content)
    except Exception as e:
        raise ValueError(f"Failed to parse song input: {e}")


def search_spotify_song(title: str, artist: str) -> Optional[Dict[str, Any]]:
    """Search a song by title + artist on Spotify"""
    if not spotify:
        raise ValueError("Spotify client not initialized. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")
    
    if not title or not artist:
        return None
    
    try:
        query = f"track:{title} artist:{artist}"
        results = spotify.search(q=query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if not items:
            return None
        
        track = items[0]
        return {
            "name": track["name"],
            "artist": ", ".join([a["name"] for a in track["artists"]]),
            "album": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track.get("preview_url"),
            "artist_id": track["artists"][0]["id"] if track["artists"] else None
        }
    except Exception as e:
        print(f"⚠️ Error searching Spotify: {e}")
        return None


def latest_songs_by_artist(artist: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get latest songs by artist from Spotify"""
    if not spotify:
        raise ValueError("Spotify client not initialized. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")
    
    if not artist:
        return []
    
    try:
        results = spotify.search(q=f"artist:{artist}", type="track", limit=50)
        tracks = results.get("tracks", {}).get("items", [])
        tracks_sorted = sorted(tracks, key=lambda t: t['album']['release_date'], reverse=True)
        
        latest = []
        seen_titles = set()
        for t in tracks_sorted:
            if t['name'] in seen_titles:
                continue
            seen_titles.add(t['name'])
            latest.append({
                "name": t["name"],
                "artist": ", ".join([a["name"] for a in t["artists"]]),
                "album": t["album"]["name"],
                "release_date": t["album"]["release_date"],
                "spotify_url": t["external_urls"]["spotify"],
                "preview_url": t.get("preview_url")
            })
            if len(latest) >= limit:
                break
        return latest
    except Exception as e:
        print(f"⚠️ Error getting latest songs: {e}")
        return []


def get_artist_description(artist_id: str) -> str:
    """Fetch artist info from Spotify and summarize via OpenAI"""
    if not spotify:
        raise ValueError("Spotify client not initialized.")
    if not client:
        raise ValueError("OpenAI client not initialized.")
    
    try:
        artist = spotify.artist(artist_id)
        bio_prompt = f"Give a short description (2-3 sentences) of the artist {artist['name']} based on this info: {artist.get('genres', [])}."
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": bio_prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error getting artist description: {e}")
        return ""


def get_song_description(song_name: str, artist_name: str) -> str:
    """Generate brief song description via OpenAI"""
    if not client:
        raise ValueError("OpenAI client not initialized.")
    
    try:
        prompt = f"Write a short description (2-3 sentences) of the song '{song_name}' by {artist_name}."
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error getting song description: {e}")
        return ""


class SpotifyFunction:
    """Handler for Spotify-related operations."""
    
    def __init__(self, artist: Optional[str] = None, song: Optional[str] = None, email_text: Optional[str] = None):
        self.artist = artist
        self.song = song
        self.email_text = email_text or ""
    
    def discover_spotify_links(self) -> Dict[str, Any]:
        """
        Discover Spotify links based on provided artist and song, or parse from email text.
        Returns formatted result compatible with FunctionCall expectations.
        """
        songs = []
        
        try:
            # If song and artist are provided, search directly
            if self.song and self.artist:
                track = search_spotify_song(self.song, self.artist)
                if track:
                    songs.append({
                        "song": track["name"],
                        "artist": track["artist"],
                        "spotify_url": track["spotify_url"],
                        "album": track.get("album"),
                        "release_date": track.get("release_date"),
                        "preview_url": track.get("preview_url")
                    })
            
            # If only artist is provided, get latest songs
            elif self.artist and not self.song:
                latest = latest_songs_by_artist(self.artist, limit=5)
                for track in latest:
                    songs.append({
                        "song": track["name"],
                        "artist": track["artist"],
                        "spotify_url": track["spotify_url"],
                        "album": track.get("album"),
                        "release_date": track.get("release_date"),
                        "preview_url": track.get("preview_url")
                    })
            
            # If we have email text but no explicit song/artist, try to parse
            elif self.email_text and not self.song and not self.artist:
                song_info = parse_song_input(self.email_text)
                if song_info.get("title"):
                    # We have a song title
                    artist = song_info.get("artist")
                    if artist:
                        track = search_spotify_song(song_info["title"], artist)
                        if track:
                            songs.append({
                                "song": track["name"],
                                "artist": track["artist"],
                                "spotify_url": track["spotify_url"],
                                "album": track.get("album"),
                                "release_date": track.get("release_date"),
                                "preview_url": track.get("preview_url")
                            })
                elif song_info.get("artist"):
                    # Only artist, no song
                    latest = latest_songs_by_artist(song_info["artist"], limit=5)
                    for track in latest:
                        songs.append({
                            "song": track["name"],
                            "artist": track["artist"],
                            "spotify_url": track["spotify_url"],
                            "album": track.get("album"),
                            "release_date": track.get("release_date"),
                            "preview_url": track.get("preview_url")
                        })
            
            # Format response
            if songs:
                return {
                    "message": f"Successfully discovered {len(songs)} Spotify link(s)",
                    "success": True,
                    "data": {
                        "songs": songs
                    }
                }
            else:
                return {
                    "message": "No Spotify links found. Please check the artist and song names.",
                    "success": False,
                    "data": {
                        "songs": []
                    }
                }
        
        except ValueError as e:
            # Configuration errors
            return {
                "message": f"Configuration error: {str(e)}. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
                "success": False,
                "data": {
                    "songs": []
                }
            }
        except Exception as e:
            # Other errors
            error_msg = f"Failed to discover Spotify links: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "message": error_msg,
                "success": False,
                "data": {
                    "songs": []
                }
            }
