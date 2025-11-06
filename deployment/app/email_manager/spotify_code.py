import os
import openai
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv("temp.env")  # loads .env

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(auth_manager=auth_manager)

results = spotify.search(q="track:Blinding Lights artist:The Weeknd", type="track", limit=1)
print(results["tracks"]["items"][0]["external_urls"]["spotify"])


from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example chat call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short description of the song 'Blinding Lights' by The Weeknd."}
    ],
    temperature=0
)

# The assistant reply
content = response.choices[0].message.content
print(content)


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Setup clients
client = OpenAI(api_key=OPENAI_API_KEY)
spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

def parse_song_input(text: str):
    """
    Extract song title and artist using OpenAI.
    title=None if missing.
    """
    prompt = f"""
    Extract the song title and artist from this text.
    Respond ONLY as JSON with fields "title" and "artist".
    If no song mentioned, set title to null.

    Text: "{text}"
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()

    # Strip markdown fences if any
    if content.startswith("```"):
        content = content.strip("`").replace("json", "").strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    import json
    return json.loads(content)

def search_spotify_song(title: str, artist: str):
    """Search a song by title + artist on Spotify"""
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
        "artist_id": track["artists"][0]["id"]
    }

def latest_songs_by_artist(artist: str, limit=5):
    """Get latest songs by artist from Spotify"""
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

def get_artist_description(artist_id):
    """Fetch artist info from Spotify and summarize via OpenAI"""
    artist = spotify.artist(artist_id)
    bio_prompt = f"Give a short description of the artist {artist['name']} based on this info: {artist.get('genres', [])}."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": bio_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def get_song_description(song_name, artist_name):
    """Generate brief song description via OpenAI"""
    prompt = f"Write a short description of the song '{song_name}' by {artist_name}."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()