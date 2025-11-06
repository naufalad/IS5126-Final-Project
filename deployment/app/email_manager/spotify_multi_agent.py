"""
Multi-Agent system for Spotify recommendations using CrewAI.
Two agents:
1. Music Analyzer Agent: Analyzes email content to understand music preferences
2. Recommendation Agent: Generates "Guess You Like" song recommendations based on analysis
"""
import os
import json
import time
from typing import Dict, Any, List
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_NAME = "gpt-4o-mini"

crewai_llm = LLM(
    model=f"openai/{OPENAI_MODEL_NAME}",
    temperature=0.3,  # Slightly higher for creativity in recommendations
    max_tokens=1500
)

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

music_analyzer_agent = Agent(
    role="Music Preference Analyzer",
    goal="Analyze email content to understand user's music preferences, mood, and interests",
    backstory="Expert at analyzing text to understand music preferences. You can identify genres, moods, artists, and musical themes from email content.",
    verbose=False,
    allow_delegation=False,
    llm=crewai_llm,
)

recommendation_agent = Agent(
    role="Music Recommendation Generator",
    goal="Generate personalized song recommendations based on analyzed preferences",
    backstory="Expert at creating personalized music recommendations. You understand music genres, moods, and can suggest songs that match user preferences. You provide song names and artists without Spotify links.",
    verbose=False,
    allow_delegation=False,
    llm=crewai_llm,
)

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_analysis_task(email_text: str) -> Task:
    """Create task for analyzing music preferences from email"""
    return Task(
        description=f"""
        Analyze this email to understand the user's music preferences:
        
        Email: {email_text[:800]}
        
        Your task:
        1. Identify any mentioned artists, songs, genres, or music-related content
        2. Determine the mood or context (e.g., workout, relaxation, party, study)
        3. Identify musical preferences (genres, styles, eras)
        4. Note any specific music-related requests or interests
        
        Return ONLY a valid JSON object with this structure:
        {{
            "mentioned_artists": ["Artist 1", "Artist 2", ...],
            "mentioned_songs": ["Song 1", "Song 2", ...],
            "genres": ["Genre 1", "Genre 2", ...],
            "mood": "mood description",
            "context": "context description",
            "preferences": "overall music preference summary"
        }}
        """,
        agent=music_analyzer_agent,
        expected_output="JSON object with music preferences analysis"
    )


def create_recommendation_task(analysis_result: dict, email_text: str) -> Task:
    """Create task for generating song recommendations"""
    analysis_str = json.dumps(analysis_result, indent=2)
    
    return Task(
        description=f"""
        Based on the music preference analysis, generate 5-8 personalized song recommendations.
        
        Analysis: {analysis_str}
        Original email context: {email_text[:500]}
        
        Your task:
        1. Generate 5-8 song recommendations that match the user's preferences
        2. Include diverse recommendations (different artists, genres if applicable)
        3. For each song, provide:
           - song_name: Full song title
           - artist_name: Artist or band name
           - reason: Brief explanation why this song matches their preferences (1 sentence)
        
        Return ONLY a valid JSON array with this structure:
        [
            {{
                "song_name": "Song Title",
                "artist_name": "Artist Name",
                "reason": "Why this song matches their preferences"
            }},
            ...
        ]
        
        Do NOT include Spotify links. Only provide song names and artists.
        """,
        agent=recommendation_agent,
        expected_output="JSON array of recommended songs with names, artists, and reasons"
    )


# ============================================================================
# MULTI-AGENT PROCESSOR
# ============================================================================

def process_spotify_recommendations_multi_agent(email_text: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Process Spotify recommendations using multiple agents.
    
    Flow:
    1. Music Analyzer Agent: Analyze email to understand music preferences
    2. Recommendation Agent: Generate "Guess You Like" song recommendations
    """
    try:
        print(f"🎵 Multi-Agent: Starting Spotify recommendation analysis")
        
        # Step 1: Analyze music preferences
        print("🤖 Agent 1: Analyzing music preferences from email...")
        analysis_task = create_analysis_task(email_text)
        analysis_result = Crew(
            agents=[music_analyzer_agent],
            tasks=[analysis_task],
            verbose=False
        ).kickoff()
        
        # Parse analysis result
        analysis_str = str(analysis_result)
        if "```json" in analysis_str:
            analysis_str = analysis_str.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_str:
            analysis_str = analysis_str.split("```")[1].split("```")[0].strip()
        
        try:
            analysis_data = json.loads(analysis_str)
            if not isinstance(analysis_data, dict):
                analysis_data = {}
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse analysis result: {analysis_str[:200]}")
            analysis_data = {}
        
        print(f"✅ Agent 1: Analysis complete")
        
        # Step 2: Generate recommendations
        print("🤖 Agent 2: Generating personalized recommendations...")
        recommendation_task = create_recommendation_task(analysis_data, email_text)
        recommendation_result = Crew(
            agents=[recommendation_agent],
            tasks=[recommendation_task],
            verbose=False
        ).kickoff()
        
        # Parse recommendation result
        recommendation_str = str(recommendation_result)
        if "```json" in recommendation_str:
            recommendation_str = recommendation_str.split("```json")[1].split("```")[0].strip()
        elif "```" in recommendation_str:
            recommendation_str = recommendation_str.split("```")[1].split("```")[0].strip()
        
        try:
            recommendations = json.loads(recommendation_str)
            if not isinstance(recommendations, list):
                recommendations = []
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse recommendations: {recommendation_str[:200]}")
            recommendations = []
        
        print(f"✅ Agent 2: Generated {len(recommendations)} recommendation(s)")
        
        # Format recommendations (no Spotify links, just options)
        formatted_recommendations = []
        for rec in recommendations[:8]:  # Limit to 8
            formatted_recommendations.append({
                "song": rec.get("song_name", "Unknown Song"),
                "artist": rec.get("artist_name", "Unknown Artist"),
                "reason": rec.get("reason", "Recommended based on your preferences"),
                "spotify_url": None,  # No link for Multi Agent recommendations
                "album": None,
                "release_date": None,
                "preview_url": None
            })
        
        return {
            "message": f"Generated {len(formatted_recommendations)} personalized song recommendations based on email content",
            "success": True,
            "data": {
                "songs": formatted_recommendations,
                "analysis": analysis_data,
                "mode": "recommendations"  # Mark as recommendations mode
            }
        }
        
    except Exception as e:
        error_msg = f"❌ Multi-Agent Spotify recommendation failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "message": error_msg,
            "success": False,
            "data": {
                "songs": [],
                "mode": "recommendations"
            }
        }

