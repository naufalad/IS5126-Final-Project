"""
Multi-Agent system for attraction discovery using CrewAI.
Two agents:
1. Extractor Agent: Extracts direct match attractions for mentioned location
2. Recommendation Agent: Determines if recommendations are needed and generates them
"""
import os
import json
import time
from typing import Dict, Any
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from classes.EmailFeatures import EmailFeatures

load_dotenv()

OPENAI_MODEL_NAME = "gpt-4o-mini"

crewai_llm = LLM(
    model=f"openai/{OPENAI_MODEL_NAME}",
    temperature=0.1,
    max_tokens=1500
)

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

extractor_agent = Agent(
    role="Attraction Extractor",
    goal="Extract attractions directly related to the location mentioned in the email",
    backstory="Expert at finding tourist attractions and points of interest for specific locations. You provide accurate information about attractions at the mentioned location.",
    verbose=False,
    allow_delegation=True,
    llm=crewai_llm,
)

recommendation_agent = Agent(
    role="Recommendation Analyzer",
    goal="Analyze email intent and determine if additional attraction recommendations are needed",
    backstory="Expert at understanding user intent from emails. You can determine if someone wants additional recommendations beyond the mentioned location. If recommendations are needed, you generate 5 additional location suggestions.",
    verbose=False,
    allow_delegation=False,
    llm=crewai_llm,
)

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_extraction_task(location: str, email_text: str) -> Task:
    """Create task for extracting direct match attractions"""
    return Task(
        description=f"""
        Extract tourist attractions for the location: {location}
        
        Email context: {email_text[:500]}
        
        Your task:
        1. Find 3 top attractions directly related to "{location}"
        2. For each attraction, provide:
           - name: Full name of the attraction
           - description: Short description (2-3 sentences)
           - fun_fact: An interesting fact about the attraction
           - map_link: Google Maps search URL (format: https://www.google.com/maps/search/<attraction_name>+{location.replace(' ', '+')})
        
        Return ONLY a valid JSON array with this structure:
        [
            {{
                "name": "Attraction Name",
                "description": "Description here",
                "fun_fact": "Fun fact here",
                "map_link": "https://www.google.com/maps/search/..."
            }}
        ]
        """,
        agent=extractor_agent,
        expected_output="JSON array of attractions with name, description, fun_fact, map_link"
    )


def create_recommendation_task(location: str, email_text: str, direct_attractions: list) -> Task:
    """Create task for determining if recommendations are needed and generating them"""
    attractions_summary = ", ".join([a.get("name", "") for a in direct_attractions[:3]])
    
    return Task(
        description=f"""
        Analyze this email and determine if the user wants additional attraction recommendations.
        
        Email: {email_text[:500]}
        Mentioned location: {location}
        Already found attractions: {attractions_summary}
        
        Your task:
        1. Determine if the email requests recommendations (keywords: recommend, suggest, help find, nearby, other, more, etc.)
        2. If YES, generate 5 additional location names that are related to or near "{location}"
        3. If NO, return empty list
        
        Return ONLY a valid JSON object with this structure:
        {{
            "needs_recommendations": true/false,
            "reasoning": "Brief explanation",
            "recommended_locations": ["Location 1", "Location 2", ...] (max 5, only if needs_recommendations is true)
        }}
        """,
        agent=recommendation_agent,
        expected_output="JSON object with needs_recommendations boolean and recommended_locations array"
    )


def create_recommendation_extraction_task(recommended_locations: list, original_location: str) -> Task:
    """Create task for extracting attractions from recommended locations"""
    locations_str = ", ".join(recommended_locations)
    
    return Task(
        description=f"""
        Extract attractions for these recommended locations: {locations_str}
        
        Original location context: {original_location}
        
        Your task:
        For each recommended location, find 1-2 top attractions. Total should be around 5 attractions.
        
        Return ONLY a valid JSON array with this structure:
        [
            {{
                "name": "Attraction Name",
                "description": "Description here",
                "fun_fact": "Fun fact here",
                "map_link": "https://www.google.com/maps/search/...",
                "location": "Location name"
            }}
        ]
        
        Ensure variety - avoid duplicates. Focus on diverse, interesting attractions.
        """,
        agent=extractor_agent,
        expected_output="JSON array of recommended attractions"
    )


# ============================================================================
# MULTI-AGENT PROCESSOR
# ============================================================================

def process_attraction_discovery_multi_agent(email_features: EmailFeatures, email_text: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Process attraction discovery using multiple agents.
    
    Flow:
    1. Extractor Agent: Extract direct match attractions (3 attractions)
    2. Recommendation Agent: Determine if recommendations needed
    3. If needed: Extractor Agent: Extract attractions from recommended locations (5 attractions)
    """
    try:
        location = email_features.location or "Unknown Location"
        
        if not location or location == "Unknown Location":
            return {
                "message": "No location provided. Cannot discover attractions.",
                "success": False,
                "data": {
                    "direct_match": [],
                    "recommendations": []
                }
            }
        
        print(f"🎭 Multi-Agent: Starting attraction discovery for {location}")
        
        # Step 1: Extract direct match attractions
        print("🤖 Agent 1: Extracting direct match attractions...")
        extraction_task = create_extraction_task(location, email_text)
        extraction_result = Crew(
            agents=[extractor_agent],
            tasks=[extraction_task],
            verbose=False
        ).kickoff()
        
        # Parse extraction result
        extraction_str = str(extraction_result)
        if "```json" in extraction_str:
            extraction_str = extraction_str.split("```json")[1].split("```")[0].strip()
        elif "```" in extraction_str:
            extraction_str = extraction_str.split("```")[1].split("```")[0].strip()
        
        try:
            direct_attractions = json.loads(extraction_str)
            if not isinstance(direct_attractions, list):
                direct_attractions = []
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse extraction result: {extraction_str[:200]}")
            direct_attractions = []
        
        # Format direct match attractions
        direct_match = []
        for attr in direct_attractions[:3]:  # Limit to 3
            direct_match.append({
                "name": attr.get("name", "Unknown Attraction"),
                "location": location,
                "type": "general",
                "description": attr.get("description", ""),
                "map_link": attr.get("map_link", ""),
                "fun_fact": attr.get("fun_fact", "")
            })
        
        print(f"✅ Agent 1: Found {len(direct_match)} direct match attraction(s)")
        
        # Step 2: Determine if recommendations are needed
        print("🤖 Agent 2: Analyzing recommendation needs...")
        recommendation_task = create_recommendation_task(location, email_text, direct_match)
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
            recommendation_analysis = json.loads(recommendation_str)
            needs_recommendations = recommendation_analysis.get("needs_recommendations", False)
            recommended_locations = recommendation_analysis.get("recommended_locations", [])
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse recommendation analysis: {recommendation_str[:200]}")
            needs_recommendations = False
            recommended_locations = []
        
        print(f"✅ Agent 2: Recommendations needed: {needs_recommendations}")
        
        # Step 3: Extract recommended attractions if needed
        recommendations = []
        if needs_recommendations and recommended_locations:
            print(f"🤖 Agent 1: Extracting attractions from {len(recommended_locations)} recommended locations...")
            recommendation_extraction_task = create_recommendation_extraction_task(recommended_locations, location)
            rec_extraction_result = Crew(
                agents=[extractor_agent],
                tasks=[recommendation_extraction_task],
                verbose=False
            ).kickoff()
            
            # Parse recommendation extraction result
            rec_extraction_str = str(rec_extraction_result)
            if "```json" in rec_extraction_str:
                rec_extraction_str = rec_extraction_str.split("```json")[1].split("```")[0].strip()
            elif "```" in rec_extraction_str:
                rec_extraction_str = rec_extraction_str.split("```")[1].split("```")[0].strip()
            
            try:
                recommended_attractions = json.loads(rec_extraction_str)
                if not isinstance(recommended_attractions, list):
                    recommended_attractions = []
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse recommendation extraction: {rec_extraction_str[:200]}")
                recommended_attractions = []
            
            # Format recommended attractions
            for attr in recommended_attractions[:5]:  # Limit to 5
                # Avoid duplicates with direct_match
                if attr.get("name") not in [a["name"] for a in direct_match]:
                    recommendations.append({
                        "name": attr.get("name", "Unknown Attraction"),
                        "location": attr.get("location", location),
                        "type": "general",
                        "description": attr.get("description", ""),
                        "map_link": attr.get("map_link", ""),
                        "fun_fact": attr.get("fun_fact", "")
                    })
            
            print(f"✅ Agent 1: Found {len(recommendations)} recommended attraction(s)")
        
        total_count = len(direct_match) + len(recommendations)
        return {
            "message": f"Successfully discovered {total_count} attraction(s) using Multi-Agent system",
            "success": True,
            "data": {
                "direct_match": direct_match,
                "recommendations": recommendations
            }
        }
        
    except Exception as e:
        error_msg = f"❌ Multi-Agent attraction discovery failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "message": error_msg,
            "success": False,
            "data": {
                "direct_match": [],
                "recommendations": []
            }
        }

