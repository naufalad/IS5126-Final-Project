"""
Multi-Agent system for attraction discovery using CrewAI.
Two agents:
1. Extractor Agent: Extracts direct match attractions for mentioned location
2. Recommendation Agent: Determines if recommendations are needed and generates them
"""
import os
import json
import time
from typing import Dict, Any, List
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from openai import OpenAI
from classes.EmailFeatures import EmailFeatures

load_dotenv()

OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Lazy initialization for OpenAI client
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client instance (lazy initialization)"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# Lazy initialization for CrewAI LLM
_crewai_llm = None

def get_crewai_llm():
    """Get or create CrewAI LLM instance (lazy initialization)"""
    global _crewai_llm
    if _crewai_llm is None:
        _crewai_llm = LLM(
            model=f"openai/{OPENAI_MODEL_NAME}",
            temperature=0.1,
            max_tokens=1000  # Reduced from 1500 for faster responses
        )
    return _crewai_llm


# ============================================================================
# LOCATION EXTRACTION FROM EMAIL
# ============================================================================

def extract_mentioned_locations_from_email(email_text: str) -> List[str]:
    """
    Extract explicitly mentioned locations from email text using OpenAI.
    Returns a list of location names that are explicitly mentioned in the email.
    """
    prompt = f"""Extract ALL locations that are EXPLICITLY MENTIONED in the following email.

Email:
{email_text}

Your task:
1. Identify ALL locations (cities, places, attractions, addresses) that are EXPLICITLY mentioned in the email
2. Return ONLY locations that are clearly stated in the email text
3. DO NOT infer or guess locations - only extract what is explicitly written
4. If a location is mentioned with quotes like "Marina Bay Sands, Singapore", extract it as "Marina Bay Sands, Singapore" (keep the FULL name, do NOT split it)
5. DO NOT split composite locations - if email mentions "Marina Bay Sands, Singapore", return it as one location "Marina Bay Sands, Singapore", NOT as separate "Marina Bay Sands" and "Singapore"
6. If multiple locations are mentioned, extract all of them
7. Return locations in the EXACT format as they appear in the email

CRITICAL RULES:
- Keep composite locations as ONE entry (e.g., "Marina Bay Sands, Singapore" stays as one location)
- DO NOT split locations by commas if they are part of a single location name
- DO NOT extract broader locations if a specific location is mentioned (e.g., if "Marina Bay Sands, Singapore" is mentioned, do NOT also extract "Singapore")
- Only extract what is EXPLICITLY written in the email

EXAMPLE:
Email: 'The location I'm thinking about is "Marina Bay Sands, Singapore".'
CORRECT: ["Marina Bay Sands, Singapore"]
WRONG: ["Marina Bay Sands, Singapore", "Singapore"] or ["Marina Bay Sands", "Singapore"]

Return ONLY a valid JSON array of location strings. No explanations, no markdown.
Example format:
["Location 1", "Location 2", "Location 3"]

If no locations are mentioned, return an empty array: []
"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        
        locations = json.loads(content)
        if not isinstance(locations, list):
            locations = [locations] if locations else []
        
        # Filter out empty strings and normalize
        locations = [loc.strip() for loc in locations if loc and loc.strip()]
        
        print(f"📍 Extracted mentioned locations from email: {locations}")
        return locations
        
    except Exception as e:
        print(f"⚠️ Error extracting locations from email: {e}")
        # Fallback: return empty list
        return []

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def get_extractor_agent():
    """Get or create extractor agent (lazy initialization)"""
    return Agent(
        role="Attraction Extractor",
        goal="Extract attractions ONLY for locations that are EXPLICITLY MENTIONED in the email text. Do not infer or add locations that are not mentioned.",
        backstory="Expert at finding tourist attractions for locations. You are very strict about only extracting attractions for locations that are explicitly stated in the email. You never add locations that are not mentioned, even if they are related or nearby. You always verify that a location appears in the email text before extracting attractions for it.",
        verbose=False,
        allow_delegation=False,  # Disabled delegation for faster processing
        llm=get_crewai_llm(),
    )

def get_recommendation_agent():
    """Get or create recommendation agent (lazy initialization)"""
    return Agent(
        role="Location Recommendation Generator",
        goal="Generate 3 related location recommendations based on email content",
        backstory="Expert at understanding travel context and generating relevant location recommendations. You analyze email content and suggest 3 related locations that would be interesting for the user.",
        verbose=False,
        allow_delegation=False,
        llm=get_crewai_llm(),
    )

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_extraction_task(locations: list, email_text: str) -> Task:
    """Create task for extracting direct match attractions for mentioned locations"""
    locations_str = ", ".join(locations)
    # Limit: If multiple locations, return 1-2 per location, max 3 total
    # If single location, return max 3 attractions
    max_attractions = 3 if len(locations) == 1 else min(3, len(locations) * 2)
    
    return Task(
        description=f"""
        CRITICAL REQUIREMENT: Extract tourist attractions ONLY for locations that are EXPLICITLY MENTIONED in the email.
        
        Locations explicitly mentioned in the email: {locations_str}
        
        Email context: {email_text[:800]}
        
        EXAMPLE - CORRECT BEHAVIOR:
        Email: "I'm planning to visit some attractions this weekend. The location I'm thinking about is 'Marina Bay Sands, Singapore'. Could you help find some nearby attractions and show me the Google Maps directions? Looking for iconic spots and a short description for each. Thanks!"
        
        CORRECT Response: Extract attractions ONLY for "Marina Bay Sands, Singapore" because this is the ONLY location explicitly mentioned in the email.
        WRONG Response: Do NOT extract attractions for "Singapore" separately, or "Marina Bay", or any other location that is not explicitly stated.
        
        Your task:
        1. Carefully read the email text above and identify which locations from the list "{locations_str}" are ACTUALLY MENTIONED in the email
        2. For EACH location that is EXPLICITLY MENTIONED in the email text, find its top attractions
        3. DO NOT include locations that are NOT mentioned in the email, even if:
           - They are related or nearby to mentioned locations
           - They are part of a larger area (e.g., if email mentions "Marina Bay Sands, Singapore", do NOT extract for "Singapore" separately)
           - They are commonly associated with the mentioned location
        4. DO NOT infer or guess locations - only use locations that are clearly stated in the email text
        5. If the email mentions a specific place like "Marina Bay Sands, Singapore", extract attractions for that specific place, NOT for the broader location "Singapore"
        6. If multiple locations are mentioned, extract 1-2 attractions per location
        7. Return EXACTLY {max_attractions} attractions total (prioritize the most important/popular ones)
        8. For each attraction, provide:
           - name: Full name of the attraction
           - description: Short description (2-3 sentences)
           - fun_fact: An interesting fact about the attraction
           - location: The EXACT location name as mentioned in the email (use the same spelling/format)
        
        Note: Google Maps links will be generated automatically, you don't need to provide them.
        
        Return ONLY a valid JSON array with EXACTLY {max_attractions} attractions:
        [
            {{
                "name": "Attraction Name",
                "description": "Description here",
                "fun_fact": "Fun fact here",
                "location": "Exact location name from email"
            }}
        ]
        
        STRICT RULES:
        - ONLY extract attractions for locations that appear EXPLICITLY in the email text
        - DO NOT add locations that are not mentioned, even if they are related or nearby
        - If email mentions "Marina Bay Sands, Singapore", extract ONLY for "Marina Bay Sands, Singapore", NOT for "Singapore" separately
        - Use the EXACT location name as it appears in the email (same spelling, same format)
        - Return EXACTLY {max_attractions} attractions, no more, no less
        - If a location is not mentioned in the email, do NOT include attractions for it
        - Verify each location appears in the email text before extracting attractions for it
        """,
        agent=get_extractor_agent(),
        expected_output=f"JSON array with exactly {max_attractions} attractions for locations explicitly mentioned in the email"
    )


def create_recommendation_location_task(email_text: str, mentioned_locations: list) -> Task:
    """Create task for generating 3 recommended locations based on email content"""
    mentioned_str = ", ".join(mentioned_locations) if mentioned_locations else "None"
    
    return Task(
        description=f"""
        Analyze this email and generate 3 location recommendations based on the email content.
        
        Email: {email_text[:500]}
        Mentioned locations in email: {mentioned_str}
        
        Your task:
        1. Analyze the email content to understand the travel context, interests, or purpose
        2. Generate EXACTLY 3 location names that are related, nearby, or would be interesting based on the email
        3. These should be different from the mentioned locations
        4. Consider the travel theme, interests, or context mentioned in the email
        
        Return ONLY a valid JSON array with EXACTLY 3 location names:
        ["Location 1", "Location 2", "Location 3"]
        
        Examples:
        - If email mentions "Singapore", you might recommend: ["Malaysia", "Thailand", "Indonesia"]
        - If email mentions "Paris", you might recommend: ["Lyon", "Nice", "Marseille"]
        - If email mentions "Tokyo", you might recommend: ["Kyoto", "Osaka", "Yokohama"]
        """,
        agent=get_recommendation_agent(),
        expected_output="JSON array with exactly 3 location names"
    )


def create_recommendation_extraction_task(recommended_locations: list) -> Task:
    """Create task for extracting attractions from recommended locations - one attraction per location"""
    locations_str = ", ".join(recommended_locations)
    
    return Task(
        description=f"""
        Extract EXACTLY ONE top attraction for each of these recommended locations: {locations_str}
        
        Your task:
        - For EACH location, find the MOST POPULAR or MOST INTERESTING attraction
        - Return EXACTLY 3 attractions (one per location)
        - For each attraction, provide:
           - name: Full name of the attraction
           - description: Short description (2-3 sentences)
           - fun_fact: An interesting fact about the attraction
           - location: The location name this attraction belongs to
        
        Note: Google Maps links will be generated automatically, you don't need to provide them.
        
        Return ONLY a valid JSON array with EXACTLY 3 attractions:
        [
            {{
                "name": "Attraction Name",
                "description": "Description here",
                "fun_fact": "Fun fact here",
                "location": "Location name"
            }},
            ...
        ]
        
        IMPORTANT: Return exactly 3 attractions, one for each recommended location.
        """,
        agent=get_extractor_agent(),
        expected_output="JSON array with exactly 3 attractions, one per recommended location"
    )


# ============================================================================
# MULTI-AGENT PROCESSOR
# ============================================================================

def process_attraction_discovery_multi_agent(email_features: EmailFeatures, email_text: str, timeout_seconds: int = 90) -> Dict[str, Any]:
    """
    Process attraction discovery using multiple agents.
    
    New Flow:
    1. Agent 1 (Extractor): Extract attractions for ALL locations mentioned in the email
    2. Agent 2 (Recommendation): Generate 3 recommended locations based on email content
    3. Agent 1 (Extractor): Extract 1 attraction per recommended location (total 3)
    """
    start_time = time.time()
    
    def check_timeout():
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Attraction discovery timed out after {timeout_seconds}s")
    
    try:
        # Extract mentioned locations from email using OpenAI
        print("🔍 Extracting mentioned locations from email text...")
        mentioned_locations = extract_mentioned_locations_from_email(email_text)
        
        # If no locations found, try fallback to email_features.location
        if not mentioned_locations:
            if email_features.location and email_features.location != "Unknown Location":
                # Split by common separators if multiple locations are in one string
                location_str = email_features.location
                import re
                locations = re.split(r'[,;]|\s+and\s+|\s+&\s+', location_str)
                mentioned_locations = [loc.strip() for loc in locations if loc.strip()]
        
        # Final fallback
        if not mentioned_locations:
            location = email_features.location or "Unknown Location"
            if location and location != "Unknown Location":
                mentioned_locations = [location]
        
        if not mentioned_locations or (len(mentioned_locations) == 1 and mentioned_locations[0] == "Unknown Location"):
            return {
                "message": "No location provided. Cannot discover attractions.",
                "success": False,
                "data": {
                    "direct_match": [],
                    "recommendations": []
                }
            }
        
        print(f"🎭 Multi-Agent: Starting attraction discovery for mentioned locations: {mentioned_locations} (timeout: {timeout_seconds}s)")
        
        # Step 1: Use Single Agent code to extract attractions for mentioned locations (same as Single Agent mode)
        check_timeout()
        print(f"🔍 Using Single Agent method to extract attractions for mentioned locations: {mentioned_locations}...")
        step1_start = time.time()
        
        # Import Single Agent function
        from email_manager.flights_code import get_attractions_with_maps
        
        # Extract attractions for each mentioned location using Single Agent method (same logic as Single Agent mode)
        direct_match = []
        max_direct = 3  # Limit direct match to 3 attractions total
        
        # Process the first mentioned location (or combine if multiple)
        # For simplicity, use the first location or combine them
        if mentioned_locations:
            # Use the first location (or combine locations if needed)
            location = mentioned_locations[0]
            if len(mentioned_locations) > 1:
                # If multiple locations, use the first one for now
                print(f"  📍 Multiple locations found, using first location: {location}")
            
            print(f"  📍 Processing location (EXACT as extracted from email): '{location}'")
            print(f"  📍 This location was extracted from email text, using it EXACTLY as is")
            
            # Use Single Agent's get_attractions_with_maps function (same as Single Agent mode)
            # This function will extract attractions AT the specific location
            attractions_data = get_attractions_with_maps(destination=location, limit=max_direct)
            
            print(f"  📍 Received {len(attractions_data)} attraction(s) from Single Agent function")
            
            # Format attractions (same format as Single Agent)
            for attr in attractions_data:
                if len(direct_match) >= max_direct:
                    break
                attr_name = attr.get("name", "Unknown Attraction")
                map_link = attr.get("map_link", "")
                if not map_link or not map_link.startswith("http"):
                    # Generate Google Maps search URL
                    search_query = f"{attr_name.replace(' ', '+')}+{location.replace(' ', '+')}"
                    map_link = f"https://www.google.com/maps/search/{search_query}"
                
                print(f"    ✅ Adding attraction: {attr_name} (location: {location})")
                direct_match.append({
                    "name": attr_name,
                    "location": location,  # Use the EXACT mentioned location from email
                    "type": "general",
                    "description": attr.get("description", ""),
                    "map_link": map_link,
                    "fun_fact": attr.get("fun_fact", "")
                })
        
        step1_elapsed = time.time() - step1_start
        print(f"✅ Single Agent extraction completed in {step1_elapsed:.2f}s")
        print(f"✅ Found {len(direct_match)} attraction(s) for mentioned locations: {mentioned_locations}")
        
        # Step 2: Agent 2 - Generate 3 recommended locations
        check_timeout()
        print("🤖 Agent 2: Generating 3 recommended locations...")
        step2_start = time.time()
        recommendation_location_task = create_recommendation_location_task(email_text, mentioned_locations)
        recommendation_result = Crew(
            agents=[get_recommendation_agent()],
            tasks=[recommendation_location_task],
            verbose=False
        ).kickoff()
        step2_elapsed = time.time() - step2_start
        print(f"✅ Agent 2 completed in {step2_elapsed:.2f}s")
        
        # Parse recommendation result
        recommendation_str = str(recommendation_result)
        if "```json" in recommendation_str:
            recommendation_str = recommendation_str.split("```json")[1].split("```")[0].strip()
        elif "```" in recommendation_str:
            recommendation_str = recommendation_str.split("```")[1].split("```")[0].strip()
        
        try:
            recommended_locations = json.loads(recommendation_str)
            if not isinstance(recommended_locations, list):
                recommended_locations = []
            # Ensure we have exactly 3 locations
            recommended_locations = recommended_locations[:3]
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse recommendation locations: {recommendation_str[:200]}")
            recommended_locations = []
        
        print(f"✅ Agent 2: Generated {len(recommended_locations)} recommended location(s): {recommended_locations}")
        
        # Step 3: Agent 1 - Extract 1 attraction per recommended location (total 3)
        recommendations = []
        if recommended_locations:
            check_timeout()
            print(f"🤖 Agent 1: Extracting 1 attraction per recommended location (total {len(recommended_locations)})...")
            step3_start = time.time()
            recommendation_extraction_task = create_recommendation_extraction_task(recommended_locations)
            rec_extraction_result = Crew(
                agents=[get_extractor_agent()],
                tasks=[recommendation_extraction_task],
                verbose=False
            ).kickoff()
            step3_elapsed = time.time() - step3_start
            print(f"✅ Agent 1 (recommendations) completed in {step3_elapsed:.2f}s")
            
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
            
            # Format recommended attractions (exactly 3, one per location)
            for attr in recommended_attractions[:3]:  # Limit to 3
                attr_name = attr.get("name", "Unknown Attraction")
                attr_location = attr.get("location", recommended_locations[0] if recommended_locations else "Unknown")
                
                # Generate Google Maps link if not provided or invalid
                map_link = attr.get("map_link", "")
                if not map_link or not map_link.startswith("http"):
                    # Generate Google Maps search URL
                    search_query = f"{attr_name.replace(' ', '+')}+{attr_location.replace(' ', '+')}"
                    map_link = f"https://www.google.com/maps/search/{search_query}"
                
                recommendations.append({
                    "name": attr_name,
                    "location": attr_location,
                    "type": "general",
                    "description": attr.get("description", ""),
                    "map_link": map_link,
                    "fun_fact": attr.get("fun_fact", "")
                })
            
            print(f"✅ Agent 1: Found {len(recommendations)} recommended attraction(s)")
        
        total_count = len(direct_match) + len(recommendations)
        total_elapsed = time.time() - start_time
        print(f"✅ Multi-Agent attraction discovery completed in {total_elapsed:.2f}s")
        
        return {
            "message": f"Successfully discovered {total_count} attraction(s) using Multi-Agent system",
            "success": True,
            "data": {
                "direct_match": direct_match,
                "recommendations": recommendations
            },
            "_processing_time": round(total_elapsed, 2)
        }
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        error_msg = f"⏱️ Attraction discovery timed out after {elapsed:.2f}s: {str(e)}"
        print(error_msg)
        # Return partial results if we have direct_match
        return {
            "message": error_msg,
            "success": False,
            "data": {
                "direct_match": direct_match if 'direct_match' in locals() else [],
                "recommendations": recommendations if 'recommendations' in locals() else []
            }
        }
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"❌ Multi-Agent attraction discovery failed after {elapsed:.2f}s: {e}"
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

