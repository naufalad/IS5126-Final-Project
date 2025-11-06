import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_attractions_with_maps(destination, limit=3):
    """
    Get attractions for a specific destination.
    Optimized for Single Agent mode - only extracts attractions for the mentioned location.
    """
    prompt = f"""List exactly {limit} top tourist attractions in "{destination}".

For each attraction, provide:
- name: Full name of the attraction
- description: Brief 2-sentence description
- fun_fact: One interesting fact
- map_link: Google Maps search URL (format: https://www.google.com/maps/search/<attraction_name>+{destination.replace(' ','+')})

Return ONLY a valid JSON array. No explanations, no markdown.
Example format:
[
  {{"name": "Attraction Name", "description": "...", "fun_fact": "...", "map_link": "https://..."}},
  ...
]
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800  # Limit tokens for faster response
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        
        import json
        attractions = json.loads(content)
        
        # Ensure it's a list and limit the count
        if not isinstance(attractions, list):
            attractions = [attractions] if attractions else []
        
        return attractions[:limit]
        
    except Exception as e:
        print(f"⚠️ Error getting attractions for {destination}: {e}")
        # Return empty list on error
        return []