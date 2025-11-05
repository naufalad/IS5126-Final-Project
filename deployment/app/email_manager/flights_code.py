import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def parse_destination_input(text: str):
    """
    Extract destination using OpenAI.
    title=None if missing.
    """
    prompt = f"""
    Extract destination from this flights email text.
    Respond ONLY as a string.
    If no destination mentioned, set it to null.

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

    return json.loads(content)

def get_attractions_with_maps(destination, limit=3):
    prompt = f"""
    List the top {limit} tourist attractions in {destination}.
    For each attraction, provide:
      - name
      - a short description
      - a fun fact
      - Google Maps link (https://www.google.com/maps/search/<attraction>+{destination.replace(' ','+')})
    Respond ONLY as JSON array of objects with fields:
      name, description, fun_fact, map_link
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json","").strip()
        if content.endswith("```"):
            content = content[:-3].strip()
    return json.loads(content)