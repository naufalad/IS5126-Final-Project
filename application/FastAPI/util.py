
from typing import Optional
from openai import OpenAI
import os

OPENAI_MODEL_NAME = "gpt-4o-mini"

def create_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = create_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content
