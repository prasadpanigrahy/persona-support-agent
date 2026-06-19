import json
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GENERATION_MODEL

def classify_customer_persona(user_message: str) -> dict:
    """Classifies incoming user support requests into target personas using strict JSON schemas."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    system_instruction = (
        "You are an advanced classification engine. Your task is to analyze the sentiment, vocabulary, "
        "and tone of an incoming support message and classify it into exactly one of three customer personas:\n"
        "1. 'Technical Expert': Uses jargon, asks about APIs, header tokens, configs, logs, or errors.\n"
        "2. 'Frustrated User': Uses emotional language, exclamation marks, caps, or mentions urgency/complaints.\n"
        "3. 'Business Executive': Focuses on business operational impact, ROI, brevity, and resolution timelines.\n\n"
        "Provide your evaluation strictly in the requested JSON structure."
    )
    
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "persona": {
                "type": "STRING", 
                "enum": ["Technical Expert", "Frustrated User", "Business Executive"]
            },
            "confidence": {"type": "NUMBER"},
            "reasoning": {"type": "STRING"}
        },
        "required": ["persona", "confidence", "reasoning"]
    }
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1
        )
    )
    return json.loads(response.text)