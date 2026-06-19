from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GENERATION_MODEL

def generate_adaptive_response(user_query: str, persona: str, context_chunks: list) -> str:
    """Assembles prompt variations to generate highly tailored responses perfectly matched to the persona tone."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    if persona == "Technical Expert":
        persona_instructions = (
            "You are a Senior Systems Engineer. Provide a thorough root-cause analysis, system configuration "
            "specifications, and precise API pathways or code blocks. Keep parameters highly exact, "
            "structured, and systematically deep."
        )
    elif persona == "Frustrated User":
        persona_instructions = (
            "You are an incredibly empathetic Customer Care Specialist. Begin with warm, explicit validation "
            "of their stressful inconvenience. Use ultra-simple, reassuring, action-oriented bulleted steps. "
            "Completely avoid technical engineering jargon."
        )
    else:  # Business Executive
        persona_instructions = (
            "You are a brief, punchy Client Relations Director. Focus solely on target business outcomes, "
            "high-level operational impact summaries, and strict resolution timelines. Keep answers exceptionally concise, "
            "skipping configurations or background code parameters."
        )
        
    context_text = "\n\n".join([f"Source [{c['source']}]: {c['text']}" for c in context_chunks])
    
    full_system_prompt = (
        f"{persona_instructions}\n\n"
        "CRITICAL RULES:\n"
        "- Ground your answer ONLY on the factual text snippets provided in the context below.\n"
        "- Do not extrapolate, infer, assume, or hallucinate facts not directly explicitly stated.\n"
        "- If context matches are insufficient to answer, report transparently that information is missing.\n\n"
        f"FACTUAL CONTEXT DOCUMENTS:\n{context_text}"
    )
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            temperature=0.2
        )
    )
    return response.text