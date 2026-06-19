import json

def should_escalate(best_score: float, threshold: float, user_query: str) -> bool:
    """Evaluates multiple business conditions to determine human agent handoff routing."""
    # Condition 1: Low retrieval data confidence
    if best_score < threshold:
        return True
        
    # Condition 2: Highly sensitive topics requiring strict procedural compliance
    sensitive_triggers = ["billing", "refund", "legal", "lawsuit", "unauthorized charges", "duplicate charge"]
    if any(trigger in user_query.lower() for trigger in sensitive_triggers):
        return True
        
    return False

def generate_handoff_summary(user_query: str, persona: str, context_chunks: list, history: list) -> dict:
    """Compiles clean structural JSON handoff summaries mapping to system guidelines requirements."""
    best_score = max([c["score"] for c in context_chunks]) if context_chunks else 0.0
    return {
        "persona": persona,
        "issue": user_query[:100] + "...",
        "conversation_history": history[-3:] if history else [],
        "documents_used": list(set([c["source"] for c in context_chunks])) if context_chunks else [],
        "confidence_score": float(best_score),
        "recommendation": "Assign to human Tier-2 representative. Inspect transactional status logs and account standing."
    }