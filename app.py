import streamlit as st
import json
from src.config import RETRIEVAL_CONFIDENCE_THRESHOLD
from src.classifier import classify_customer_persona
from src.rag_pipeline import LocalRAGPipeline
from src.generator import generate_adaptive_response
from src.escalator import should_escalate, generate_handoff_summary

st.set_page_config(page_title="Persona Adaptive Support Hub", layout="wide")

# Persistent Application RAG State Engine Init
if "rag" not in st.session_state:
    with st.spinner("Initializing Vector Engine Storage indexes..."):
        pipeline = LocalRAGPipeline()
        if len(pipeline.collection.get()['ids']) == 0:
            pipeline.ingest_knowledge_base()
        st.session_state.rag = pipeline

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Adaptive Support Agent Dashboard")
st.caption("AI Engineering Assignment Submission Framework - Evaluation Workspace")
st.markdown("---")

left_panel, right_panel = st.columns([3, 2])

with left_panel:
    st.subheader("💬 Active Interaction Hub")
    user_input = st.text_input("Enter incoming user communication text:", placeholder="Type client inquiry here...")
    
    if st.button("Process Live Query") and user_input:
        with st.spinner("Analyzing communication payload..."):
            classification = classify_customer_persona(user_input)
            persona = classification["persona"]
            conf_class = classification["confidence"]
        
        retrieved_chunks = st.session_state.rag.retrieve_context(user_input)
        best_score = max([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0.0
        
        # Check against escalation triggers
        escalation_active = should_escalate(best_score, RETRIEVAL_CONFIDENCE_THRESHOLD, user_input)
        
        if escalation_active:
            bot_response = (
                "⚠️ **System Flagged for Human Intervention**: I deeply apologize, but I am unable to safely "
                "resolve this specific inquiry with our automated standard documentation maps. I am transferring "
                "this session directly to a live billing/technical specialist for prompt review."
            )
            handoff_record = generate_handoff_summary(user_input, persona, retrieved_chunks, st.session_state.chat_history)
        else:
            bot_response = generate_adaptive_response(user_input, persona, retrieved_chunks)
            handoff_record = None
            
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
        
        # Expose all operational diagnostic details transparently
        st.session_state.telemetry = {
            "persona": persona,
            "classification_confidence": conf_class,
            "reasoning": classification["reasoning"],
            "best_retrieved_score": best_score,
            "sources": list(set([c["source"] for c in retrieved_chunks])),
            "escalated": escalation_active,
            "handoff_json": handoff_record
        }

    # Render Active Messenger Bubbles
    for chat in reversed(st.session_state.chat_history):
        st.chat_message("user").write(chat["user"])
        st.chat_message("assistant").write(chat["bot"])

with right_panel:
    st.subheader("📊 System Telemetry & RAG Grounding Logs")
    if "telemetry" in st.session_state:
        t = st.session_state.telemetry
        
        st.metric(label="Active Tone Assignment State", value=t["persona"])
        st.progress(t["classification_confidence"])
        st.write(f"**Classifier Justification Logic:** {t['reasoning']}")
        st.markdown("---")
        
        st.metric(label="RAG Vector Match Confidence Metric", value=f"{t['best_retrieved_score']:.4f}")
        st.write("**Referenced Source Anchors:**", t["sources"])
        st.markdown("---")
        
        if t["escalated"]:
            st.error("🚨 Escalation Workflow Triggered")
            st.subheader("📋 Output Handoff JSON Payload")
            st.json(t["handoff_json"])
        else:
            st.success("🟢 Autonomous Guardrail Active: Grounded Response Delivered")
    else:
        st.info("Awaiting user input message query stream to process runtime analytics parameters.")