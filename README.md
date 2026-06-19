# Persona-Adaptive Customer Support Agent

An intelligent, production-ready customer support agent built with Python, Streamlit, and the Google GenAI SDK. The system uses structured data parsing to analyze incoming communication footprints, maps them to specific user archetypes, conducts real-time semantic similarity retrieval using a local persistent vector database, and enforces mathematical guardrails to seamlessly route conversations between autonomous resolution and structured human handoffs.

---

## 🛠️ Tech Stack & Dependencies

*   **Core Language:** `Python >= 3.11`
*   **Web Framework & Dashboard UI:** `Streamlit >= 1.30.0`
*   **LLM Core Engine:** `Google GenAI SDK (gemini-2.5-flash)`
*   **Embedding Model Engine:** `Google GenAI SDK (gemini-embedding-001)`
*   **Vector Database Storage:** `ChromaDB >= 0.4.0`
*   **Data Segmentation Splitter:** `LangChain Text Splitters >= 0.0.1`
*   **Geometric Layout Parser:** `PyPDF >= 3.8.0`
*   **Environment Variable Security:** `Python-Dotenv >= 1.0.0`

---

## 📐 System Architecture Diagram

```text
                  +-----------------------+
                  |  Customer Query Input  |
                  +-----------+-----------+
                              |
                     [src/classifier.py]
                              v
                +---------------------------+
                | Persona Detection Engine  | ---> [Output Persona Tag]
                +-------------+-------------+      (Tech / Frustrated / Exec)
                              |
                     [src/rag_pipeline.py]
                              v
                +---------------------------+
                | Cosine Similarity Search  | <--- [Knowledge Base /data]
                +-------------+-------------+      (TXT, MD, parsed PDF)
                              |
                     [src/escalator.py]
                              v
               /=============================\
              /   Retrieval Confidence Check  \
             <  Is Max Cosine Score >= 0.45?   >
              \   And No Sensitive Keywords?  /
               \=============================/
                        /           \
                 (Yes) /             \ (No / Sensitive Trigger)
                      /               \
         [src/generator.py]            +----------------------------+
                      v                |  Human Escalation Block    |
        +---------------------------+  +--------------+-------------+
        | Grounded Adaptive Prompt  |                 |
        |      Compiler Engine      |        [Structured JSON Data]
        +-------------+-------------+                 v
                      |                +----------------------------+
                      v                | Live Agent Handoff Summary |
        +---------------------------+  +----------------------------+
        | Dynamic Target Response   |
        +---------------------------+


Persona Detection Strategy
The system reads conversational behavior patterns dynamically through src/classifier.py. Instead of using brittle, rigid string-matching patterns, the input is contextualized via an advanced system prompt instruction layer running on gemini-2.5-flash.

Classification Rules & Archetypes
Technical Expert: Triggered when strings contain architectural jargon, configuration properties, terminal code parameters, or HTTP error endpoints.

Frustrated User: Triggered when strings convey emotional stress indicators, intensive exclamation patterns, capitalized emphasis layout configurations, or explicit system complaints.

Business Executive: Triggered when strings emphasize business outcome summaries, target pricing limits, structural resource metrics, or strict deployment timelines.

Structural JSON Grounding

To achieve deterministic predictability, the classification query leverages a strict JSON layout parameter map enforcing a required data structure:

{
  "persona": "Technical Expert | Frustrated User | Business Executive",
  "confidence": 0.95,
  "reasoning": "Detailed justification context string."
}

RAG Pipeline Design1. Parsing & ChunkingDocuments populate the /data folder in various text formats (.txt, .md, .pdf). PyPDF safely converts geometric structures into standard strings. The engine segments long paragraphs using LangChain’s RecursiveCharacterTextSplitter configured with a chunk_size of 500 characters and an iterative chunk_overlap of 50 characters. This overlap preserves crucial system settings (like API endpoints or authentication steps) across semantic block boundaries.2. Embeddings & Local Vector IndexingSegmented blocks are vectorized into a dense multidimensional grid using gemini-embedding-001. The mathematical models and text arrays are indexed in a local instance folder (./chroma_db) managed by a PersistentClient config. This allows real-time similarity checks without requiring remote cloud database clusters.3. Contextual RetrievalDuring an active session, real-time user query vectors are calculated. The engine computes similarity using a Distance Matrix calculation:$$\text{Similarity}(Q, D) = 1.0 - \text{Chroma Distance Metric}$$The system extracts the top $k=3$ contextual fragments matching the semantic inquiry.🚨 Escalation Logic & GuardrailsThe application enforces a high-security boundary via src/escalator.py to prevent hallucination drops or improper routing.System TriggersAn interaction is instantly routed away from automated LLM response generation and passed to a human technician when:Low Document Proximity: The maximum semantic score of the retrieved fragments falls below the strict 0.45 confidence limit.Sensitive Administrative Subjects: The text pattern explicitly mentions protected business keywords such as billing, refund, legal, lawsuit, or duplicate charge.Handoff Output SpecificationsWhen triggered, an automated warning notifies the user interface while compiling a structured JSON data block for live operator handoff logging:JSON{
  "persona": "Frustrated User",
  "issue": "Where is the guide to clear cookies? It's been an hour and nothing is loading...",
  "conversation_history": [],
  "documents_used": ["billing_policy.txt", "api_troubleshooting.md", "password_reset_guide.pdf"],
  "confidence_score": 0.2516,
  "recommendation": "Assign to human Tier-2 representative. Inspect transactional status logs and account standing."
}
⚙️ Environment VariablesSecure secrets are managed via a .env file located at the project root. This file must remain ignored by git versions and must never be exposed publicly.Code snippetGEMINI_API_KEY="your_actual_google_gemini_api_key_here"
🚀 Setup & Execution InstructionsFollow these instructions to set up the workspace locally on Windows, macOS, or Linux systems:Clone the project directory layout and change into the folder:Bash    cd persona-support-agent
    ```
2.  **Initialize your Python virtual environment architecture:**
```bash
    python -m venv venv
    ```
3.  **Activate the runtime framework environment:**
    *   **Windows (Git Bash/MINGW64):** `source venv/Scripts/activate`
    *   **Windows (CMD/PowerShell):** `.\venv\Scripts\activate`
    *   **macOS/Linux:** `source venv/bin/activate`
4.  **Install the required package dependencies:**
```bash
    pip install -r requirements.txt
    ```
5.  **Launch the Streamlit analytical telemetry application dashboard:**
```bash
    streamlit run app.py
    ```

---

## 🎯 Verification Testing Scenarios

Test the agent using these 5 standard validation queries to evaluate its response adaptation and edge-case guardrails:

| # | User Message Input | Expected Persona | Target Pipeline Behavior |
|---|---|---|---|
| 1 | *"Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!"* | **Frustrated User** | **Trigger Escalation Case:** High emotional tone matches, but low RAG confidence (`<0.45`) generates a clean human handoff JSON block. |
| 2 | *"What are the header parameter requirements for your bearer token auth implementation?"* | **Technical Expert** | Retrieves `api_troubleshooting.md`. Outputs exact structural markdown specs, authorization details, and API code blocks. |
| 3 | *"Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved."* | **Business Executive** | **Trigger Escalation Case:** Detects high financial risk context; generates an automated handoff schema mapping. |
| 4 | *"I'm experiencing an issue with your database integration that's causing internal errors."* | **Technical Expert** | Gathers technical details and outlines step-by-step resolution pathways. |
| 5 | *"My billing statement has unexpected duplicate charges. I demand an immediate refund!"* | **Frustrated User** | **Trigger Escalation Case:** Recognizes the financial refund request, blocks autonomous LLM output generation, and surfaces human operator telemetry parameters. |

---

## ⚠️ Known Limitations & Future Roadmap

*   **Asynchronous Processing:** Long-form vector generation blocks the UI rendering engine during startup. Transitioning to async ingestion handlers will improve processing scaling.

*   **Persistent Multi-Session Memory:** Current data loops track thread variables usibles using standard user memory layers. Integrating a persistent cache store (like Redis or SQLite) will enable true continuous multi-turn session tracking.

# To Start / run the project -:
------------------------------------
Open the GitBash terminal

1. python -m venv venv
2. source venv/Scripts/activate
3. pip install -r requirements.txt
4. pip install pypdf
5. pip install langchain-text-splitters
6. Get your own Gemini-api-key and paste it in your .env folder
7. streamlit run app.py

# Live deployed Link -: https://persona-support-agent-iufeffvqtfzwen8v3xfwxr.streamlit.app/
