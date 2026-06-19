import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_DIR = "./chroma_db"

# Model Declarations
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Application Hard Thresholds
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.45
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3