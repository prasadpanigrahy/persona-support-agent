import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import chromadb
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class LocalRAGPipeline:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name="support_kb")

    def get_embedding(self, text: str) -> list:
        """Generates dense vector embeddings using gemini-embedding-001."""
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL, 
            contents=text
        )
        return response.embeddings[0].values

    def parse_file(self, file_path: str) -> str:
        """Parses content out of unstructured formats safely, checking for empty or corrupt files."""
        # Safeguard: Skip processing if file is completely empty
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty or unitialized document path: {file_path}")
            return ""
            
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        try:
            if ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif ext == ".pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            # Operational guardrail: print error log to console but don't crash app
            print(f"Warning: Could not extract data payload from {file_path} due to error: {e}")
            return ""
            
        return text
    def ingest_knowledge_base(self, data_dir: str = "./data"):
        """Ingests workspace data, segments it using recursive character text splitters, and indices vectors."""
        if not os.path.exists(data_dir):
            return
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            if os.path.isfile(file_path):
                content = self.parse_file(file_path)
                if not content.strip():
                    continue
                
                chunks = splitter.split_text(content)
                for idx, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)
                    chunk_id = f"{file_name}_chunk_{idx}"
                    
                    self.collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        metadatas=[{"source": file_name, "chunk_index": idx}],
                        documents=[chunk]
                    )

    def retrieve_context(self, query: str, top_k: int = 3) -> list:
        """Calculates distance parameters and returns top contextual matches with simulated confidence metrics."""
        query_vector = self.get_embedding(query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        
        retrieved_items = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = 1.0 - distance  # Convert distance metrics directly into standard accuracy score
                
                retrieved_items.append({
                    "text": results['documents'][0][i],
                    "source": results['metadatas'][0][i]['source'],
                    "score": round(score, 4)
                })
        return retrieved_items