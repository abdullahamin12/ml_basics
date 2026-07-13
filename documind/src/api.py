from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from .chunking.langchain_chunker import LangChainChunker
from .embedding.nvidia_embedder import NvidiaEmbedder
from .vectorstore.database import QdrantVectorStore
from .llm.nvidia_llm import NvidiaLLMClient
from .pipeline import RAGPipeline

# 1. Import the concrete PostgreSQL class, not the base class
from .chat_history.chat import PostgresChatHistory

pipeline = None
db_repo = None # Add a global variable for the database

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    global db_repo
    
    # 2. Instantiate the database connection (using Docker networking)
    db_repo = PostgresChatHistory(
        host="postgres", 
        database="documind", 
        user="myuser", 
        password="123"
    )
    
    pipeline = RAGPipeline(
        chunker=LangChainChunker(),
        embedder=NvidiaEmbedder(),
        vector_store=QdrantVectorStore(host="qdrant"),
        llm_client=NvidiaLLMClient()
    )
    yield

app = FastAPI(lifespan=lifespan)

class IngestRequest(BaseModel):
    document: str

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest(request: IngestRequest):
    pipeline.ingest(request.document)
    return {"status": "ingested successfully"}

@app.post("/query")
def query(request: QueryRequest):
    # 3. EXTRACT: Pull the last 5 messages from PostgreSQL
    history = db_repo.extract_history(limit=5)
    
    # 4. QUERY: Pass the question AND the history to the pipeline
    ans = pipeline.query(request.question, chat_history=history)
    
    # 5. SAVE: Store the new interaction
    db_repo.save_history(request.question, ans)
    
    return {"Answer": ans}