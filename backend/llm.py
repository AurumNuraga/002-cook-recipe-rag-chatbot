import os
from dotenv import load_dotenv
from data_loader import load_documents

from llama_index.core import Settings
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model_id = "llama-3.1-8b-instant"

Settings.llm = Groq(
model=groq_model_id,
api_key=groq_api_key,
temperature=0.3
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base"
)

client = QdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="indonesian_recipes"
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=Settings.embed_model,
)

vector_retriever = index.as_retriever(
    similarity_top_k=3
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=vector_retriever,
    llm=Settings.llm,
    response_mode="compact"
)

question = "Apa resep lengkap ayam pedas yang mudah dibuat di rumah?"

response = query_engine.query(question)

print(response)