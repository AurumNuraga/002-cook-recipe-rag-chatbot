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
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()

hf_api_key = os.getenv("HF_TOKEN")

if not hf_api_key:
    raise ValueError("HF_TOKEN not found in .env file")

hf_model_id = "Qwen/Qwen2.5-7B-Instruct"

Settings.llm = HuggingFaceInferenceAPI(
    model_name=hf_model_id,
    token=hf_api_key,
    temperature=0.5,
    max_tokens=2048,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

documents = load_documents(r"./dataset/indonesian_recipes.csv")

bm25_retriever = BM25Retriever.from_defaults(
    nodes=documents,
    similarity_top_k=3
)

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=3,
    num_queries=1,       
    use_async=False,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=Settings.llm,
    response_mode="compact"
)

question = "Apa resep lengkap ayam pedas yang mudah dibuat di rumah?"

response = query_engine.query(question)

print(response)