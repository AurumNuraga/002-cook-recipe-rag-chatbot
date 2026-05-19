import pandas as pd
import ast
from data_loader import load_documents

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

documents = load_documents(r"./dataset/indonesian_recipes.csv")

print(f"Total documents created: {len(documents)}")
print("\n=== Sample Document ===\n")
print(documents[0].text[:500])
print("\n=== Metadata ===")
print(documents[0].metadata)

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

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print("Embedding selesai dan data tersimpan ke Qdrant.")