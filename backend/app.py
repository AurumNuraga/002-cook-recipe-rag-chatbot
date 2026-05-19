import os
import streamlit as st
from dotenv import load_dotenv
from data_loader import load_documents
import warnings
import logging

from llama_index.core import Settings
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Chatbot Resep Indonesia", page_icon="🍲", layout="centered")
st.title("🍲 Chatbot Resep Makanan Indonesia")
st.caption("Tanya saya tentang resep masakan Nusantara!")

@st.cache_resource(show_spinner="Memuat model dan database resep... Mohon tunggu.")
def init_query_engine():
    load_dotenv(override=True)
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("GROQ_API_KEY tidak ditemukan di file .env!")
        st.stop()

    groq_model_id = "llama-3.1-8b-instant"

    Settings.llm = Groq(
    model=groq_model_id,
    api_key=groq_api_key,
    temperature=0.3
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base"
    )

    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name="indonesian_recipes")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    vector_retriever = index.as_retriever(similarity_top_k=3)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=vector_retriever,
        llm=Settings.llm,
        response_mode="compact"
    )
    
    return query_engine

query_engine = init_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Resep makanan Indonesia apa yang ingin kamu masak hari ini?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tulis pertanyaan resepmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari resep..."):
            try:
                optimized_prompt = (
                    f"Pengguna bertanya tentang resep: {prompt}\n\n"
                    "Tugasmu: Carikan resep yang sesuai dari dokumen yang diberikan.\n\n"
                    "Sajikan jawabanmu dengan format Markdown yang rapi dan terstruktur persis seperti ini:\n"
                    "**[Nama Masakan]**\n\n"
                    "**Bahan-bahan:**\n(tuliskan dalam bentuk bullet points)\n\n"
                    "**Cara Membuat:**\n(tuliskan langkah demi langkah menggunakan nomor urut secara lengkap, jangan ada yang dipotong)."
                )
                response = query_engine.query(optimized_prompt)
                
                response_str = str(response)
                st.markdown(response_str)
                
                st.session_state.messages.append({"role": "assistant", "content": response_str})
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi API: {e}")