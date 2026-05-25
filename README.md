# 002-cook-recipe-rag-chatbot 🍲

An AI-powered Retrieval-Augmented Generation (RAG) chatbot specialized in exploring and retrieving traditional Indonesian food recipes. This application leverages **LlamaIndex** as the orchestration framework, **Qdrant** as the high-performance vector database, **Groq (Llama 3.1 8B)** for lightning-fast and accurate generation, and **Streamlit** for a clean, user-friendly chat interface.

---

## 🚀 Features

* **Semantic Recipe Search:** Uses multilingual embedding models to accurately understand culinary queries and match them with Indonesian recipe datasets.
* **Structured Output:** Generates perfectly formatted markdown outputs detailing the **Dish Name**, **Ingredients** (bullet points), and **Cooking Steps** (numbered list).
* **Vector Infrastructure:** Employs Qdrant local instance for fast, scalable vector store lookups.
* **Interactive Web UI:** Clean, intuitive, state-preserved conversational interface built using Streamlit.

---

## 🛠️ Tech Stack & Architecture

* **LLM Orchestration:** `LlamaIndex`
* **Generative Model:** `Groq (llama-3.1-8b-instant)` via LlamaIndex-Groq integration.
* **Embedding Model:** `intfloat/multilingual-e5-base` via HuggingFace (ideal for handling Indonesian nuances).
* **Vector Database:** `Qdrant` (running locally on port `6333`).
* **UI Frontend:** `Streamlit`

<img width="1918" height="865" alt="Screenshot 2026-05-19 183343" src="https://github.com/user-attachments/assets/a8a8f3d5-ae31-45e5-9033-84920c152460" />

---

## 📂 Repository Structure

```text
├── dataset/
│   └── indonesian_recipes.csv   # The source dataset containing cooking recipes
├── app.py                      # Main Streamlit web application script
├── data_loader.py              # Script to clean, parse, and structure CSV rows into LlamaIndex Documents
├── embedding.py                # Script to generate embeddings and ingest them into Qdrant
├── llm.py                      # Local testing script for CLI-based query pipeline evaluation
├── requirements.txt            # Python dependencies
└── .env                        # Local environment file for storing secrets (API keys)
