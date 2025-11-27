import os
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_local_corpus(
    faiss_dir="data/legal_vectorstore/faiss_index",
    json_folder="data/indian_judgements/sc_data/english"
):
    """
    Load the local Indian Supreme Court corpus.
    Priority:
      1️⃣ Load prebuilt FAISS vectorstore (fast path)
      2️⃣ If not found, load raw JSON texts from judgments folder (slow fallback)
    """

    # Try to load prebuilt FAISS vectorstore
    if os.path.exists(faiss_dir):
        try:
            st.sidebar.info("🔍 Loading prebuilt FAISS index...")
            emb = HuggingFaceEmbeddings(
                model_name="law-ai/InLegalBERT",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
            vs = FAISS.load_local(
                faiss_dir,
                embeddings=emb,
                allow_dangerous_deserialization=True
            )
            st.sidebar.success("✅ Loaded FAISS vectorstore successfully!")
            return vs
        except Exception as e:
            st.warning(f"⚠️ Could not load FAISS index: {e}")

    # Fallback: load raw texts if FAISS index not found
    st.sidebar.warning("⚠️ No FAISS index found. Loading raw judgments instead...")
    corpus_texts = []
    for root, _, files in os.walk(json_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            text = data.get("text") or data.get("judgment_text", "")
                            if text.strip():
                                corpus_texts.append(text)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    st.sidebar.success(f"✅ Loaded {len(corpus_texts)} judgments (raw mode)")
    return corpus_texts
