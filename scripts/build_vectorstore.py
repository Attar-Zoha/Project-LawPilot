# scripts/build_vectorstore.py
import json
from pathlib import Path
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    # prefer the community import warning-free
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import pickle

def load_docs(docs_dir):
    docs = []
    for f in Path(docs_dir).glob("*.json"):
        obj = json.load(open(f, 'r', encoding='utf-8'))
        text = obj.get("text","").strip()
        if not text:
            continue
        metadata = obj.get("metadata", {})
        metadata["path"] = obj.get("path")
        docs.append({"text": text, "metadata": metadata})
    return docs

def main(docs_dir, output_dir, embedding_model="law-ai/InLegalBERT", use_fallback_if_fail=True, limit=None):
    docs = load_docs(docs_dir)
    if limit:
        docs = docs[:limit]
    print(f"Loaded {len(docs)} docs")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = []
    metadatas = []
    for d in docs:
        chunks = text_splitter.split_text(d["text"])
        for i, c in enumerate(chunks):
            texts.append(c)
            md = d["metadata"].copy()
            md["chunk"] = i
            md["source"] = d["metadata"].get("path")
            metadatas.append(md)

    print(f"Total text chunks: {len(texts)}")

    # Try InLegalBERT first (if installed/cached). If it fails fallback to all-MiniLM
    try:
        emb = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    except Exception as e:
        print(f"Embedding load failed for {embedding_model}: {e}")
        if use_fallback_if_fail:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            print("Falling back to", embedding_model)
            emb = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
        else:
            raise

    vector_store = FAISS.from_texts(texts, emb, metadatas=metadatas)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # save faiss + metadata
    vector_path = output_dir / "faiss_index"
    with open(output_dir / "faiss_store.pkl", "wb") as f:
        pickle.dump({"metadatas": metadatas, "texts_count": len(texts)}, f)

    # LangChain FAISS has a save_local method
    vector_store.save_local(str(vector_path))
    print("Saved vectorstore to", vector_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="data/legal_corpus/docs")
    parser.add_argument("--out-dir", default="data/legal_vectorstore")
    parser.add_argument("--embedding-model", default="law-ai/InLegalBERT")
    parser.add_argument("--limit", type=int, default=None, help="limit number of docs for a quick test")
    args = parser.parse_args()
    main(args.docs_dir, args.out_dir, args.embedding_model, True, args.limit)
