from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
emb = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Load the persisted FAISS index safely (allow dangerous deserialization)
vs = FAISS.load_local(
    "data/legal_vectorstore/faiss_index",
    embeddings=emb,
    allow_dangerous_deserialization=True
)

# Check if it loaded correctly
print("Vectorstore loaded:", vs)
print("Number of vectors in index:", vs.index.ntotal)

# Test a simple query
query = "Fundamental rights in Indian Constitution"
docs = vs.similarity_search(query, k=3)  # top 3 relevant chunks

print("\nTop 3 retrieved chunks:")
for i, d in enumerate(docs, 1):
    print(f"--- Chunk {i} ---\n{d.page_content[:500]}...\n")

