import pickle
import os, time, json, glob

import streamlit as st
from dotenv import load_dotenv

import fitz  # PyMuPDF
import torch
import numpy as np
import faiss
import pyperclip

from docx import Document as DocxDocument

from google import genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.llms import GoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains.summarize import load_summarize_chain

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


from transformers import AutoTokenizer, AutoModel

from load_corpus import load_local_corpus

# ---------------- BASIC SETUP ----------------
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    st.error("Google API Key missing. Please add it to .env as API_KEY.")
    st.stop()
client = genai.Client(api_key=api_key)

# ---------------- SAFE TEXT EXTRACTION ----------------
# def safe_extract_text(resp):
#     try:
#         if hasattr(resp, "text") and resp.text:
#             return resp.text
#         if hasattr(resp, "candidates") and resp.candidates:
#             parts = resp.candidates[0].content.parts
#             if parts and hasattr(parts[0], "text"):
#                 return parts[0].text
#     except Exception:
#         pass
#     return "(No content returned)"

# ---------------- GEMINI CALL ----------------
def robust_generate(prompt, retries=2):
    models = ["gemini-2.5-flash", "gemini-2.0-flash"]

    for model_name in models:
        for _ in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response and response.text:
                    return response.text
                

            except Exception as e:
                st.error(f"Gemini error: {e}")
                if "429" in str(e):
                    break
                time.sleep(1)

    return "(Error generating response)"

# ---------------- FILE HANDLING ----------------
def get_document_text(files):
    txt = ""
    for f in files:
        if f.name.endswith(".pdf"):
            doc = fitz.open(stream=f.read(), filetype="pdf")
            for p in doc:
                txt += p.get_text()
        elif f.name.endswith(".docx"):
            d = DocxDocument(f)
            for para in d.paragraphs:
                txt += para.text + "\n"
    return txt

# ---------------- LEGALBERT EMBEDDING ----------------
LEGAL_MODEL = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(LEGAL_MODEL)
model = AutoModel.from_pretrained(LEGAL_MODEL)

def embed_text_legalbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# ---------------- LOCAL CORPUS LOADER ----------------
def load_legal_corpus():
    """Load local Indian Supreme Court judgments from JSON or PDFs and create embeddings."""
    try:
        all_texts = load_local_corpus()
        if not all_texts:
            st.warning("No judgments found. Please ensure the dataset is downloaded and unzipped properly.")
            return

        # Embed all texts
        embeddings = np.vstack([embed_text_legalbert(t) for t in all_texts])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Store in session
        st.session_state.legal_index = index
        st.session_state.legal_texts = all_texts
        st.sidebar.success("✅ Supreme Court Judgments loaded successfully!")

        # Optional: save as FAISS index for future use
        emb = HuggingFaceEmbeddings(
            model_name="law-ai/InLegalBERT",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        vs = FAISS.from_texts(all_texts, embedding=emb)
        vs.save_local("data/legal_vectorstore/faiss_index")
        st.sidebar.info("✅ FAISS vectorstore saved for future runs.")

    except Exception as e:
        st.error(f"Error loading legal corpus: {e}")

# ---------------- TEXT SPLITTER & VECTORSTORE ----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(text)

def get_vector_store(chunks, model_name="law-ai/InLegalBERT"):
    try:
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        vs = FAISS.from_texts(chunks, embedding=emb)
        # st.session_state.vector_store, st.session_state.raw_text = vs, "\n".join(chunks)
        st.session_state.vector_store = vs
        st.session_state.chunks = chunks
        st.session_state.raw_text = "\n".join(chunks[:3])
        st.sidebar.success("Documents processed ✅")
    except Exception as e:
        st.error(f"Embedding error: {e}")

# ---------------- CHAINS ----------------
# # def get_conversational_chain():
#     """QA chain with structured legal prompt for Document QA (RAG)."""
#     p = PromptTemplate(
#         template="""
#         You are LawPilot, a professional legal research assistant.
#         Use ONLY the provided context from Indian Supreme Court judgments to answer the user's question.
#         Do NOT add any information from outside sources.
#         Answer in a detailed, accurate, and structured way using the format below.
#         If any information is not available in the context, write 'Not specified in retrieved context.'

#         Question: {question}
#         Context: {context}

#         Answer in the following format:
#         Case Name: 
#         Year: 
#         Key Holdings: 
#         Reasoning: 
#         Majority/Dissent Points: 
#         Relevant Articles/Citations: 
#         Notes: 
#         """,
#         input_variables=["context", "question"],
#     )

#     model = GoogleGenerativeAI(
#     model="models/gemini-1.5-flash",
#     temperature=0.3,
#     google_api_key=api_key
# )

#     return load_qa_chain(model, chain_type="stuff", prompt=p)

# def get_summary_chain():
#     m = GoogleGenerativeAI(
#         model="models/gemini-1.5-flash",
#         temperature=0.2,
#         google_api_key=api_key
#     )
#     return load_summarize_chain(m, chain_type="map_reduce")

# ---------------- PRECEDENT PROMPT & HANDLER ----------------
precedent_prompt = PromptTemplate(
    template="""
You are LawPilot, a legal research assistant specializing in Indian Supreme Court precedents.
Use ONLY the retrieved precedent context to answer the user's question. Do NOT include information from outside sources.
If the required information is not present in the context, write 'Not specified in retrieved precedent.'

Question: {question}
Precedent Context: {context}

Provide a detailed, professional answer suitable for a legal professional in the following format. Each item should appear on a new line:

1. Case Name:

2. Year of Decision:  

3. Domain: (e.g., Constitutional Law, Consumer Law, Criminal Law, Employment, Environmental Law, Tax/GST/Income, Other)  

4. Subject Matter: (Brief description of what the case was about)  

5. Judgment: (Outcome / decision of the Court)  

6. Key Holdings / Ratio Decidendi:  

7. Reasoning / Court Observations:  

8. Majority / Dissenting Opinions:  

9. Relevant Articles / Sections / Citations:  

10. Practical Notes / Implications for Legal Practice:  

Ensure clarity, precision, and professional tone suitable for referencing in legal work.
""",
    input_variables=["context", "question"]
)

def handle_precedent_qna(q):
    if not st.session_state.vector_store:
        return "⚠️ Load legal corpus first."

    docs = st.session_state.vector_store.similarity_search(q, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = precedent_prompt.format(
        context=context,
        question=q
    )
    return robust_generate(prompt)

# ---------------- HANDLERS ----------------
def handle_doc_qna(q):
    if not st.session_state.vector_store:
        return "⚠️ Upload and process documents first."

    docs = st.session_state.vector_store.similarity_search(q, k=4)

    context = "\n\n".join([d.page_content for d in docs])

    # Recent chat history
    history = "\n".join(
        [f"{r}: {m}" for r, m in st.session_state.chat_history[-4:]]
    )

    prompt = f"""
        You are LawPilot, a legal assistant.

        Use ONLY the following document context and conversation history.

        Conversation History:
        {history}

        Document Context:
        {context}

        Current Question:
        {q}

        If answer is unavailable, say:
        "Not specified in the document."

        Answer in a clear, structured legal format.
        """
    return robust_generate(prompt)

def handle_general_qna(user_question):
    if "legal_index" in st.session_state and st.session_state.legal_index:
        query_vec = embed_text_legalbert(user_question)
        D, I = st.session_state.legal_index.search(query_vec, k=3)
        context = "\n".join([st.session_state.legal_texts[i] for i in I[0]])
        prompt = (
            f"You are LawPilot, an AI legal assistant. "
            f"Use the following context from Supreme Court judgments to answer:\n\n{context}\n\nQuestion: {user_question}"
        )
        return robust_generate(prompt)
    else:
        prompt = f"You are LawPilot, a helpful Legal AI Assistant. Answer: {user_question}"
        return robust_generate(prompt)

def generate_summary(instr):

    if not st.session_state.vector_store:
        return "Upload and process documents first."

    # Retrieve relevant chunks instead of full document
    docs = st.session_state.vector_store.similarity_search(
        "summary of legal document",
        k=4
    )

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
        Summarize the following legal document.

        Instruction:
        {instr}

        Document:
        {context}
        """

    return robust_generate(prompt)

    
def translate_text(text, lang):
    return robust_generate(f"Translate to {lang}: {text}")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="LawPilot - Legal AI", page_icon="⚖️", layout="wide")
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "raw_text" not in st.session_state: st.session_state.raw_text = ""

with st.sidebar:
    st.image("logo.png", width=250)
    st.title("LawPilot Menu")

    # --- Load Legal Corpus ---
    if st.button("Load Legal Corpus", use_container_width=True):
        with st.spinner("Loading Legal Corpus..."):
            try:
                emb = HuggingFaceEmbeddings(
                    model_name="law-ai/InLegalBERT",
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                )
                vs = FAISS.load_local(
                    "data/legal_vectorstore/faiss_index",
                    embeddings=emb,
                    allow_dangerous_deserialization=True
                )
                st.session_state.vector_store = vs
                st.success("✅ Prebuilt Legal Corpus loaded successfully (FAISS).")
            except Exception as e:
                st.warning("⚠️ Could not load persisted FAISS index. Embedding raw documents in-memory...")
                load_legal_corpus()

    st.write("---")

    # --- Upload & Process Docs ---
    st.header("Upload Your Documents")
    up = st.file_uploader("Upload legal documents", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Process Documents", use_container_width=True):
        if up:
            with st.spinner("Analyzing documents..."):
                txt = get_document_text(up)
                chunks = get_text_chunks(txt)
                get_vector_store(chunks)
        else:
            st.warning("Please upload at least one file.")

    st.write("---")

   # --- Summarize Uploaded Docs ---
    if st.session_state.vector_store:

        st.subheader("Document Summary")

        ins = st.text_input(
            "Custom Instruction",
            "Provide a concise legal summary."
        )

        if st.button("Generate Summary"):

            with st.spinner("Generating summary..."):

                summary = generate_summary(ins)

                st.session_state.summary_output = summary

        # DISPLAY SUMMARY
        if "summary_output" in st.session_state:

            st.text_area(
                "Generated Summary",
                st.session_state.summary_output,
                height=300
            )

        # --- Chat History Preview ---
    st.subheader("Chat History")

    if st.session_state.chat_history:

        for role, msg in st.session_state.chat_history[-5:]:

            if role == "You":
                st.markdown(f"👤 **You:** {msg[:80]}")

            else:
                st.markdown(f"🤖 **LawPilot:** {msg[:80]}...")

    else:
        st.caption("No chat history yet.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.rerun()

# ---------------- MAIN CHAT INTERFACE ----------------
st.title("⚖️ LawPilot – Legal AI Assistant")
chat_tab, trans_tab = st.tabs(["Chat", "Translate"])

# # --- Chat Tab ---
# with chat_tab:
#     for role, msg in st.session_state.chat_history:
#         with st.chat_message(role, avatar="👤" if role == "You" else "🤖"):
#             st.markdown(msg)

# --- Chat Tab ---
with chat_tab:
    for idx, (role, msg) in enumerate(st.session_state.chat_history):

        with st.chat_message(role, avatar="👤" if role == "You" else "🤖"):

            # Show message normally
            st.markdown(msg)

            # Copy button only for LawPilot responses
            if role == "LawPilot":

                if st.button("📋 Copy Response", key=f"copy_{idx}"):

                    pyperclip.copy(msg)

                    st.success("Response copied to clipboard.")

# --- Translate Tab ---
with trans_tab:
    txt = st.text_area("Enter English text", height=150)
    lang = st.selectbox("Target Language", ["Hindi", "Marathi"])
    if st.button("Translate"):
        if txt:
            with st.spinner("Translating..."):
                st.text_area("", translate_text(txt, lang), height=150)
        else:
            st.warning("Enter text first.")

# --- Chat Input & Routing ---
chat_mode = st.radio("Select Query Mode:", ["Document QA", "General Legal Q&A", "Precedent QA"], horizontal=True)
prompt = st.chat_input("Ask a legal question...")
if prompt:
    st.session_state.chat_history.append(("You", prompt))
    
    if chat_mode == "Document QA":
        if st.session_state.vector_store:
            ans = handle_doc_qna(prompt)
        else:
            ans = "⚠️ Upload and process documents first for Document QA."
    elif chat_mode == "Precedent QA":
        ans = handle_precedent_qna(prompt)
    else:  # General Legal Q&A
        ans = handle_general_qna(prompt)
    
    st.session_state.chat_history.append(("LawPilot", ans))
    st.rerun()


