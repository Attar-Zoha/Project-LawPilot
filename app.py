import pickle
from langchain.schema import Document as LCDocument  # renamed to avoid clash with python-docx

import streamlit as st
import os, time, json, glob
from dotenv import load_dotenv
import fitz, torch
from docx import Document
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss

from load_corpus import load_local_corpus

# ---------------- BASIC SETUP ----------------
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    st.error("Google API Key missing. Please add it to .env as API_KEY.")
    st.stop()
genai.configure(api_key=api_key)

# ---------------- SAFE TEXT EXTRACTION ----------------
def safe_extract_text(resp):
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    except Exception:
        pass
    return "(No content returned)"

# ---------------- GEMINI CALL ----------------
def robust_generate(prompt, retries=2):
    for model_name in ["gemini-2.5-flash", "gemini-1.5-flash"]:
        model = genai.GenerativeModel(model_name)
        for _ in range(retries + 1):
            try:
                r = model.generate_content(prompt)
                t = safe_extract_text(r)
                if t.strip():
                    return t
            except Exception as e:
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
            d = Document(f)
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
        st.session_state.vector_store, st.session_state.raw_text = vs, "\n".join(chunks)
        st.sidebar.success("Documents processed ✅")
    except Exception as e:
        st.error(f"Embedding error: {e}")

# ---------------- CHAINS ----------------
def get_conversational_chain():
    """QA chain with structured legal prompt for Document QA (RAG)."""
    p = PromptTemplate(
    template="""
You are LawPilot, a professional legal research assistant.
Use ONLY the provided context from Indian Supreme Court judgments to answer the user's question.
Do NOT add any information from outside sources.
Answer in a detailed, accurate, and structured way using the format below.
If any information is not available in the context, write 'Not specified in retrieved context.'

Question: {question}
Context: {context}

Answer in the following format:
Case Name: 
Year: 
Key Holdings: 
Reasoning: 
Majority/Dissent Points: 
Relevant Articles/Citations: 
Notes: 
""",
    input_variables=["context", "question"],
)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, 
        google_api_key=api_key
    )

    return load_qa_chain(model, chain_type="stuff", prompt=p)

def get_summary_chain():
    m = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    return load_summarize_chain(m, chain_type="map_reduce")

# ---------------- PRECEDENT PROMPT & HANDLER ----------------
precedent_prompt = PromptTemplate(
    template="""
You are LawPilot, a legal research assistant specializing in Indian Supreme Court precedents.
Use ONLY the retrieved precedent context to answer the user's question. Do NOT include information from outside sources.
If the required information is not present in the context, write 'Not specified in retrieved precedent.'

Question: {question}
Precedent Context: {context}

Provide a detailed, professional answer suitable for a legal professional in the following format. Each item should appear on a new line:

1️. Case Name:  
2️. Year of Decision:  
3️. Domain: (e.g., Constitutional Law, Consumer Law, Criminal Law, Employment, Environmental Law, Tax/GST/Income, Other)  
4. Subject Matter: (Brief description of what the case was about)  
5️. Judgment: (Outcome / decision of the Court)  
6️. Key Holdings / Ratio Decidendi:  
7️. Reasoning / Court Observations:  
8️. Majority / Dissenting Opinions:  
9️. Relevant Articles / Sections / Citations:  
10. Practical Notes / Implications for Legal Practice:  

Ensure clarity, precision, and professional tone suitable for referencing in legal work.
""",
    input_variables=["context", "question"]
)


def handle_precedent_qna(q):
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        return "⚠️ Legal corpus not loaded. Please load Supreme Court judgments first."
    
    docs = st.session_state.vector_store.similarity_search(q, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=precedent_prompt)

    res = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
    return res["output_text"]

# ---------------- HANDLERS ----------------
def handle_doc_qna(q):
    docs = st.session_state.vector_store.similarity_search(q)
    c = get_conversational_chain()
    r = c({"input_documents": docs, "question": q}, return_only_outputs=True)
    return r["output_text"]

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
    if "raw_text" not in st.session_state or not st.session_state.raw_text:
        st.error("Upload a document first.")
        return
    docs = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800).create_documents(
        [st.session_state.raw_text]
    )
    chain = get_summary_chain()
    with st.spinner("Summarizing..."):
        init = chain.run(docs)
        refined = robust_generate(f"Refine this summary as per '{instr}'.\n\n{init}")
    st.session_state.chat_history.append(("LawPilot", f"**Summary ({instr})**\n\n{refined}"))

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
        ins = st.text_input("Custom Instruction", "Provide a concise legal summary.")
        if st.button("Generate Summary"):
            generate_summary(ins)

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.rerun()

# ---------------- MAIN CHAT INTERFACE ----------------
st.title("⚖️ LawPilot – Legal AI Assistant")
chat_tab, trans_tab = st.tabs(["Chat", "Translate"])

# --- Chat Tab ---
with chat_tab:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role, avatar="👤" if role == "You" else "🤖"):
            st.markdown(msg)

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
