# ⚖️ **LawPilot — Your Legal AI Guide**
### 🤖 *An AI-powered Legal Research Assistant built with Streamlit and Gemini API*

---

## 📌 Project Overview

**LawPilot** is a Streamlit-based **AI Legal Research and Assistance Platform** that helps users interact intelligently with their legal documents and queries.

It enables users to:

- 📑 **Upload and process legal documents** (`PDF` / `DOCX`)  
- 🤖 **Ask general legal questions** or **context-aware questions** from uploaded documents  
- 📝 **Generate detailed and customized summaries** of legal texts  
- 🌍 **Translate English legal content** into regional languages (**Hindi** and **Marathi**)  
- 🧠 **Leverage AI-based retrieval and summarization** using **Gemini models** and **local vector embeddings**

LawPilot integrates **Google Gemini 2.5 Flash**, **FAISS** for semantic retrieval, and **HuggingFace MiniLM embeddings** for offline, quota-free document processing.

---

## 🛠 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Framework** | Streamlit |
| **Language** | Python 3.10+ |
| **AI Model** | Google Gemini 2.5 Flash (`google-generativeai`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **NLP / Pipeline** | LangChain (QA, Summarization, PromptTemplates) |
| **Document Parsing** | PyMuPDF (`fitz`), python-docx |
| **Environment Variables** | python-dotenv |
| **UI Enhancements** | Custom Streamlit CSS styling |
| **Translation & Summary** | Gemini model-based generation |

---

## 📸 LawPilot Interface Overview

Here’s how **LawPilot** functions in action:

| Feature | Screenshot |
|----------|-------------|
| **Home Page** | ![Home Page](Outputs/Home.png) |
| **General Q&A** | ![General Q&A](Outputs/GeneralQnA.png) |
| **Document Q&A** | ![Document Q&A](Outputs/DocumentQnA.png) |
| **Summarization** | ![Summarization](Outputs/Summarization.png) |
| **Translation** | ![Translation](Outputs/Translation.png) |

---

## 🔄 System Workflow

LawPilot follows a **Retrieval-Augmented Generation (RAG)** pipeline that works in three major stages:

![Architecture Diagram](Outputs/Architecture_Diagram.png)

1. 🧾 **Document Processing & Indexing**  
   Uploaded legal documents (`.pdf`, `.docx`) are extracted using PyMuPDF / python-docx, split into chunks, and stored as embeddings in **FAISS** using **HuggingFace MiniLM**.

2. 🔍 **Retrieval**  
   For document-based queries, LawPilot performs **semantic search** to fetch the most relevant sections from the document vector store.

3. 🤖 **Generation & Response**  
   The retrieved context or direct user query is passed to **Gemini 2.5 Flash**, which generates an accurate, context-aware answer or summary.

---

## 📊 Key Features

| Feature | Description |
|----------|-------------|
| 📂 **Multi-Document Upload** | Upload multiple `.pdf` or `.docx` files for simultaneous analysis |
| 💬 **Two-Mode Q&A** | Ask either general legal queries or document-specific questions |
| 📝 **Custom Summaries** | Create concise or detailed summaries as per user instructions |
| 🌍 **Language Translation** | Translate legal content to **Hindi** or **Marathi** |
| 💾 **Offline Embeddings** | Uses free local `sentence-transformers` instead of paid API embeddings |
| 🧠 **Smart Retrieval** | Combines semantic and contextual understanding for accuracy |
| 🖼️ **Modern UI** | Minimal, responsive Streamlit interface with custom font and theme |

---

## 📁 Folder Structure

LawPilot/
├── app.py # Main Streamlit app
├── requirements.txt # List of dependencies
├── .env # API key (not committed)
├── logo.png # App logo (used in sidebar)
├── Outputs/ # Screenshots and diagrams
│ ├── Home.png
│ ├── GeneralQnA.png
│ ├── DocumentQnA.png
│ ├── Summarization.png
│ ├── Translation.png
│ └── Architecture_Diagram.png
├── sample_docs/ # Test legal documents
│ ├── Contract.pdf
│ └── Judgement.docx
├── .gitignore
├── LICENSE
└── README.md


---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Git**
- A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

---

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/sanketjadhav09/ClearClause-Legal-AI-Assistant.git
    cd ClearClause-Legal-AI-Assistant
    ```

2.  **Create and Activate a Virtual Environment**
    - This keeps the project dependencies isolated.

    - **On Windows:**
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - **On macOS/Linux:**
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up the Environment Variable**
    - Create a file named `.env` in the root of the project folder.
    - Add your Google Gemini API key to this file as follows:
      ```
      API_KEY="YOUR_SECRET_API_KEY_HERE"
      ```

---

## 💻 Usage

Once the installation is complete, you can run the application locally with a single command:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to interact with the ClearClause Legal AI Assistant.

---