# Hanoi University (HANU) - AI Assistant
## Course: Advanced Topics in Information Technology (ATI) - Fall 2026

An AI-powered academic assistant for students and lecturers at Hanoi University, implementing **Temporal-Aware Retrieval-Augmented Generation (RAG)**, automated data ingestion with **SHA-256 change detection**, persistent vector storage via **ChromaDB**, and grounded answer generation with **Gemini 2.5 Flash**.

---

## 🏛️ System Architecture

```
                  ┌───────────────────────────────┐
                  │    University Academic API    │
                  │   (Announcements & Notices)   │
                  └───────────────┬───────────────┘
                                  │ (HTTP POST / Fetch)
                                  ▼
                    ┌───────────────────────────┐
                    │  Data Ingestion Pipeline  │
                    │   - HTML / Table Parser   │
                    │   - SHA-256 Hash Check    │
                    │   - Semantic Chunker      │
                    └─────────────┬─────────────┘
                                  │ (Upsert Chunks + Dates)
                                  ▼
┌──────────────┐          ┌───────────────┐          ┌───────────────────────┐
│  Student UI  │ ◄──────► │  FastAPI Hub  │ ◄──────► │  Persistent ChromaDB  │
│ (Streamlit)  │          │ (/chat, /sync)│          │  (Vector Collection)  │
└──────────────┘          └───────┬───────┘          └───────────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │    Temporal-Aware Ranker    │
                   │  1. Validity Filter (Dates) │
                   │  2. Recency Boost (Weights) │
                   │  3. Threshold Fallback Check│
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │      Gemini 2.5 Flash       │
                   │   Grounded Answer & Cites   │
                   └─────────────────────────────┘
```

---

## 📁 Project Directory Structure

```
d:/2026 - 2027 HANU/ATI/
├── backend/
│   ├── config.py              # Environment settings & threshold configs
│   ├── main.py                # FastAPI app with CORS, lifespan & APScheduler
│   ├── routes/
│   │   ├── chat.py            # POST /chat (question answering)
│   │   └── admin.py           # POST /sync, GET /status, GET /health
│   ├── services/
│   │   ├── ingestion.py       # HTML parsing, SHA-256 hash comparison, text chunking
│   │   ├── vector_db.py       # Persistent ChromaDB service
│   │   ├── temporal_ranker.py # Expiration filtering & recency boost ranking
│   │   ├── rag_engine.py      # Grounded prompt builder & Gemini 2.5 Flash caller
│   │   └── scheduler.py       # APScheduler periodic sync job
│   └── mock_portal/
│       └── portal_data.py     # Realistic Hanoi University announcements & notices
│
├── frontend/
│   └── app.py                 # Streamlit UI (Student Chat Tab + Admin Control Panel)
│
├── tests/
│   ├── test_system.py         # Unit tests for chunking, hashing, temporal logic
│   └── test_api.py            # Integration tests for FastAPI endpoints
│
├── .env                       # Local environment variables
├── .env.example               # Template environment variables
├── .gitignore                 # Git ignore rules (secrets & temp files)
├── requirements.txt           # Python package requirements
├── MIDTERM_REPORT.md          # Complete markdown documentation of the midterm report
└── README.md                  # Project documentation & run guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.11+** (Python 3.13 tested and supported)
- Windows PowerShell or Bash terminal

### 2. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

*(Or create a new one if cloning fresh: `python -m venv .venv` followed by `pip install -r requirements.txt`)*

### 3. Configure `.env`
Open `.env` and set your **Gemini API Key**:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ADMIN_API_KEY=hanu_admin_secret_2026
```
*(You can get a free key from [Google AI Studio](https://aistudio.google.com/). If left blank, the app will run with helpful simulated answers citing the verified database chunks).*

---

### 4. Running the Application

You can run the full system using **Streamlit** (which embeds both the conversational chat and administrative sync features) or run both the FastAPI backend and Streamlit in tandem:

#### Option A: Run Streamlit Interactive UI (Chat + Admin Panel)
```powershell
.\.venv\Scripts\streamlit run frontend/app.py
```
Open your browser at: **`http://localhost:8501`**

#### Option B: Run FastAPI REST Backend Server
```powershell
.\.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
```
Interactive API documentation (Swagger UI) is available at: **`http://127.0.0.1:8000/docs`**

---

## 🧪 Running the Test Suite

Run all unit tests and API integration tests:

```powershell
# Run core system unit tests (chunking, hashing, temporal filter, ChromaDB)
.\.venv\Scripts\python -m unittest tests/test_system.py

# Run FastAPI endpoint integration tests
.\.venv\Scripts\python -m unittest tests/test_api.py
```

---

## 💡 Key Highlights for the Midterm Presentation

1. **Temporal-Aware Ranking**: Prevents students from receiving outdated tuition rates or expired exam dates by filtering `expiration_date < today` and giving recency boosts to newly released notices.
2. **Incremental Ingestion (SHA-256)**: Re-indexing skips identical documents, saving compute and storage while maintaining continuous synchronization via APScheduler.
3. **Admin Control Panel**: Real-time tracking of ingested vectors, sync run logs, and manual "Sync Now" overrides protected by an Admin Secret Key.
4. **Zero-Hallucination Fallback**: Checks relevance threshold against cosine distance; if no relevant university document exists, advises the student to contact the FIT office directly (Room 201 Building C).
