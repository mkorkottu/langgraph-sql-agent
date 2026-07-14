# 🚂 LangGraph SQL Agent

A production-style conversational AI agent that lets users query railway 
infrastructure databases using plain English. Built as a POC demonstrating 
LangGraph, RAG with Hybrid Search, conversation memory, FastAPI, and React.

## 🎯 What it does

Ask questions in plain English and get intelligent answers backed by real data:

- *"Which track segments have CRITICAL status?"*
- *"How many of those are in the Clovis subdivision?"*
- *"What maintenance has been done on those segments?"*

Claude remembers the full conversation — "those" in Q2 refers to Q1 results 
automatically, without you repeating context.

## 🏗️ Architecture

React (frontend)
↓ HTTP/REST
FastAPI (backend)
↓
LangGraph Agent
↓
Hybrid RAG Search
(BM25 + Dense Semantic + CrossEncoder Reranker)
↓
Claude (Anthropic) → SQL Generation → SQLite → Answer

## ✨ Features

- **Natural language to SQL** — Claude generates SQL from plain English questions
- **Hybrid RAG search** — BM25 keyword + dense semantic + CrossEncoder reranker
- **Full conversation memory** — multi-turn context across questions
- **SQL validation** — blocks unsafe queries (DROP, DELETE, UPDATE etc.)
- **Self-reflection** — retries with corrected SQL on query failure
- **React chat interface** — dark theme, markdown rendering, expandable SQL/data/RAG panels

## 🛠️ Tech Stack

**Backend**
- Python 3.11
- LangGraph — agent orchestration and state management
- FastAPI — REST API
- Anthropic Claude (claude-sonnet-4-6) — SQL generation and answer formatting
- ChromaDB — vector store for RAG
- Sentence Transformers (all-MiniLM-L6-v2) — dense embeddings
- CrossEncoder (ms-marco-MiniLM-L-6-v2) — reranker
- rank-bm25 — keyword search
- SQLite — local database

**Frontend**
- React 18
- Axios — HTTP client
- react-markdown — markdown rendering

## 📁 Project Structure

langgraph-sql-agent/
├── backend/
│   ├── agent.py          # LangGraph agent with memory + RAG
│   ├── embeddings.py     # Hybrid search (BM25 + Semantic + Reranker)
│   ├── main.py           # FastAPI REST API
│   ├── setup_db.py       # Mock railway database setup
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── App.js        # React chat interface
│       └── App.css       # Dark theme styling
│
├── .gitignore
└── README.md

## 🚀 Running Locally

### Prerequisites
- Python 3.11
- Node.js 18+
- Anthropic API key

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Create mock railway database
python setup_db.py

# Build RAG knowledge base
python embeddings.py

# Start FastAPI server
python -m uvicorn main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

App runs at `http://localhost:3000`, API at `http://localhost:8000`

## 🔍 How Hybrid Search Works

1. **BM25 keyword search** — finds exact word matches (like Ctrl+F)
2. **Dense semantic search** — finds meaning-based matches via embeddings
3. **Merge** — combines both result sets and deduplicates
4. **CrossEncoder reranker** — scores all candidates and picks the best 3

This outperforms pure semantic search on specific terms (segment IDs, 
maintenance types) while still handling natural language questions well.

## 💡 Example Multi-Turn Conversation

User: Which segments have CRITICAL status?
→ Claude queries DB, returns 10 CRITICAL segments across subdivisions
User: How many of those are in Clovis?
→ Claude remembers "those" = CRITICAL segments, finds 2 in Clovis
→ RAG adds: "Clovis known for soil expansion issues"
User: What maintenance was done on those?
→ Claude remembers "those" = 2 Clovis segments specifically
→ Returns maintenance history with costs and completion statu

## 🎓 Key Concepts Demonstrated

- LangGraph state machine with conditional edges
- RAG pipeline with hybrid retrieval and reranking
- Multi-turn conversational memory via message history
- Production MLOps patterns (validation, self-reflection, fault tolerance)
- Full-stack AI application (React + FastAPI + LLM)