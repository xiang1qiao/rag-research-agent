# RAG Research Agent

A production-oriented Research Agent built with FastAPI, ChromaDB, DeepSeek and LangGraph.

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline and extends it into a Tool Calling Agent with Query Rewrite, Conversation Memory, Rerank, and LangGraph-based multi-node workflows.

---

## Overview

Traditional LLMs often suffer from hallucinations and lack access to external knowledge.

This project addresses these limitations by combining:

* Retrieval-Augmented Generation (RAG)
* Query Rewrite
* Conversation Memory
* CrossEncoder Rerank
* Tool Calling Agent
* LangGraph Workflow

The system can retrieve information from uploaded documents, perform calculations, invoke external tools, and generate grounded answers with source attribution.

---

## Architecture

```text
                    User Question
                           │
                           ▼
                    Planner Agent
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   search_docs()    calculator()     web_search()
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
                  Answer Generator
                           │
                           ▼
                    Reviewer Agent
                           │
                           ▼
                     Final Response
```

---

## Core Features

### Document Knowledge Base

* Upload TXT / PDF / DOCX files
* Automatic document parsing
* Text chunking
* Embedding generation
* Persistent vector storage with ChromaDB

### Retrieval-Augmented Generation

* Semantic retrieval based on embeddings
* Context injection into LLM prompts
* Source attribution support

### Query Rewrite

The system automatically rewrites ambiguous questions into standalone retrieval-friendly queries.

Example:

```text
Q1: Who is Zhang San?

Q2: What does he research?
```

Rewritten as:

```text
What are Zhang San's research areas?
```

This significantly improves retrieval accuracy in multi-turn conversations.

### Conversation Memory

The system stores recent conversation history and uses it as additional context during retrieval and answer generation.

Supported scenarios:

```text
User: Who is Zhang San?

Assistant: ...

User: What technologies does he use?
```

### CrossEncoder Rerank

Retrieval workflow:

```text
User Question
      │
      ▼
Vector Recall (Top 10)
      │
      ▼
CrossEncoder Rerank
      │
      ▼
Top 4 Relevant Chunks
      │
      ▼
Answer Generation
```

This improves answer relevance and reduces noisy retrieval results.

### Tool Calling Agent

The Agent dynamically selects tools according to user intent.

Available tools:

| Tool        | Description                    |
| ----------- | ------------------------------ |
| search_docs | Knowledge base retrieval       |
| calculator  | Mathematical calculation       |
| web_search  | External information retrieval |

Example:

```text
Question:
123 * 456

Tool:
calculator
```

```text
Question:
What technologies does Li Mu know?

Tool:
search_docs
```

### LangGraph Multi-Agent Workflow

Implemented using LangGraph StateGraph.

Workflow:

```text
Planner
   ↓
Tool Executor
   ↓
Answer Generator
   ↓
Reviewer
```

Each node has a dedicated responsibility, enabling modular and extensible agent orchestration.

---

## Tech Stack

### Backend

* Python
* FastAPI

### Retrieval

* ChromaDB
* Sentence Transformers

### Rerank

* CrossEncoder

### LLM

* DeepSeek API

### Agent Framework

* LangGraph

### Frontend

* HTML
* CSS
* JavaScript

---

## Project Structure

```text
rag-research-agent
│
├── app
│   ├── main.py
│   ├── rag.py
│   ├── agent.py
│   ├── graph_agent.py
│   ├── tools.py
│   ├── document_loader.py
│   ├── config.py
│   └── index.html
│
├── uploads
├── vector_db
├── requirements.txt
├── README.md
└── .env
```

---

## Workflow

### RAG Pipeline

```text
Document Upload
      │
      ▼
Document Parsing
      │
      ▼
Chunk Splitting
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB Storage
      │
      ▼
Question Retrieval
      │
      ▼
Rerank
      │
      ▼
DeepSeek Answer Generation
```

### Agent Pipeline

```text
User Question
      │
      ▼
Planner
      │
      ▼
Tool Selection
      │
      ▼
Tool Execution
      │
      ▼
Answer Generation
      │
      ▼
Review
      │
      ▼
Final Response
```

---

## Highlights

* Built a complete RAG pipeline from document ingestion to answer generation.
* Implemented Query Rewrite for retrieval optimization.
* Designed Conversation Memory for multi-turn interactions.
* Integrated CrossEncoder Rerank to improve retrieval quality.
* Developed a Tool Calling Agent capable of dynamic tool selection.
* Built a multi-node Agent workflow using LangGraph.
* Added source attribution to improve answer explainability.

---

## Future Improvements

* Hybrid Search (BM25 + Vector Search)
* MCP (Model Context Protocol)
* Multi-Knowledge-Base Management
* Multi-Agent Collaboration
* Streaming Responses
* Agent Evaluation Framework
* Long-Term Memory
* Human-in-the-Loop Review

---

## Demo

### Document Upload

![upload](assets/upload.png)

### RAG QA

![rag](assets/rag-demo.png)

### Query Rewrite

![rewrite](assets/query-rewrite.png)

### LangGraph Agent

![agent](assets/langgraph-agent.png)

---

## Author

xiang1qiao

AI Agent / RAG / LLM Application Development
