# RAG Research Agent

基于 FastAPI + ChromaDB + DeepSeek + LangGraph 构建的 Research Agent。

支持：

* 文档上传与知识库构建
* RAG 检索增强问答
* Query Rewrite
* Conversation Memory
* CrossEncoder Rerank
* Tool Calling
* LangGraph 多节点 Agent Workflow

## Architecture

User
↓
Planner
↓
Tool Selection
↓
Search Docs / Calculator / Web Search
↓
Answer Generator
↓
Reviewer
↓
Final Answer

## Features

* PDF / DOCX / TXT 文档解析
* Embedding 向量化存储
* ChromaDB 检索
* DeepSeek 大模型回答
* Query Rewrite
* Multi-turn Conversation Memory
* CrossEncoder Rerank
* Tool Calling Agent
* LangGraph Workflow

## Tech Stack

Python · FastAPI · ChromaDB · SentenceTransformers · CrossEncoder · DeepSeek API · LangGraph
