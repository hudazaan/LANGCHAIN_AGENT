# Major League Hacking (MLH)- Global Hack Week: Open Source (October'25) 
## Workshop Agenda: 
- introduction to building your first agent using Gemini and Langchain by Quinn Dines. 

## Core Technologies/Stack used: 

- Programming Language: Python
- AI/ML Frameworks & Libraries: 
  - LangChain - Framework for building AI agents
    1. langchain-core - Core abstractions (messages, tools, runnables)
    2. langchain-huggingface - HuggingFace LLM integration (replaced langchain-google-genai)
    3. (removed langchain-tavily - Tavily web search integration) 
  - Pydantic - Data validation and settings management 
    1. pydantic 

- LLM Providers & APIs: 
  - HuggingFace (Meta LLaMA 3 / Zephyr models) (replaced gemini-2.0-flash-lite model)
    1. huggingface-hub - HuggingFace API client (replaced google-generativeai - Google's Generative AI Python SDK) 

- Vector Database & Embeddings: 
  - FAISS - Facebook AI Similarity Search for vector indexing
    1. faiss-cpu
  - Sentence Transformers - Text embeddings and semantic search
    1. Model: sentence-transformers/all-MiniLM-L6-v2 

- Frontend & UI
  - Gradio - Web UI library for ML demos 
    1. gradio (not fully implemented yet in main loop) 

- Utility Libraries: 
  1. python-dotenv - Environment variable and API key management
  2. pickle - Serialization for chunk storage (built-in) 
  3. pathlib, os, time, json, typing - Python Standard library modules (pre-installed, import direct)

- Web Search & APIs: 
  - (removed Tavily Search API - AI-optimized web search for agents)
  - Wikipedia REST API - Knowledge retrieval via wiki_lookup tool
  - Requests (requests) - HTTP client library

- External APIs Required: 
  - HuggingFace API (requires HF_TOKEN in .env) (replaced Google GEMINI_API_KEY) 
  - (removed Tavily Search API (requires TAVILY_API_KEY in .env))
  - Wikipedia REST API (public, no key required)

- Architecture & Patterns: 
  - ReAct Agent Pattern - Reason → Act → Observe loop
  - RAG (Retrieval-Augmented Generation) - Document chunking, embedding, and retrieval
  - Tool-based Architecture - Modular tools (wiki_lookup, retrieve_relevant_docs)
  - Retry Logic - Exponential backoff for API reliability
  - Rate Limiting - Throttling for API calls