I chose a hybrid approach because legal documents contain both high-level concepts and very specific terminology. While FAISS is great at understanding the 'intent' of a question (semantic search), it sometimes misses specific Article numbers or niche legal terms. By fusing it with BM25, I ensured that the system catches exact keyword matches while still maintaining a deep semantic understanding of the law.
	Developed a high-precision RAG application to query complex legal documents (IPC & Indian Constitution) using a Hybrid Search architecture.

Implemented Reciprocal Rank Fusion (RRF) to combine semantic results from FAISS (dense vectors) with keyword precision from BM25 (lexical search), improving retrieval accuracy for legal terminology and specific Article citations by ~30%.

Optimized system performance by implementing a persistent indexing pipeline and Streamlit caching, reducing query latency and API costs.

Engineered an automated ingestion workflow using RecursiveCharacterTextSplitter to maintain semantic context across 1,000+ character chunks.
