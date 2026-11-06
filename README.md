# 🚀 RAGOptimizer

**RAGOptimizer** is an experimental framework that evaluates different Retrieval-Augmented Generation (RAG) pipeline configurations to identify the best-performing setup for a given document and evaluation dataset.

It compares different **chunking strategies, embedding models, and retrieval methods**, then evaluates both **retrieval quality** and **final answer quality**.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │    User Uploads     │
                    │                     │
                    │ • Source Document   │
                    │ • Evaluation Dataset│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document Ingestion  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      Chunking Strategies       │
              │                                │
              │ Fixed-size │ Semantic          │
              │ Hierarchical │ Structure-aware │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Generate Chunk IDs  │
                    │ + Attach Metadata   │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Embedding Models         │
              │                                │
              │ MiniLM │ E5-small │ BGE-base   │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │      ChromaDB       │
                    │                     │
                    │ ID + Chunk Text     │
                    │ Metadata + Vector   │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Retrieval Methods        │
              │                                │
              │ Dense │ Hybrid │ HyDE          │
              │ Query Expansion │ Reranking    │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Top-K Retrieved     │
                    │ Chunks              │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
      ┌──────────────────────┐     ┌──────────────────────┐
      │ Retrieval Evaluation │     │  LLM Answer Generation│
      │                      │     │                      │
      │ Hit Rate@K           │     │ Question + Context   │
      │ Precision@K          │     │        ↓             │
      │ Recall@K             │     │ Generated Answer     │
      │ MRR                  │     └──────────┬───────────┘
      │ nDCG@K               │                │
      └──────────────────────┘                ▼
                                   ┌──────────────────────┐
                                   │ Response Evaluation  │
                                   │                      │
                                   │ Faithfulness         │
                                   │ Relevance            │
                                   │ Correctness          │
                                   │ Completeness         │
                                   │ Hallucination Rate   │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ Pipeline Comparison  │
                                   │ & Recommendation     │
                                   └──────────────────────┘
