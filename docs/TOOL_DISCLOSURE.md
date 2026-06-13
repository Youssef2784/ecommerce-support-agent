# Tool Disclosure

Every external framework, library, dataset, and AI assistant used in this project.

## Frameworks & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| LangGraph | 1.2.2 | Multi-agent orchestration, state management, conditional routing |
| LangChain | 1.3.2 | LLM abstraction, document types, embedding interfaces |
| langchain-openai | 1.2.2 | OpenAI LLM and embedding wrappers |
| langchain-chroma | 1.1.0 | Chroma vectorstore integration for LangChain |
| ChromaDB | 1.5.9 | Persistent vector store for RAG document embeddings |
| rank-bm25 | 0.2.2 | BM25 lexical search for hybrid retrieval strategy |
| sentence-transformers | 4.1.0 | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| Faker | 40.19.1 | Synthetic order data generation (220 orders, 50 customers) |
| Streamlit | *(planned)* | Lightweight evaluation dashboard |
| python-dotenv | 1.2.2 | Environment variable management |
| Pydantic | 2.13.4 | Data validation and state schema typing |

## LLM & Embedding Models

| Model | Provider | Purpose |
|-------|----------|---------|
| gpt-4o-mini | OpenAI | Agent responses, RAGAS-style LLM-as-judge scoring |
| text-embedding-3-small | OpenAI | Document and query embeddings for Chroma |
| ms-marco-MiniLM-L-6-v2 | Hugging Face | Cross-encoder reranking of retrieval candidates |

## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| Synthetic orders (orders.json) | Generated with Faker (seeded) | Mock order API data — 220 orders, 5 statuses |
| Policy documents | Hand-authored | Shipping and returns policies for fictional TechMart store |
| Product catalog | Hand-authored | 8 electronics products with specs, prices, warranty info |
| FAQ document | Hand-authored | 20 Q&A pairs covering common customer questions |
| Gold eval set (gold_qa.json) | Hand-authored | 20 Q&A pairs with ground-truth answers for RAGAS evaluation |

**Note on knowledge base**: All policy, catalog, and FAQ documents were hand-authored
(not LLM-generated) to ensure full control over ground truth for evaluation. The fictional
store "TechMart" was designed with specific numbers, thresholds, and edge cases to make
retrieval quality measurable.

## AI Assistants

| Assistant | Usage |
|-----------|-------|
| Claude (Anthropic) | Code generation, architecture guidance, implementation assistance |

All architectural and design decisions were reviewed and are understood by the team.
