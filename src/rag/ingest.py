"""Ingest knowledge base documents into Chroma vector store.

Chunking strategy (documented for the report):
- Splitter: RecursiveCharacterTextSplitter with markdown-aware separators.
- Chunk size: 600 characters, overlap: 100 characters.
- Why 600/100: policy docs are structured as short sections (1-3 paragraphs per topic).
  600 chars typically captures one complete policy point (e.g., "Standard Shipping" rules)
  without merging unrelated policies. 100-char overlap preserves sentence continuity at
  chunk boundaries, so a question about a rule split across chunks can still match.
  Smaller chunks (300) fragmented multi-condition rules; larger chunks (1000) diluted
  precision by mixing shipping and returns content in a single chunk.
- Metadata: each chunk carries `source` (filename), `doc_type` (policy|catalog|faq),
  and `category` for downstream metadata filtering.

Run:  python -m src.rag.ingest
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Paths ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_CHROMA_DIR = _PROJECT_ROOT / "chroma_db"

load_dotenv(_PROJECT_ROOT / ".env")

# --- Chunking parameters (justified above) ---
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Map filenames to metadata
_DOC_METADATA = {
    "shipping_policy.md": {"doc_type": "policy", "category": "shipping"},
    "returns_policy.md": {"doc_type": "policy", "category": "returns"},
    "electronics_catalog.md": {"doc_type": "catalog", "category": "electronics"},
    "faq.md": {"doc_type": "faq", "category": "general"},
}


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text using markdown-aware recursive splitting.

    Uses a simple implementation to avoid the langchain_text_splitters import
    which eagerly pulls in sentence_transformers and hangs on some platforms.
    """
    separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def _split_recursive(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = seps[0] if seps else ""
        remaining_seps = seps[1:] if seps else []

        if sep and sep in text:
            parts = text.split(sep)
        elif remaining_seps:
            return _split_recursive(text, remaining_seps)
        else:
            # Last resort: split by character count
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        # Merge small parts into chunks
        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If part itself is too big, recurse with finer separators
                if len(part) > chunk_size and remaining_seps:
                    chunks.extend(_split_recursive(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split_recursive(text, separators)

    # Add overlap: prepend the tail of the previous chunk to each chunk
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_tail = raw_chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + " " + raw_chunks[i])
        return overlapped

    return raw_chunks


def load_and_chunk_documents() -> list[Document]:
    """Load all knowledge base documents and split into chunks with metadata."""
    all_chunks = []

    doc_dirs = [
        _DATA_DIR / "policies",
        _DATA_DIR / "catalog",
    ]

    for doc_dir in doc_dirs:
        for filepath in sorted(doc_dir.glob("*.md")):
            filename = filepath.name
            metadata = _DOC_METADATA.get(filename, {"doc_type": "unknown", "category": "unknown"})

            text = filepath.read_text(encoding="utf-8")
            chunk_texts = _split_text(text)

            for chunk_text in chunk_texts:
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"source": filename, **metadata},
                ))

            logger.info(f"  {filename}: {len(chunk_texts)} chunks")

    logger.info(f"Total: {len(all_chunks)} chunks from {len(_DOC_METADATA)} documents")
    return all_chunks


def create_vectorstore(chunks: list[Document], embedding_fn=None):
    """Embed chunks and persist to Chroma."""
    from langchain_chroma import Chroma

    if embedding_fn is None:
        embedding_fn = _get_default_embeddings()

    # Clear existing DB for clean re-ingest
    if _CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(_CHROMA_DIR)
        logger.info(f"Cleared existing Chroma DB at {_CHROMA_DIR}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=str(_CHROMA_DIR),
        collection_name="techmart_kb",
    )

    logger.info(f"Chroma DB created at {_CHROMA_DIR} with {len(chunks)} vectors")
    return vectorstore


def load_vectorstore(embedding_fn=None):
    """Load an existing Chroma vectorstore from disk."""
    from langchain_chroma import Chroma

    if embedding_fn is None:
        embedding_fn = _get_default_embeddings()

    return Chroma(
        persist_directory=str(_CHROMA_DIR),
        embedding_function=embedding_fn,
        collection_name="techmart_kb",
    )


def _get_default_embeddings():
    """Return the default embedding function (OpenAI text-embedding-3-small)."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, str(_PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("=== TechMart Knowledge Base Ingestion ===\n")
    chunks = load_and_chunk_documents()

    print(f"\n--- Sample chunks (first 3) ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} [{chunk.metadata}]:")
        print(f"  {chunk.page_content[:200]}...")

    print(f"\n--- Creating vector store ---")
    vectorstore = create_vectorstore(chunks)
    print(f"\nDone. {len(chunks)} chunks indexed in Chroma at {_CHROMA_DIR}")
