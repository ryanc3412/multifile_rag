"""
rag_indexer.py

Small, self-contained RAG indexer using OpenAI embeddings and FAISS for local
development. Designed to be easy to swap out the vector store for Supabase
later (see TODOs near SupabaseBackend and notes at the bottom).

Usage examples (also available via `cli.py`):
  - Index a directory of .docx files:
      python cli.py --index ./documents
  - Search the index:
      python cli.py --search "some phrase" --k 5

What to change for Supabase later:
  - Replace FaissBackend(...) instantiation with SupabaseBackend(...)
  - Move metadata persistence into Postgres `documents` table with a vector
    column (pgvector). See the TODO and SQL schema at the bottom of this file.

Requirements:
  - Python 3.10+
  - Uses .env for OPENAI_API_KEY (python-dotenv)
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging early so import-time errors can log cleanly
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import faiss; if not available, set to None and provide helpful errors later.
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover - environment may not have faiss during static checks
    faiss = None  # type: ignore
    logger.debug("faiss import failed: %s", e)

# Load .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. Embedding calls will fail until set.")

# Centralize embedding model here to avoid accidental mismatches between
# indexing and query-time embeddings. Change in one place only.
EMBEDDING_MODEL = "text-embedding-3-small"

# Fixed chunk size and overlap values optimized for grant writing RAG system
# Large enough to capture context while small enough for precise retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


class VectorStoreBackend(ABC):
    """
    Abstract vector store backend interface.

    Implementations must be unit-testable and provide methods to add vectors,
    run searches, and persist/load state.
    """

    @abstractmethod
    def add_vectors(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        """
        Add vectors with associated metadata. Embeddings should be 2D numpy array.

        Args:
            embeddings: shape (N, D)
            metadatas: list of N dicts containing at least source_file, chunk_id, start, end, text
        """

    @abstractmethod
    def search(self, embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Search and return list of tuples (idx, score). Score semantics depend on backend.

        For FaissBackend we return (idx, score) where score is a distance (lower is better).
        """

    @abstractmethod
    def save(self) -> None:
        """
        Persist index and metadata to disk or remote.
        """

    @abstractmethod
    def load(self) -> None:
        """
        Load index and metadata from disk or remote.
        """


@dataclass
class FaissBackend(VectorStoreBackend):
    """
    FAISS-based vector store backend that persists index + JSONL metadata.

    This implementation uses cosine similarity by normalizing embeddings and
    using an IndexFlatIP (inner product). We convert similarity to a distance
    score using `1 - similarity` so that lower scores mean closer (distance).
    """

    persist_dir: str = "indexes"
    index_filename: str = "faiss.index"
    meta_filename: str = "faiss_meta.jsonl"
    # If None, we create the FAISS index lazily when embeddings are first added.
    # Avoids hardcoding a default dimension which can mismatch newer embedding models.
    dimension: Optional[int] = None

    def __post_init__(self):
        self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / self.index_filename
        self.meta_path = self.persist_dir / self.meta_filename
        self._metadatas: list[dict] = []
        self._index: Any = None
        # _d holds the configured/index dimension. None means unknown until we see embeddings.
        self._d: Optional[int] = self.dimension
        # deterministic seed for tests (faiss IndexFlat doesn't use RNG, but keep for future)
        np.random.seed(42)

        # If persisted index exists, load it (this will set _d). Otherwise, don't
        # create an index now if dimension is unknown — wait until first add_vectors
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.load()
            except Exception:
                logger.exception("Failed to load existing index; starting fresh.")
                # If a default dimension was provided, create an index; otherwise lazy-create later
                if self._d is not None:
                    self._create_index(self._d)

    def _create_index(self, d: int):
        """Create a FAISS index of dimension d. d must be a positive integer."""
        if not isinstance(d, int) or d <= 0:
            raise ValueError("Dimension must be a positive integer to create FAISS index")
        self._d = d
        # Use inner product over normalized vectors to represent cosine similarity
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FaissBackend")
        self._index = faiss.IndexFlatIP(d)
        logger.info("Created new FAISS IndexFlatIP with dimension=%d", d)

    def add_vectors(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D numpy array")
        n, d = embeddings.shape
        
        if len(metadatas) != n:
            raise ValueError(f"Metadata count ({len(metadatas)}) must match embedding count ({n})")
        
        # Validate metadata structure
        required_keys = {"source_file", "chunk_id", "text", "page_number"}
        for i, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                raise ValueError(f"Metadata {i} must be a dictionary")
            missing_keys = required_keys - set(meta.keys())
            if missing_keys:
                raise ValueError(f"Metadata {i} missing required keys: {missing_keys}")
        
        if self._index is None:
            self._create_index(d)
        if d != self._d:
            raise ValueError(f"Embedding dimension mismatch: index expects {self._d} got {d}")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = embeddings / norms

        # Append to faiss
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FaissBackend")
        self._index.add(emb_norm.astype(np.float32))

        # Persist metadata in memory and append to file on save
        self._metadatas.extend(metadatas)
        logger.info("Added %d vectors to FAISS index (total=%d)", n, self._index.ntotal)

    def search(self, embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        # normalize
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        emb_norm = (embedding / norm).astype(np.float32)

        D, I = self._index.search(emb_norm, k)
        # D is similarity (inner product); convert to distance where lower is better
        results: list[tuple[int, float]] = []
        for idx, sim in zip(I[0], D[0]):
            if idx == -1:
                continue
            score = float(1.0 - float(sim))
            results.append((int(idx), score))
        return results

    def save(self) -> None:
        # Persist FAISS index
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FaissBackend")
        faiss.write_index(self._index, str(self.index_path))
        # Persist metadata as JSONL
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            for m in self._metadatas:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        logger.info("Saved FAISS index to %s and metadata to %s", self.index_path, self.meta_path)

    def load(self) -> None:
        # Load metadata
        with open(self.meta_path, "r", encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        self._metadatas = lines
        # Infer dimension from first embedding if stored; otherwise rely on self._d
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FaissBackend")
        self._index = faiss.read_index(str(self.index_path))
        self._d = self._index.d
        logger.info("Loaded FAISS index (ntotal=%d) and %d metadata entries", self._index.ntotal, len(self._metadatas))

    def get_metadata(self, idx: int) -> dict:
        return self._metadatas[idx]


class SupabaseBackend(VectorStoreBackend):
    """
    Stub for future Supabase vector store. Implement the same methods as FaissBackend.

    TODO: Implement actual HTTP/pg client calls. When porting:
      - Store chunk text in `content` column
      - Store metadata JSONB in `metadata` column
      - Store embedding in pgvector column `embedding`
      - Use Supabase vector search or raw SQL `ORDER BY embedding <=> query_embedding`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SupabaseBackend is a stub. Replace FaissBackend with SupabaseBackend here when ready.")

    def add_vectors(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        raise NotImplementedError()

    def search(self, embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        raise NotImplementedError()

    def save(self) -> None:
        raise NotImplementedError()

    def load(self) -> None:
        raise NotImplementedError()


def embed_texts(texts: list[str], model: str | None = None, batch_size: int = 50, max_retries: int = 3, retry_backoff: float = 1.0) -> np.ndarray:
    """
    Embed a list of texts using OpenAI embeddings API in batches (50 chunks at a time).

    Returns:
        numpy.ndarray of shape (len(texts), dim)
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    
    if not texts:
        raise ValueError("No texts provided for embedding")
    
    # Validate text lengths (OpenAI has limits)
    max_tokens = 8191  # text-embedding-3-small limit
    for i, text in enumerate(texts):
        if len(text) > max_tokens * 4:  # Rough estimate: 4 chars per token
            logger.warning("Text %d is very long (%d chars), may exceed token limits", i, len(text))

    if model is None:
        model = EMBEDDING_MODEL

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embs: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch = texts[i : i + batch_size]
        
        logger.info("Processing batch %d/%d (%d texts)", batch_num, total_batches, len(batch))
        
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(input=batch, model=model)
                batch_embs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
                all_embs.extend(batch_embs)
                logger.debug("Successfully embedded batch %d/%d", batch_num, total_batches)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    logger.exception("Failed to get embeddings after %d attempts for batch %d", attempt, batch_num)
                    raise
                wait = retry_backoff * (2 ** (attempt - 1))
                logger.warning("Embedding request failed (attempt=%d, batch=%d). Retrying in %.1fs. Error: %s", 
                              attempt, batch_num, wait, str(e))
                time.sleep(wait)

    logger.info("Successfully generated embeddings for %d texts", len(texts))
    return np.vstack(all_embs)


def index_documents(dir_path: str, backend: VectorStoreBackend, persist_path: str | None = None):
    """Index files in dir_path using provided backend and persist results.

    Args:
        dir_path: directory containing documents (.docx, .pdf)
        backend: VectorStoreBackend implementation (FaissBackend for now)
        persist_path: optional path to persist index/metadata; used by backend
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"dir_path {dir_path} does not exist or is not a directory")

    # Validate configuration
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for indexing. Please set it in your .env file.")
    
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0:
        raise ValueError(f"Invalid chunk configuration: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")

    # Set up text splitter with improved separators for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    
    texts: list[str] = []
    metadatas: list[dict] = []
    chunk_id = 0
    processed_files = 0
    failed_files = 0

    # Process both .docx and .pdf files
    supported_files = list(p.glob("*.docx")) + list(p.glob("*.pdf"))
    supported_files.sort()  # Sort for consistent ordering
    
    if not supported_files:
        logger.warning("No supported files (.docx, .pdf) found in %s", dir_path)
        return
    
    logger.info("Found %d files to process in %s", len(supported_files), dir_path)

    for file_path in supported_files:
        try:
            logger.info("Processing file: %s", file_path.name)
            
            # Get appropriate loader based on file type
            if file_path.suffix.lower() == ".docx":
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif file_path.suffix.lower() == ".pdf":
                loader = PyMuPDFLoader(str(file_path))
            else:
                logger.warning("Unsupported file type: %s", file_path)
                continue

            # Load documents - handle both single and multi-page documents
            docs = loader.load()
            if not docs:
                logger.warning("No content extracted from %s", file_path.name)
                continue
            
            file_chunk_count = 0
            # Process each document (PDFs may have multiple pages as separate docs)
            for doc_idx, doc in enumerate(docs):
                if not doc.page_content or not doc.page_content.strip():
                    logger.warning("Empty content in %s (doc %d)", file_path.name, doc_idx)
                    continue
                
                # Extract page number from metadata
                page_number = None
                if file_path.suffix.lower() == ".pdf":
                    # PyMuPDFLoader provides page info in metadata
                    page_number = doc.metadata.get('page', 0) + 1  # Convert 0-based to 1-based
                    logger.debug("PDF page number extracted: %d for doc %d", page_number, doc_idx)
                elif file_path.suffix.lower() == ".docx":
                    # Word documents don't have reliable page info from UnstructuredWordDocumentLoader
                    # For multi-page Word docs, we can estimate based on content length
                    # Rough estimate: ~500-1000 chars per page for typical documents
                    content_length = len(doc.page_content)
                    estimated_pages = max(1, content_length // 750)  # ~750 chars per page estimate
                    
                    # For now, we'll use a simple approach: if this is the first doc from a Word file,
                    # assume it's page 1. In the future, we could implement more sophisticated
                    # page detection by analyzing the content for page breaks.
                    page_number = 1
                    logger.debug("Word doc estimated pages: %d, using page 1 for doc %d", estimated_pages, doc_idx)
                
                # Fallback: if page_number is still None, set to 1
                if page_number is None:
                    page_number = 1
                    logger.warning("Could not determine page number for %s doc %d, defaulting to 1", file_path.name, doc_idx)
                
                chunks = text_splitter.create_documents([doc.page_content])
                
                if not chunks:
                    logger.warning("No chunks created from %s (doc %d)", file_path.name, doc_idx)
                    continue
                
                for chunk in chunks:
                    # Skip very short chunks that are likely not meaningful
                    if len(chunk.page_content.strip()) < 50:
                        continue
                        
                    texts.append(chunk.page_content)
                    metadatas.append({
                        "source_file": file_path.name,
                        "chunk_id": chunk_id,
                        "start_offset": 0,  # LangChain doesn't provide char offsets
                        "end_offset": len(chunk.page_content),
                        "text": chunk.page_content,
                        "embedding_model": EMBEDDING_MODEL,
                        "file_size": file_path.stat().st_size,
                        "file_modified": file_path.stat().st_mtime,
                        "page_number": page_number,
                    })
                    chunk_id += 1
                    file_chunk_count += 1
            
            logger.info("Created %d chunks from %s", file_chunk_count, file_path.name)
            processed_files += 1
                
        except Exception as e:
            logger.exception("Failed to process file %s: %s", file_path, str(e))
            failed_files += 1
            continue

    if not texts:
        logger.warning("No documents found to index in %s", dir_path)
        return
    
    logger.info("Processing complete: %d files processed, %d failed, %d total chunks", 
                processed_files, failed_files, len(texts))

    try:
        logger.info("Generating embeddings for %d chunks...", len(texts))
        embeddings = embed_texts(texts)
        logger.info("Adding vectors to backend...")
        backend.add_vectors(embeddings, metadatas)
        logger.info("Saving index...")
        backend.save()
        logger.info("Indexing completed successfully!")
    except Exception as e:
        logger.exception("Failed during embedding or indexing: %s", str(e))
        raise


def search(query: str, backend: FaissBackend, k: int = 5) -> list[dict]:
    """Search the vector store and return JSON-serializable hits.

    Returns list of dicts with keys: quote, source_file, chunk_id, start_offset, end_offset, score, page_number
    Score is a distance where lower is better (0 = identical under cosine similarity approximation).
    """
    # Use the centralized embedding model for queries as well. This avoids the
    # common pitfall where index_documents used one model and search used another.
    q_emb = embed_texts([query], model=EMBEDDING_MODEL)

    # defensive runtime check: ensure embedding dimension matches index
    if backend._d is not None and q_emb.shape[1] != backend._d:
        raise RuntimeError(
            f"Embedding dimension mismatch: query embedding dim={q_emb.shape[1]} "
            f"but index expects dim={backend._d}. This often means the embedding model "
            "used for indexing differs from the query embedding model."
        )
    hits = backend.search(q_emb[0], k)
    results: list[dict] = []
    for idx, score in hits:
        meta = backend.get_metadata(idx)
        results.append(
            {
                "quote": meta.get("text"),
                "source_file": meta.get("source_file"),
                "chunk_id": meta.get("chunk_id"),
                "start_offset": meta.get("start_offset"),
                "end_offset": meta.get("end_offset"),
                "score": score,
                "embedding_model": meta.get("embedding_model"),
                "page_number": meta.get("page_number"),
            }
        )
    return results


# SQL example for Supabase migration (store metadata + embedding in Postgres with pgvector):
#
# CREATE TABLE documents (
#   id bigserial PRIMARY KEY,
#   content text NOT NULL,
#   metadata jsonb,
#   embedding vector(1536)
# );
# CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
#
# When switching to SupabaseBackend:
#  - Insert rows into `documents` with content=chunk text, metadata JSONB containing
#    source_file/chunk_id/start_offset/end_offset, and embedding vector.
#  - To search, compute query embedding, then run `SELECT *, embedding <=> query_embedding AS distance
#    FROM documents ORDER BY distance LIMIT k` (pgvector operator `<=>` gives L2 distance).
