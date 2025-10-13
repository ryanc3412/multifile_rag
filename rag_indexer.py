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
from typing import List, Tuple, Any

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document

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


class VectorStoreBackend(ABC):
    """
    Abstract vector store backend interface.

    Implementations must be unit-testable and provide methods to add vectors,
    run searches, and persist/load state.
    """

    @abstractmethod
    def add_vectors(self, embeddings: np.ndarray, metadatas: List[dict]) -> None:
        """
        Add vectors with associated metadata. Embeddings should be 2D numpy array.

        Args:
            embeddings: shape (N, D)
            metadatas: list of N dicts containing at least source_file, chunk_id, start, end, text
        """

    @abstractmethod
    def search(self, embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
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
    dimension: int = 1536  # default for many OpenAI embedding models; updated on first add

    def __post_init__(self):
        self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / self.index_filename
        self.meta_path = self.persist_dir / self.meta_filename
        self._metadatas: List[dict] = []
        self._index: Any = None
        self._d = self.dimension
        # deterministic seed for tests (faiss IndexFlat doesn't use RNG, but keep for future)
        np.random.seed(42)

        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.load()
            except Exception:
                logger.exception("Failed to load existing index; starting fresh.")
                self._create_index(self._d)
        else:
            self._create_index(self._d)

    def _create_index(self, d: int):
        self._d = d
        # Use inner product over normalized vectors to represent cosine similarity
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FaissBackend")
        self._index = faiss.IndexFlatIP(d)
        logger.info("Created new FAISS IndexFlatIP with dimension=%d", d)

    def add_vectors(self, embeddings: np.ndarray, metadatas: List[dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D numpy array")
        n, d = embeddings.shape
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

    def search(self, embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
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
        results: List[Tuple[int, float]] = []
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

    def add_vectors(self, embeddings: np.ndarray, metadatas: List[dict]) -> None:
        raise NotImplementedError()

    def search(self, embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError()

    def save(self) -> None:
        raise NotImplementedError()

    def load(self) -> None:
        raise NotImplementedError()


def embed_texts(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 50, max_retries: int = 3, retry_backoff: float = 1.0) -> np.ndarray:
    """
    Embed a list of texts using OpenAI embeddings API in batches (50 chunks at a time).

    Returns:
        numpy.ndarray of shape (len(texts), dim)
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(input=batch, model=model)
                batch_embs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
                all_embs.extend(batch_embs)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    logger.exception("Failed to get embeddings after %d attempts", attempt)
                    raise
                wait = retry_backoff * (2 ** (attempt - 1))
                logger.warning("Embedding request failed (attempt=%d). Retrying in %.1fs. Error: %s", attempt, wait, str(e))
                time.sleep(wait)

    return np.vstack(all_embs)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[dict]:
    """
    Split text into overlapping chunks trying to preserve sentence boundaries.

    Args:
        text: The text to split into chunks
        chunk_size: Maximum characters per chunk (default: 300)
        overlap: Number of characters to overlap between chunks (default: 50)
    
    The function tries to split on natural boundaries in this order:
    1. Newlines (\n)
    2. End of sentences (. ? !)
    3. Spaces between words
    4. Hard cut if no natural breaks found

    Returns list of dicts: {'text':..., 'start': int, 'end': int}
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    separators = ["\n", ". ", "? ", "! "]

    length = len(text)
    start = 0
    chunks: List[dict] = []
    chunk_id = 0

    while start < length:
        end = min(start + chunk_size, length)
        window = text[start:end]
        split_pos = None
        # try to split near the end of window on preferred separators
        for sep in separators:
            pos = window.rfind(sep)
            if pos != -1 and pos + len(sep) >= chunk_size - 100:  # favor near end
                split_pos = start + pos + len(sep)
                break
        if split_pos is None and end < length:
            # as fallback try to find last space
            pos = window.rfind(" ")
            if pos != -1 and pos > 0:
                split_pos = start + pos

        if split_pos is None:
            split_pos = end

        chunk_text_str = text[start:split_pos]
        chunks.append({"text": chunk_text_str, "start": start, "end": split_pos, "chunk_id": chunk_id})
        chunk_id += 1

        # move forward accounting for overlap
        prev_start = start
        # desired next start to include `overlap` characters from the previous chunk
        next_start = split_pos - overlap
        # ensure we always make forward progress; if next_start would not advance
        # beyond prev_start, fall back to split_pos (no overlap) to avoid infinite loops
        if next_start <= prev_start:
            start = split_pos
        else:
            start = next_start

    return chunks


def index_documents(dir_path: str, backend: VectorStoreBackend, persist_path: str | None = None, chunk_size: int = 600, overlap: int = 120):
    """Index all .docx files in dir_path using provided backend and persist results.

    Args:
        dir_path: directory containing .docx files
        backend: VectorStoreBackend implementation (FaissBackend for now)
        persist_path: optional path to persist index/metadata; used by backend
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"dir_path {dir_path} does not exist or is not a directory")

    texts: List[str] = []
    metadatas: List[dict] = []

    for docx_file in sorted(p.glob("*.docx")):
        try:
            doc = Document(docx_file)
        except Exception:
            logger.exception("Failed to open %s; skipping", docx_file)
            continue
        full_text = "\n".join(p.text for p in doc.paragraphs)
        if not full_text.strip():
            logger.info("File %s is empty; skipping", docx_file.name)
            continue

        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            texts.append(c["text"])
            metadatas.append({
                "source_file": docx_file.name,
                "chunk_id": int(c["chunk_id"]),
                "start_offset": int(c["start"]),
                "end_offset": int(c["end"]),
                "text": c["text"],
            })

    if not texts:
        logger.warning("No documents found to index in %s", dir_path)
        return

    embeddings = embed_texts(texts)
    backend.add_vectors(embeddings, metadatas)
    backend.save()


def search(query: str, backend: FaissBackend, k: int = 5, model: str = "text-embedding-3-small") -> List[dict]:
    """Search the vector store and return JSON-serializable hits.

    Returns list of dicts with keys: quote, source_file, chunk_id, start_offset, end_offset, score
    Score is a distance where lower is better (0 = identical under cosine similarity approximation).
    """
    q_emb = embed_texts([query], model=model)
    hits = backend.search(q_emb[0], k)
    results: List[dict] = []
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
