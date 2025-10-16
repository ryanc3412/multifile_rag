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

    if model is None:
        model = EMBEDDING_MODEL

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embs: list[np.ndarray] = []
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

def chunk_text(text: str) -> list[dict]:
    """
    Split text into overlapping chunks trying to preserve sentence boundaries and word boundaries.

    Returns list of dicts: {'text':..., 'start': int, 'end': int, 'chunk_id': int}
    """
    if CHUNK_SIZE <= CHUNK_OVERLAP:
        raise ValueError("CHUNK_SIZE must be larger than CHUNK_OVERLAP")

    def find_word_boundary(pos: int, forward: bool = True) -> int:
        """Find the nearest word boundary (space) in either direction."""
        if forward:
            while pos < length and text[pos] != " ":
                pos += 1
            return pos
        else:
            # Walk backwards until we hit a space or start of text
            while pos > 0 and text[pos - 1] != " ":
                pos -= 1
            return pos

    # Prefer multi-char separators that include trailing space so next chunk starts cleanly.
    # Order matters: higher-priority separators come first.
    multi_sep = ["\n", ". ", "? ", "! "]
    single_sep = [".", "?", "!"]
    length = len(text)
    start = 0
    chunks: list[dict] = []
    chunk_id = 0

    min_split_point = CHUNK_SIZE - CHUNK_OVERLAP  # require this many chars before a natural split

    # Ensure we start at a clean word boundary
    start = find_word_boundary(start, forward=False)

    while start < length:
        end = min(start + CHUNK_SIZE, length)
        window = text[start:end]

        best_abs_split = None

        # 1) Try multi-char separators (exact match) searching for the last occurrence in window
        for sep in multi_sep:
            pos = window.rfind(sep)
            if pos != -1 and (pos + len(sep)) >= min_split_point:
                # split after the separator so next chunk begins cleanly
                candidate = start + pos + len(sep)
                # choose the candidate nearest the window end (largest pos)
                if best_abs_split is None or candidate > best_abs_split:
                    best_abs_split = candidate

        # 2) If no multi-char sep found, try single-char punctuation but enforce min_split_point
        if best_abs_split is None:
            for sep in single_sep:
                pos = window.rfind(sep)
                if pos != -1 and (pos + 1) >= min_split_point:
                    candidate = start + pos + 1
                    if best_abs_split is None or candidate > best_abs_split:
                        best_abs_split = candidate

        # 3) Fallback to last word boundary if it's after the min_split_point
        if best_abs_split is None and end < length:
            pos = window.rfind(" ")
            if pos != -1 and pos >= min_split_point:
                best_abs_split = start + pos + 1  # split after the space for clean word boundary
            else:
                # If no natural boundary found, force a word boundary near the end
                best_abs_split = find_word_boundary(end, forward=False)

        # 4) If still none, use end (hard cut). But ensure forward progress
        if best_abs_split is None:
            split_pos = end
        else:
            split_pos = min(best_abs_split, end)

        # Safety: ensure split_pos > start (force minimal progress if necessary)
        if split_pos <= start:
            # advance by at least 1 character (or half chunk size) to avoid infinite loops on pathological input
            forced = min(length, start + max(1, CHUNK_SIZE // 2))
            split_pos = forced

        # Clean up chunk text and ensure it starts/ends at word boundaries
        chunk_text_str = text[start:split_pos].strip()
        chunks.append({
            "text": chunk_text_str,
            "start": start,
            "end": split_pos,
            "chunk_id": chunk_id
        })
        chunk_id += 1

        # compute next start using overlap, but ensure it advances and starts at a word boundary
        next_start = split_pos - CHUNK_OVERLAP
        next_start = find_word_boundary(next_start, forward=False)  # move to previous word boundary
        
        if next_start <= start:
            start = split_pos  # no overlap if that'll cause no progress
        else:
            start = next_start

    return chunks


def index_documents(dir_path: str, backend: VectorStoreBackend, persist_path: str | None = None):
    """Index all .docx files in dir_path using provided backend and persist results.

    Args:
        dir_path: directory containing .docx files
        backend: VectorStoreBackend implementation (FaissBackend for now)
        persist_path: optional path to persist index/metadata; used by backend
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"dir_path {dir_path} does not exist or is not a directory")

    texts: list[str] = []
    metadatas: list[dict] = []

    # TODO: Add support for other file types (PDF, TXT)
    # Make helper functions to load different file types
    # Look into langchain_community document loaders 

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

        chunks = chunk_text(full_text)
        for c in chunks:
            texts.append(c["text"])
            metadatas.append({
                "source_file": docx_file.name,
                "chunk_id": int(c["chunk_id"]),
                "start_offset": int(c["start"]),
                "end_offset": int(c["end"]),
                "text": c["text"],
                # record which embedding model was used to create these vectors
                "embedding_model": EMBEDDING_MODEL,
            })

    if not texts:
        logger.warning("No documents found to index in %s", dir_path)
        return

    embeddings = embed_texts(texts)
    backend.add_vectors(embeddings, metadatas)
    backend.save()


def search(query: str, backend: FaissBackend, k: int = 5) -> list[dict]:
    """Search the vector store and return JSON-serializable hits.

    Returns list of dicts with keys: quote, source_file, chunk_id, start_offset, end_offset, score
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
