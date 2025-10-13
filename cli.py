"""
Usage:
  python cli.py --index /path/to/docx_dir
  python cli.py --search "your query" --k 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag_indexer import FaissBackend, index_documents, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, help="Directory of .docx files to index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors to return")
    parser.add_argument("--persist", type=str, default="indexes", help="Persist directory for FAISS index")
    parser.add_argument("--chunk-size", type=int, default=300, help="Maximum characters per chunk (default: 300)")
    parser.add_argument("--overlap", type=int, default=50, help="Number of characters to overlap between chunks (default: 50)")

    args = parser.parse_args()

    backend = FaissBackend(persist_dir=args.persist)

    if args.index:
        dirp = Path(args.index)
        logger.info("Indexing directory %s with chunk_size=%d, overlap=%d", 
                   dirp, args.chunk_size, args.overlap)
        index_documents(str(dirp), backend, chunk_size=args.chunk_size, overlap=args.overlap)

    if args.search:
        results = search(args.search, backend, k=args.k)
        import json

        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
