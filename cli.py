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

    args = parser.parse_args()

    backend = FaissBackend(persist_dir=args.persist)

    if args.index:
        dirp = Path(args.index)
        logger.info("Indexing directory %s", dirp)
        index_documents(str(dirp), backend)

    if args.search:
        results = search(args.search, backend, k=args.k)
        import json

        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
