"""config.py — single source of truth for service endpoints and corpus paths.

Why this exists
---------------
D1-D3 each grew their own defaults read straight from ``os.getenv(...)``.  That
produced one real bug: ``graph_selector.py`` defaulted ``MONGO_DB`` to
``"pdf_agent"`` while ``ingest.py`` / ``retriever.py`` / ``seed.py`` all used
``"csai415"`` — so D3 graph expansion silently queried an empty database.

For Deliverable 4 we centralise every endpoint here, load an optional ``.env``
file once, and expose consistent constants.  New code imports from here; legacy
modules keep their ``os.getenv`` calls but now read the same values because this
module populates ``os.environ`` with the unified defaults at import time.

Load order (highest priority first):
  1. a value already present in the real environment,
  2. a value in the project ``.env`` file,
  3. the hard default below.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent          # .../Deliverable 4/app
PROJECT_ROOT = APP_DIR.parent                      # .../Deliverable 4
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
QA_DIR = DATA_DIR / "qa"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"         # tuned adapters, caches, eval json


# ---------------------------------------------------------------------------
# Minimal .env loader (no python-dotenv dependency required)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file without overwriting reals."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Unified defaults — written back into os.environ so legacy getenv() agrees
# ---------------------------------------------------------------------------

_DEFAULTS = {
    # MongoDB
    "MONGO_URI": "mongodb://localhost:27017",
    "MONGO_DB": "csai415",                 # <- the one true database name
    "MONGO_COLLECTION": "chunks",
    # Qdrant
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "QDRANT_COLLECTION": "chunks",
    # Neo4j
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "csai415pass",
    "NEO4J_DATABASE": "neo4j",
    # Models
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # SLM (Deliverable 4)
    "SLM_BASE_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
    "SLM_ADAPTER_DIR": str(ARTIFACTS_DIR / "slm_lora"),
    "SLM_BACKEND": "extractive",           # extractive | base | tuned  (safe CPU default)
    "SLM_MAX_NEW_TOKENS": "200",           # lower (e.g. 128) for faster CPU generation
    # API
    "API_BASE_URL": "http://localhost:8000",
}

for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)


# Convenience accessors -----------------------------------------------------

def get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


# Frequently used, typed -----------------------------------------------------

MONGO_URI = os.environ["MONGO_URI"]
MONGO_DB = os.environ["MONGO_DB"]
MONGO_COLLECTION = os.environ["MONGO_COLLECTION"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
SLM_BASE_MODEL = os.environ["SLM_BASE_MODEL"]
SLM_ADAPTER_DIR = os.environ["SLM_ADAPTER_DIR"]
SLM_BACKEND = os.environ["SLM_BACKEND"]
