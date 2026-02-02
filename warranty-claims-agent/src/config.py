"""
Configuration module for the Warranty Claims Agent.

Loads environment variables and provides centralized configuration management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# LLM Configuration
# Supports both OpenAI and Ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "ollama"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama Configuration (alternative)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))
CHROMA_COLLECTION_NAME = "warranty_policies"

# Data Directories
INBOX_DIR = Path(os.getenv("INBOX_DIR", str(DATA_DIR / "inbox")))
OUTBOX_DIR = Path(os.getenv("OUTBOX_DIR", str(DATA_DIR / "outbox")))
POLICIES_DIR = Path(os.getenv("POLICIES_DIR", str(DATA_DIR / "policies")))
LABELS_DIR = Path(os.getenv("LABELS_DIR", str(DATA_DIR / "labels")))
TEST_CLAIMS_DIR = DATA_DIR / "test_claims"

# SQLite Database for LangGraph Checkpoints
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "claims.db")))

# Phoenix Tracing
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "warranty-claims")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Warranty Configuration
WARRANTY_WINDOW_DAYS = 90  # Default warranty period in days
EXTENDED_WARRANTY_DAYS = 180  # For SalonMaster products

# Product Catalog - Maps product names to IDs
PRODUCT_CATALOG = {
    "AirFlow Basic": "HD-001",
    "AirFlow Pro": "HD-002",
    "TravelDry Mini": "HD-003",
    "TravelDry Plus": "HD-004",
    "SalonMaster 3000": "HD-005",
    "SalonMaster Elite": "HD-006",
    "QuietBlow Compact": "HD-007",
    "QuietBlow Deluxe": "HD-008",
    "KidSafe Dryer": "HD-009",
    "ProStyle Ionic": "HD-010",
}

# Reverse mapping: ID to product name
PRODUCT_ID_TO_NAME = {v: k for k, v in PRODUCT_CATALOG.items()}

# Products with extended warranty (180 days)
EXTENDED_WARRANTY_PRODUCTS = ["HD-005", "HD-006"]


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        INBOX_DIR,
        OUTBOX_DIR,
        POLICIES_DIR,
        LABELS_DIR,
        CHROMA_PERSIST_DIR,
        TEST_CLAIMS_DIR,
        TEST_CLAIMS_DIR / "spam",
        TEST_CLAIMS_DIR / "valid",
        TEST_CLAIMS_DIR / "invalid",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
ensure_directories()
