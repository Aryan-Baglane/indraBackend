"""
INDRA Backend Configuration
Centralized environment variables and configuration management
Updated to fix Qdrant Cloud connectivity and LLM stability.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file (if exists)
load_dotenv()

# ==================== API KEYS ====================

# OpenRouter LLM API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"

# Qdrant Vector Database
# FIX: Removed :6333 for Cloud URL to use standard HTTPS (Port 443)
QDRANT_URL = "https://de0cf931-a860-4f62-a3b5-258c6dfa7317.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7nmUZs4Yhl-_DsIPUpICM11KZp-CkxpfvG_xTO3mm5E")

# Firebase
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", 
    os.path.join(os.path.dirname(__file__), "firebase-service-account.json")
)

# ==================== MODEL CONFIGURATION ====================

# LLM Model (Using a more stable free model for reliability)
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2500"))

# RAG Configuration
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "gis_rwh_rag_indra")
RAG_RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "2"))

# Embedding Model (local)
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
)

# ==================== GIS DATA ====================

GIS_DATA_PATH = os.getenv(
    "GIS_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "INDRA_Processed_Data.csv")
)

# ==================== SERVER CONFIGURATION ====================

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:5173,http://localhost:3000"
).split(",")

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
# Set to True if you want to see the detailed logs on startup
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# ==================== RETRY CONFIGURATION ====================

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))

# ==================== VALIDATION ====================

def validate_config():
    """Validate that required configuration is present"""
    errors = []
    
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is missing")
    if not QDRANT_URL:
        errors.append("QDRANT_URL is missing")
    if ":" in QDRANT_URL.replace("https://", ""):
        # Warning for port usage in cloud URLs
        print("Note: QDRANT_URL contains a port. If using Qdrant Cloud, ensure this is correct (usually not needed).")
    
    if not os.path.exists(GIS_DATA_PATH):
        errors.append(f"GIS data file not found at {GIS_DATA_PATH}")
    
    if errors:
        print("⚠️ Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def get_config_summary():
    """Summary without secrets"""
    return {
        "llm_model": LLM_MODEL,
        "qdrant_url": QDRANT_URL,
        "gis_data_path": GIS_DATA_PATH,
        "debug_mode": DEBUG_MODE
    }

if validate_config() and DEBUG_MODE:
    print("✅ INDRA Configuration Validated & Loaded")