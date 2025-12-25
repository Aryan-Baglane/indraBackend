"""
INDRA Backend Configuration
Centralized environment variables and configuration management
All sensitive keys are loaded from environment variables for production security
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file (if exists)
load_dotenv()


# ==================== API KEYS ====================

# OpenRouter LLM API (used for all AI features)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Qdrant Vector Database (for RAG)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Firebase (for auth and database)
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", 
    os.path.join(os.path.dirname(__file__), "firebase-service-account.json")
)


# ==================== MODEL CONFIGURATION ====================

# LLM Model (OpenRouter)
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2500"))

# RAG Configuration
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "standrd_rag")
RAG_RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "2"))

# Embedding Model (local)
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
)


# ==================== GIS DATA ====================

# GIS CSV Data Path (contains rainfall, groundwater data for India)
GIS_DATA_PATH = os.getenv(
    "GIS_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "INDRA_Processed_Data.csv")
)


# ==================== SERVER CONFIGURATION ====================

# CORS Origins (comma-separated in env var)
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:5173,http://localhost:3000"
).split(",")

# Server Settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


# ==================== RETRY CONFIGURATION ====================

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))


# ==================== VALIDATION ====================

def validate_config():
    """Validate that required configuration is present"""
    errors = []
    
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is required")
    
    if not QDRANT_URL:
        errors.append("QDRANT_URL is required")
    
    if not QDRANT_API_KEY:
        errors.append("QDRANT_API_KEY is required")
    
    if not os.path.exists(GIS_DATA_PATH):
        errors.append(f"GIS data file not found at {GIS_DATA_PATH}")
    
    if errors:
        print("⚠️ Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def get_config_summary():
    """Get a summary of current configuration (without exposing secrets)"""
    return {
        "llm_model": LLM_MODEL,
        "qdrant_url": QDRANT_URL[:30] + "..." if QDRANT_URL else "NOT SET",
        "qdrant_api_key": "SET" if QDRANT_API_KEY else "NOT SET",
        "openrouter_api_key": "SET" if OPENROUTER_API_KEY else "NOT SET",
        "gis_data_path": GIS_DATA_PATH,
        "embedding_model_path": EMBEDDING_MODEL_PATH,
        "cors_origins": CORS_ORIGINS,
        "debug_mode": DEBUG_MODE
    }


# Print config status on import (only in debug mode)
if DEBUG_MODE:
    print("INDRA Configuration Loaded:")
    for key, value in get_config_summary().items():
        print(f"  {key}: {value}")
