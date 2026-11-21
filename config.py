"""Centralized configuration and constants for RESUMATE AI."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "resumate_history.db"
CSS_PATH = ASSETS_DIR / "style.css"

# Load local .env without overriding already-injected runtime environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# Application metadata
APP_NAME = "RESUMATE AI"
APP_TAGLINE = "AI-Powered Resume Analyzer & ATS Compatibility Studio"
APP_VERSION = "1.0.0"

# File upload & validation constraints
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PDF_SIZE_MB = 10
ALLOWED_EXTENSIONS = (".pdf",)
MIN_RESUME_CHARS = 50
MIN_JD_WORDS = 10
MIN_JD_CHARS = 40
MAX_JD_CHARS = 30000
MAX_RESUME_CHARS_FOR_AI = 35000

# Gemini AI configuration
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"

# Allowed/supported Gemini models (for safe override handling)
SUPPORTED_GEMINI_MODELS = {
    "gemini-4-hydrogen",
    "gemini-4-neon",
    "gemini-4-carbon",
    "gemini-4-argon",
    "gemini-3.1-flash-lite",
}

# Match level thresholds
MATCH_LEVELS = (
    (85, 100, "Excellent Match", "#10B981"),
    (70, 84, "Good Match", "#3B82F6"),
    (50, 69, "Moderate Match", "#F59E0B"),
    (0, 49, "Needs Improvement", "#EF4444"),
)

# Disclaimer text required across UI and reports
ATS_DISCLAIMER = (
    "Estimated Compatibility Score: This ATS score is an AI-estimated compatibility "
    "benchmark comparing your resume against the provided job description. It is not "
    "affiliated with or guaranteed to match any specific employer's proprietary ATS algorithm."
)


def get_gemini_api_key() -> str:
    """Safely retrieve the Gemini API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in {"YOUR_GEMINI_API_KEY", "MY_GEMINI_API_KEY"}:
        return ""
    return api_key


def get_gemini_model() -> str:
    """Return the configured Gemini model, defaulting to gemini-4-hydrogen."""
    configured = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    return configured if configured else DEFAULT_GEMINI_MODEL
