"""Central configuration for the visual self-verification experiment."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REFERENCES_DIR = DATA_DIR / "references"

# ─── API Backend Selection ───────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─── Models ──────────────────────────────────────────────────────────────────
# "provider" determines which client to use: "openai", "anthropic", or "gemini"
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
        "model_version": "2025-04-14",
        "api_version": "2025-04-01-preview",
        "supports_temperature": True,
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
        "model_version": "2025-04-14",
        "api_version": "2025-04-01-preview",
        "supports_temperature": True,
    },
    "gpt-4.1-nano": {
        "provider": "openai",
        "model_id": "gpt-4.1-nano",
        "model_version": "2025-04-14",
        "api_version": "2025-04-01-preview",
        "supports_temperature": True,
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "model_version": "2025-05-14",
        "api_version": "vertex-2023-10-16",
        "supports_temperature": True,
    },
    # gemini-2.5-flash (used for the originally submitted results) was retired by
    # Google and returns 404 for new API keys; gemini-3.6-flash is the successor
    # Google's own error message directs callers to.
    "gemini-3.6-flash": {
        "provider": "gemini",
        "model_id": "gemini-3.6-flash",
        "model_version": "001",
        "api_version": "v1beta",
        "supports_temperature": True,
    },
}

TEXT_REF_MODEL = {
    "provider": "openai",
    "model_id": "gpt-4.1-nano",
    "model_version": "2025-04-14",
    "api_version": "2025-04-01-preview",
    "supports_temperature": True,
}

CONDITIONS = ["A", "B", "C", "D"]

MAX_RETRIES = 6
REQUEST_TIMEOUT = 120

COST_PER_1K_INPUT_TOKENS = {
    "gpt-4.1": 0.002,
    "gpt-4.1-mini": 0.0004,
    "gpt-4.1-nano": 0.0001,
    "claude-sonnet-4": 0.003,
    "gemini-3.6-flash": 0.00075,
}
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4.1": 0.008,
    "gpt-4.1-mini": 0.0016,
    "gpt-4.1-nano": 0.0004,
    "claude-sonnet-4": 0.015,
    "gemini-3.6-flash": 0.00375,
}
