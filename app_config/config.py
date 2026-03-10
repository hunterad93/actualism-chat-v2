from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
MODEL_CONFIG: dict[str, dict[str, str]] = {
    "google/gemini-3-flash-preview": {"provider": "openrouter"},
    "moonshotai/kimi-k2.5": {"provider": "openrouter"},
    "deepseek/deepseek-v3.2": {"provider": "openrouter"},
    "gpt-5-mini": {"provider": "openai"},
}
CHAINLIT_MODEL_PROVIDER = os.getenv("CHAINLIT_MODEL_PROVIDER", "openai").strip().lower()
CHAINLIT_AUTH_USERNAME = os.getenv("CHAINLIT_AUTH_USERNAME")
CHAINLIT_AUTH_PASSWORD = os.getenv("CHAINLIT_AUTH_PASSWORD")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "60"))
MAX_TOOL_ROUNDS = int(os.getenv("OPENAI_MAX_TOOL_ROUNDS", "8"))
SITE_BASE_URL = "https://www.actualfreedom.com.au"
MAX_CHAT_TURNS = int(os.getenv("MAX_CHAT_TURNS", "5"))
MAX_TOOL_MATCHES_IN_CONTEXT = int(os.getenv("MAX_TOOL_MATCHES_IN_CONTEXT", "8"))
MAX_TOOL_LINES_PER_MATCH = int(os.getenv("MAX_TOOL_LINES_PER_MATCH", "8"))
MAX_TOOL_LINE_CHARS = int(os.getenv("MAX_TOOL_LINE_CHARS", "240"))
