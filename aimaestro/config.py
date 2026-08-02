"""Environment-driven configuration for aiMaestro."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
DEFAULT_DB_PATH = "data/aimaestro.db"
ENV_PREFIX = "AIMAESTRO_"

#: Which API key each provider needs.
PROVIDER_KEYS = {
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

#: Substrings that mark a value as a stand-in rather than a real key.
_PLACEHOLDER_MARKERS = ("your-", "sk-xxx", "changeme", "<", "xxx", "placeholder")


@dataclass(kw_only=True)
class Configuration:
    """Settings resolved from the environment first, then the runnable config."""

    user_id: str = "default-user"
    model: str = DEFAULT_MODEL
    db_path: str = DEFAULT_DB_PATH

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {}
        for f in fields(cls):
            if not f.init:
                continue
            values[f.name] = os.environ.get(
                f"{ENV_PREFIX}{f.name.upper()}"
            ) or configurable.get(f.name)
        return cls(**{k: v for k, v in values.items() if v})


def provider_of(model: str) -> str | None:
    """The provider half of a ``provider:model`` identifier."""
    return model.split(":", 1)[0] if ":" in model else None


def api_key_error(model: str) -> str | None:
    """Describe what is wrong with the API key, or return None if it looks usable.

    Checked before the first model call so a missing key reads as an instruction
    rather than a stack trace.
    """
    var = PROVIDER_KEYS.get(provider_of(model) or "")
    if var is None:
        return None

    value = os.environ.get(var, "").strip()
    if not value:
        return (
            f"{var} is not set, and {model} needs it.\n"
            f"  Add it to your .env file.\n"
            f"  A free Gemini key: https://aistudio.google.com/apikey"
        )
    if any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
        return (
            f"{var} is still a placeholder ({value!r}).\n"
            f"  Replace it with a real key in your .env file.\n"
            f"  A free Gemini key: https://aistudio.google.com/apikey"
        )
    return None
