import pytest

from aimaestro.config import DEFAULT_MODEL, Configuration, api_key_error


def test_defaults_when_no_env(monkeypatch):
    monkeypatch.delenv("AIMAESTRO_MODEL", raising=False)
    monkeypatch.delenv("AIMAESTRO_USER_ID", raising=False)
    cfg = Configuration.from_runnable_config(None)
    assert cfg.model == DEFAULT_MODEL
    assert cfg.user_id == "default-user"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AIMAESTRO_MODEL", "openai:gpt-4o")
    cfg = Configuration.from_runnable_config(None)
    assert cfg.model == "openai:gpt-4o"


def test_env_var_beats_runnable_config(monkeypatch):
    monkeypatch.setenv("AIMAESTRO_USER_ID", "from-env")
    cfg = Configuration.from_runnable_config({"configurable": {"user_id": "from-config"}})
    assert cfg.user_id == "from-env"


def test_runnable_config_used_when_env_absent(monkeypatch):
    monkeypatch.delenv("AIMAESTRO_USER_ID", raising=False)
    cfg = Configuration.from_runnable_config({"configurable": {"user_id": "kratika"}})
    assert cfg.user_id == "kratika"


def test_api_key_error_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    err = api_key_error("google_genai:gemini-2.5-flash")
    assert err is not None and "GOOGLE_API_KEY" in err


def test_api_key_error_detects_placeholder(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your-google-api-key-here")
    err = api_key_error("google_genai:gemini-2.5-flash")
    assert err is not None and "placeholder" in err.lower()


def test_api_key_error_none_when_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyRealLookingKey123")
    assert api_key_error("google_genai:gemini-2.5-flash") is None


def test_unknown_provider_needs_no_key():
    assert api_key_error("ollama:llama3") is None
