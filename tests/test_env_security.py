"""Credentials come only from the environment / .env, and are never echoed back."""
import os
import re
from pathlib import Path

from capstone import env

ROOT = Path(__file__).resolve().parent.parent


def test_dotenv_overrides_and_skips_blank(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nTWELVEDATA_API_KEY=abc123xyz\nTEJ_API_KEY=\n", encoding="utf-8")
    monkeypatch.setenv("TWELVEDATA_API_KEY", "old-value")
    monkeypatch.delenv("TEJ_API_KEY", raising=False)
    env.load(f)
    assert os.environ["TWELVEDATA_API_KEY"] == "abc123xyz"
    assert "TEJ_API_KEY" not in os.environ  # blank values are ignored


def test_redact_hides_secret_values(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_API_KEY", "supersecretkey123")
    msg = "HTTP 401 for key supersecretkey123"
    assert "supersecretkey123" not in env.redact(msg)
    assert env.redact("nothing to hide") == "nothing to hide"


def test_no_hardcoded_credentials_in_source():
    """Scan every tracked text source for things that look like a real key assignment."""
    pattern = re.compile(r"(API_KEY|SUPABASE_KEY|apikey)[ \t]*[=:][ \t]*['\"]?(?!paste|your)[A-Za-z0-9_\-]{20,}", re.I)
    hits = []
    for path in ROOT.rglob("*"):
        if any(part in {".git", ".venv", "data", "models", "__pycache__"} for part in path.parts):
            continue
        if path.name == ".env" or path.suffix not in {".py", ".toml", ".md", ".yml", ".example", ".txt", ".sql"}:
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_gitignore_excludes_secrets():
    text = (ROOT / ".gitignore").read_text()
    assert ".env" in text.splitlines() and ".streamlit/secrets.toml" in text
