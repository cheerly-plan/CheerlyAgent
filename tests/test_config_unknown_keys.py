"""未知 .env 键告警：只读键名、不读值、不拒绝启动。"""

import tempfile
from pathlib import Path

from ai_agent.core import config


def _write_env(root: Path, text: str) -> None:
    (root / ".env").write_text(text, encoding="utf-8")


def test_known_and_allowlisted_keys_are_silent(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_env(
            root,
            "BASE_URL=http://x/v1\n"
            "API_KEY=some-secret\n"
            "POSTGRES_PASSWORD=another-secret\n"
            "# MEMORY_TOP_K=5\n",
        )
        monkeypatch.setattr(config, "_PROJECT_ROOT", root)
        assert config.unknown_dotenv_keys() == []


def test_unknown_keys_are_reported_without_values(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_env(
            root,
            "MEMOY_TOP_K=sentinel-value-12345\n"
            "KB_OCR_MAX_FAILURS=2\n"
            "AUTH_TOKEN=known-key\n",
        )
        monkeypatch.setattr(config, "_PROJECT_ROOT", root)
        got = config.unknown_dotenv_keys()
        assert got == ["KB_OCR_MAX_FAILURS", "MEMOY_TOP_K"]
        assert "sentinel-value-12345" not in "".join(got)


def test_missing_env_file_is_safe(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(config, "_PROJECT_ROOT", Path(d))
        assert config.unknown_dotenv_keys() == []
