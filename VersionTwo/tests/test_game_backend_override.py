"""Env-overridable game backend URL, for self-hosted / offline play.

Points the active game at a local backend (e.g. the ZorkAI Docker stack on
localhost:5100-5103) without editing the committed cloud defaults.
"""
import config


def test_base_url_and_endpoint_are_overridable(monkeypatch):
    monkeypatch.setenv("PLAYZORK_GAME_BASE_URL", "http://localhost:5102")
    monkeypatch.setenv("PLAYZORK_GAME_ENDPOINT", "/EscapeRoom")
    cfg = config.get_game_config()
    assert cfg["base_url"] == "http://localhost:5102"
    assert cfg["endpoint"] == "/EscapeRoom"
    # Non-URL fields still come from the backend definition.
    assert cfg["name"] == config.GAME_BACKENDS[config.ACTIVE_GAME]["name"]


def test_unset_falls_back_to_committed_defaults(monkeypatch):
    monkeypatch.delenv("PLAYZORK_GAME_BASE_URL", raising=False)
    monkeypatch.delenv("PLAYZORK_GAME_ENDPOINT", raising=False)
    cfg = config.get_game_config()
    default = config.GAME_BACKENDS[config.ACTIVE_GAME]
    assert cfg["base_url"] == default["base_url"]
    assert cfg["endpoint"] == default["endpoint"]


def test_override_does_not_mutate_the_committed_defaults(monkeypatch):
    before = dict(config.GAME_BACKENDS[config.ACTIVE_GAME])
    monkeypatch.setenv("PLAYZORK_GAME_BASE_URL", "http://localhost:9999")
    config.get_game_config()
    assert config.GAME_BACKENDS[config.ACTIVE_GAME] == before
