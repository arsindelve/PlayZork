"""Provider selection, including the vLLM backend (prepared for GPU serving).

The switch must be a config change, not a port: `PLAYZORK_LLM_PROVIDER=vllm`
plus a base URL. vLLM speaks the OpenAI API, so the existing client works
unchanged — what differs is that it does real continuous batching, where this
Mac's Ollama was measured at flat throughput regardless of concurrency.
"""
import importlib
import os

import pytest

import config as config_module


@pytest.fixture
def reload_config(monkeypatch):
    def _reload(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(config_module)
    yield _reload
    # Restore a valid provider BEFORE reloading: monkeypatch undoes env vars
    # after this finalizer, so an invalid value would still be set here and the
    # reload would raise during teardown.
    os.environ["PLAYZORK_LLM_PROVIDER"] = "ollama"
    importlib.reload(config_module)


def test_vllm_is_an_accepted_provider(reload_config):
    cfg = reload_config(PLAYZORK_LLM_PROVIDER="vllm")

    assert cfg.LLM_PROVIDER == "vllm"


def test_vllm_client_points_at_the_configured_server(reload_config):
    cfg = reload_config(PLAYZORK_LLM_PROVIDER="vllm",
                        PLAYZORK_VLLM_BASE_URL="http://gpu-box:8000/v1",
                        PLAYZORK_VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct")

    llm = cfg.get_expensive_llm(temperature=0)

    assert llm.openai_api_base == "http://gpu-box:8000/v1"
    assert llm.model_name == "Qwen/Qwen2.5-14B-Instruct"


def test_vllm_needs_no_real_api_key(reload_config):
    """A local server ignores it, but the OpenAI client requires one to exist."""
    cfg = reload_config(PLAYZORK_LLM_PROVIDER="vllm")

    assert cfg.get_cheap_llm(temperature=0) is not None


def test_both_tiers_share_one_served_model(reload_config):
    """vLLM serves one model per process; matching Ollama's single-warm-model
    arrangement avoids swap cost."""
    cfg = reload_config(PLAYZORK_LLM_PROVIDER="vllm")

    assert cfg.get_cheap_llm(0).model_name == cfg.get_expensive_llm(0).model_name


def test_an_unknown_provider_is_rejected_loudly(reload_config):
    with pytest.raises(ValueError, match="PLAYZORK_LLM_PROVIDER"):
        reload_config(PLAYZORK_LLM_PROVIDER="llamafile")


def test_ollama_remains_the_default(reload_config):
    cfg = reload_config(PLAYZORK_LLM_PROVIDER="ollama")

    assert cfg.LLM_PROVIDER == "ollama"
    assert cfg.get_expensive_llm(0) is not None
