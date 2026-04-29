"""
Configuration for LLM models used throutry weghout the application.

CHANGE PROVIDER HERE TO SWITCH BETWEEN OPENAI AND OLLAMA
"""
from functools import lru_cache

# ═══════════════════════════════════════════════════════════
# GAME CONFIGURATION
# ═══════════════════════════════════════════════════════════
GAME_NAME = "Planetfall"
GAME_OBJECTIVE = "Reach a score of 80 points"
GAME_OBJECTIVE_SCORE = 80  # Numeric value for scoring logic

# Session ID for tracking game sessions
SESSION_ID = "E9"

# ═══════════════════════════════════════════════════════════
# GAME BACKEND CONFIGURATION
# ═══════════════════════════════════════════════════════════
GAME_BACKENDS = {
    "zork": {
        "base_url": "https://bxqzfka0hc.execute-api.us-east-1.amazonaws.com",
        "endpoint": "/Prod/ZorkOne",
        "name": "Zork I",
        "objective": "Reach a score of 350 points",
        "target_score": 350
    },
    "planetfall": {
        "base_url": "https://6kvs9n5pj4.execute-api.us-east-1.amazonaws.com",
        "endpoint": "/Prod/Planetfall",
        "name": "Planetfall",
        "objective": "Complete the mission",
        "target_score": 80
    },
    "escaperoom": {
        "base_url": "http://localhost:5000",
        "endpoint": "/EscapeRoom",
        "name": "Escape Room",
        "objective": "Escape the room",
        "target_score": 100
    }
}

# Active game backend - change this to switch games
ACTIVE_GAME = "escaperoom"  # Options: "zork", "planetfall", or "escaperoom"

# Helper function to get the current game config
def get_game_config():
    """Get the configuration for the currently active game."""
    if ACTIVE_GAME not in GAME_BACKENDS:
        raise ValueError(f"Invalid ACTIVE_GAME: {ACTIVE_GAME}. Must be one of {list(GAME_BACKENDS.keys())}")
    return GAME_BACKENDS[ACTIVE_GAME]

# ═══════════════════════════════════════════════════════════
# CHANGE THIS ONE LINE TO SWITCH PROVIDERS
# ═══════════════════════════════════════════════════════════
LLM_PROVIDER = "ollama"  # Options: "openai" or "ollama"

# ═══════════════════════════════════════════════════════════
# TIMEOUT AND RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════
LLM_TIMEOUT_SECONDS = 300  # Timeout for each LLM call
LLM_MAX_RETRIES = 5       # Maximum retry attempts

# Wall-clock budget for the full per-turn decision graph
# (spawn_agents → research → decide → close → observe → persist).
# A turn that exceeds this raises asyncio.TimeoutError instead of stalling.
TURN_BUDGET_SECONDS = 600


# ═══════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════
MODELS = {
    "ollama": {
        "cheap": "qwen2.5:14b",      # Research, summarization, deduplication (~9GB, ~80 tok/s)
        "expensive": "qwen2.5:14b",  # Decision-making, agent proposals (same model — stays warm, no swap)
    },
    "openai": {
        "cheap": "gpt-5-nano-2025-08-07",
        "expensive": "gpt-5-mini-2025-08-07",
    }
}


# ═══════════════════════════════════════════════════════════
# FACTORY FUNCTIONS - Use these everywhere to get LLMs
# ═══════════════════════════════════════════════════════════
# Clients are memoized per (provider, tier, temperature) so hot paths
# reuse a single ChatOpenAI/ChatOllama instance — preserving HTTP
# keepalive and avoiding per-turn client construction overhead.
@lru_cache(maxsize=None)
def _build_llm(provider: str, tier: str, temperature: float):
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=MODELS["ollama"][tier], temperature=temperature)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODELS["openai"][tier], temperature=temperature)
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {provider}")


def get_cheap_llm(temperature: float = 0):
    """Get the cheap LLM instance (research, summarization, deduplication)."""
    return _build_llm(LLM_PROVIDER, "cheap", temperature)


def get_expensive_llm(temperature: float = 0):
    """Get the expensive LLM instance (decisions, proposals, observation)."""
    return _build_llm(LLM_PROVIDER, "expensive", temperature)
