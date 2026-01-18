"""
Configuration for LLM models used throughout the application.

CHANGE PROVIDER HERE TO SWITCH BETWEEN OPENAI AND OLLAMA
"""

# ═══════════════════════════════════════════════════════════
# GAME CONFIGURATION
# ═══════════════════════════════════════════════════════════
GAME_NAME = "Planetfall"
GAME_OBJECTIVE = "Reach a score of 80 points"
GAME_OBJECTIVE_SCORE = 80  # Numeric value for scoring logic

# Session ID for tracking game sessions
SESSION_ID = "P-V5"

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
    }
}

# Active game backend - change this to switch games
ACTIVE_GAME = "planetfall"  # Options: "zork" or "planetfall"

# Helper function to get the current game config
def get_game_config():
    """Get the configuration for the currently active game."""
    if ACTIVE_GAME not in GAME_BACKENDS:
        raise ValueError(f"Invalid ACTIVE_GAME: {ACTIVE_GAME}. Must be one of {list(GAME_BACKENDS.keys())}")
    return GAME_BACKENDS[ACTIVE_GAME]

# ═══════════════════════════════════════════════════════════
# CHANGE THIS ONE LINE TO SWITCH PROVIDERS
# ═══════════════════════════════════════════════════════════
LLM_PROVIDER = "openai"  # Options: "openai" or "ollama"

# ═══════════════════════════════════════════════════════════
# TIMEOUT AND RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════
LLM_TIMEOUT_SECONDS = 300  # Timeout for each LLM call
LLM_MAX_RETRIES = 5       # Maximum retry attempts


# ═══════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════
MODELS = {
    "ollama": {
        "cheap": "llama3.2",      # Research, summarization, deduplication
        "expensive": "llama3.3",  # Decision-making, agent proposals
    },
    "openai": {
        "cheap": "gpt-5-nano-2025-08-07",
        "expensive": "gpt-5-mini-2025-08-07",
    }
}


# ═══════════════════════════════════════════════════════════
# FACTORY FUNCTIONS - Use these everywhere to get LLMs
# ═══════════════════════════════════════════════════════════
def get_cheap_llm(temperature: float = 0):
    """
    Get the cheap LLM instance.

    Used for: research, summarization, deduplication
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=MODELS["ollama"]["cheap"], temperature=temperature)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODELS["openai"]["cheap"], temperature=temperature)
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")


def get_expensive_llm(temperature: float = 0):
    """
    Get the expensive LLM instance.

    Used for: decision making, agent proposals, observation, issue detection
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=MODELS["ollama"]["expensive"], temperature=temperature)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODELS["openai"]["expensive"], temperature=temperature)
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")
