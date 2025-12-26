"""
Configuration for LLM models used throughout the application.

CHANGE PROVIDER HERE TO SWITCH BETWEEN OPENAI AND OLLAMA
"""

# ═══════════════════════════════════════════════════════════
# CHANGE THIS ONE LINE TO SWITCH PROVIDERS
# ═══════════════════════════════════════════════════════════
LLM_PROVIDER = "ollama"  # Options: "openai" or "ollama"


# ═══════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════
MODELS = {
    "ollama": {
        "cheap": "llama3.2",      # Research, summarization, deduplication
        "expensive": "llama3.2",  # Decision making, agent proposals
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
