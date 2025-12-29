"""
Configuration for LLM models used throughout the application.

CHANGE PROVIDER HERE TO SWITCH BETWEEN OPENAI AND OLLAMA
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

# ═══════════════════════════════════════════════════════════
# GAME CONFIGURATION
# ═══════════════════════════════════════════════════════════
GAME_NAME = "Zork I"
GAME_OBJECTIVE = "Reach a score of 350 points"
GAME_OBJECTIVE_SCORE = 350  # Numeric value for scoring logic

# ═══════════════════════════════════════════════════════════
# CHANGE THIS ONE LINE TO SWITCH PROVIDERS
# ═══════════════════════════════════════════════════════════
LLM_PROVIDER = "openai"  # Options: "openai" or "ollama"

# ═══════════════════════════════════════════════════════════
# TIMEOUT AND RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════
LLM_TIMEOUT_SECONDS = 90  # Timeout for each LLM call
LLM_MAX_RETRIES = 3       # Maximum retry attempts


# ═══════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════
MODELS = {
    "ollama": {
        "cheap": "llama3.3",      # Research, summarization, deduplication
        "expensive": "llama3.3",  # Decision making, agent proposals
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


# ═══════════════════════════════════════════════════════════
# RETRY WRAPPER - Use for all LLM invocations
# ═══════════════════════════════════════════════════════════
def invoke_with_retry(chain, input_data, operation_name: str = "LLM call"):
    """
    Invoke an LLM chain with timeout and retry logic.

    Args:
        chain: The LangChain chain/agent to invoke
        input_data: Input to pass to the chain
        operation_name: Human-readable name for logging

    Returns:
        The response from the chain

    Raises:
        Exception: If all retries fail
    """
    logger = logging.getLogger(__name__)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info(f"[{operation_name}] Attempt {attempt}/{LLM_MAX_RETRIES} (timeout: {LLM_TIMEOUT_SECONDS}s)")

            # Use ThreadPoolExecutor to run with timeout
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(chain.invoke, input_data)
                result = future.result(timeout=LLM_TIMEOUT_SECONDS)
                logger.info(f"[{operation_name}] Success on attempt {attempt}")
                executor.shutdown(wait=False)
                return result
            except FuturesTimeoutError:
                logger.warning(f"[{operation_name}] Timeout after {LLM_TIMEOUT_SECONDS}s on attempt {attempt}")
                # Cancel the future and shutdown executor without waiting
                future.cancel()
                executor.shutdown(wait=False)

                if attempt < LLM_MAX_RETRIES:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    logger.info(f"[{operation_name}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    logger.info(f"[{operation_name}] Retrying now...")
                else:
                    raise TimeoutError(f"{operation_name} timed out after {LLM_MAX_RETRIES} attempts")
            except Exception as e:
                logger.error(f"[{operation_name}] Error on attempt {attempt}: {e}")
                executor.shutdown(wait=False)
                if attempt < LLM_MAX_RETRIES:
                    wait_time = 2 ** attempt
                    logger.info(f"[{operation_name}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    logger.info(f"[{operation_name}] Retrying now...")
                else:
                    raise

        except TimeoutError:
            raise  # Re-raise timeout error

    raise Exception(f"{operation_name} failed after {LLM_MAX_RETRIES} attempts")
