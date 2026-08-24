"""Runtime configuration for the game backend and LLM provider."""
from functools import lru_cache
import logging
import os
from pathlib import Path

from dotenv import load_dotenv


# Load repository-local configuration before evaluating module constants. This
# also makes direct imports (tests, scripts, and IDE runs) behave like main.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

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

# Active game backend. Override in .env without editing source.
ACTIVE_GAME = os.getenv("PLAYZORK_GAME", "zork").strip().lower()

# Helper function to get the current game config
def get_game_config():
    """Get the configuration for the currently active game."""
    if ACTIVE_GAME not in GAME_BACKENDS:
        raise ValueError(f"Invalid ACTIVE_GAME: {ACTIVE_GAME}. Must be one of {list(GAME_BACKENDS.keys())}")
    return GAME_BACKENDS[ACTIVE_GAME]


# Game-facing prompts and scoring must describe the selected backend.
_ACTIVE_GAME_CONFIG = get_game_config()
GAME_NAME = _ACTIVE_GAME_CONFIG["name"]
GAME_OBJECTIVE = _ACTIVE_GAME_CONFIG["objective"]
GAME_OBJECTIVE_SCORE = _ACTIVE_GAME_CONFIG["target_score"]

# Session ID for both the game backend and local persistence.
SESSION_ID = os.getenv("PLAYZORK_SESSION_ID", "local-zork-session").strip()
if not SESSION_ID:
    raise ValueError("PLAYZORK_SESSION_ID must not be empty")

# ═══════════════════════════════════════════════════════════
# LLM PROVIDER CONFIGURATION
# ═══════════════════════════════════════════════════════════
LLM_PROVIDER = os.getenv("PLAYZORK_LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in {"openai", "ollama", "vllm"}:
    raise ValueError("PLAYZORK_LLM_PROVIDER must be 'openai', 'ollama' or 'vllm'")

# vLLM exposes an OpenAI-compatible server, so it needs a base URL and the
# served model name rather than API credentials. Unlike Ollama it does real
# continuous batching, which is the difference that matters here: this project
# fans out 5-10 concurrent calls per turn, and Ollama was measured serving them
# at FLAT throughput (see STATUS.md 2026-08-24) — i.e. no concurrency at all.
VLLM_BASE_URL = os.getenv("PLAYZORK_VLLM_BASE_URL", "http://localhost:8000/v1").strip()
VLLM_MODEL = os.getenv("PLAYZORK_VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct").strip()

# ═══════════════════════════════════════════════════════════
# EXPERIMENT CONDITION
# ═══════════════════════════════════════════════════════════
# Which architecture plays the game. This is the thesis's independent
# variable, so it is a first-class runtime setting rather than a code edit:
#
#   multi_agent  - advocacy agents + arbiter (the treatment)
#   single_shot  - one inference per turn with everything in context (control)
#
# The control is given the SAME information the treatment assembles, minus the
# deliberation. A weak baseline would make the comparison meaningless.
EXPERIMENT_CONDITION = os.getenv("PLAYZORK_CONDITION", "multi_agent").strip().lower()
if EXPERIMENT_CONDITION not in {"multi_agent", "single_shot"}:
    raise ValueError("PLAYZORK_CONDITION must be 'multi_agent' or 'single_shot'")

# ═══════════════════════════════════════════════════════════
# TIMEOUT AND RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════
# These numbers are coupled, not independent (GitHub issue #3). The turn budget
# wraps the entire decision graph, and each guarded LLM call's retry envelope
# lives *inside* it. If the envelope is larger than the budget, retry attempts
# 2..N can never run: the outer deadline always fires first and the retry policy
# is dead code. The old values (300s x 5 = ~1530s envelope inside a 600s budget)
# had exactly that defect.
#
# The per-attempt timeout is sized from measured latency, not guessed. On the
# 2026-08-21 smoke run, individual qwen2.5:14b calls took 43-113s under agent
# contention, so the 60-90s cap suggested in #3 would abort healthy calls — and
# per #27 each abort piles another request onto the server that was already
# slow. 180s clears the measured worst case with headroom while still catching
# a genuine hang.
LLM_TIMEOUT_SECONDS = int(os.getenv("PLAYZORK_LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_RETRIES = int(os.getenv("PLAYZORK_LLM_MAX_RETRIES", "3"))


def retry_envelope_seconds(timeout_seconds: int = None, max_retries: int = None) -> int:
    """Worst-case wall clock for ONE guarded LLM call.

    Every attempt times out, plus the exponential backoff llm_utils sleeps
    between attempts (2**attempt after attempts 1..n-1). Keep this in sync with
    `llm_utils.invoke_with_retry` / `ainvoke_with_retry`.
    """
    timeout_seconds = LLM_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    attempts = max(1, LLM_MAX_RETRIES if max_retries is None else max_retries)
    backoff = sum(2 ** attempt for attempt in range(1, attempts))
    return attempts * timeout_seconds + backoff


LLM_RETRY_ENVELOPE_SECONDS = retry_envelope_seconds()

# Wall-clock budget for the full per-turn decision graph
# (spawn_agents → research → decide → close → observe → persist).
# A turn that exceeds this raises asyncio.TimeoutError; GameSession catches it
# and recovers with a fallback command rather than ending the run (#1).
#
# Floor: the budget must hold at least one full retry envelope plus room for
# the rest of the turn's work, or retries are unreachable. A configured value
# below the floor is raised to it.
TURN_BUDGET_FLOOR_SECONDS = LLM_RETRY_ENVELOPE_SECONDS * 2
_CONFIGURED_TURN_BUDGET = int(os.getenv("PLAYZORK_TURN_BUDGET_SECONDS", "1200"))
TURN_BUDGET_SECONDS = max(_CONFIGURED_TURN_BUDGET, TURN_BUDGET_FLOOR_SECONDS)

if TURN_BUDGET_SECONDS > _CONFIGURED_TURN_BUDGET:
    logging.getLogger(__name__).warning(
        "PLAYZORK_TURN_BUDGET_SECONDS=%ss is below the retry envelope floor "
        "(%ss for %s attempts x %ss); raising it to %ss so LLM retries are "
        "reachable.",
        _CONFIGURED_TURN_BUDGET,
        TURN_BUDGET_FLOOR_SECONDS,
        LLM_MAX_RETRIES,
        LLM_TIMEOUT_SECONDS,
        TURN_BUDGET_SECONDS,
    )


# How many turns of raw history the BigPictureAnalyzer folds into its prompt.
# It runs on the EXPENSIVE model every turn, so this window is the dominant
# term in per-turn latency growth: turn time on the 2026-08-22 checkpoint run
# climbed 79s -> 228s and plateaued exactly as this window filled. Lower it to
# trade strategic depth for throughput when running experiments.
BIG_PICTURE_HISTORY_TURNS = int(os.getenv("PLAYZORK_BIG_PICTURE_HISTORY_TURNS", "20"))

# Hard caps on the two rolling summaries, in characters.
#
# Both feed EVERY agent prompt each turn, so their growth is multiplied by the
# per-turn call count. Measured over a 26-turn run: the RECENT summary grew
# 140 -> 1334 chars despite covering a fixed 15-turn window, and the
# long-running one 374 -> 1653 and still climbing. They were the last
# unbounded inputs after every other window was capped (#24 option 3).
#
# Enforced by truncation, not only by asking the model nicely — a prompt
# instruction is a hint, and this one has to hold.
RECENT_SUMMARY_MAX_CHARS = int(os.getenv("PLAYZORK_RECENT_SUMMARY_MAX_CHARS", "1200"))
LONG_SUMMARY_MAX_CHARS = int(os.getenv("PLAYZORK_LONG_SUMMARY_MAX_CHARS", "2500"))

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
        "expensive": "gpt-5.6-sol",
    },
    # vLLM serves one model per process, so both tiers point at the same
    # served name — matching the Ollama arrangement, where a single warm model
    # avoids swap cost. Override with PLAYZORK_VLLM_MODEL.
    "vllm": {
        "cheap": None,
        "expensive": None,
    },
}


# ═══════════════════════════════════════════════════════════
# FACTORY FUNCTIONS - Use these everywhere to get LLMs
# ═══════════════════════════════════════════════════════════
# Clients are memoized per (provider, tier, temperature) so hot paths
# reuse a single ChatOpenAI/ChatOllama instance — preserving HTTP
# keepalive and avoiding per-turn client construction overhead.
@lru_cache(maxsize=None)
def _build_llm(provider: str, tier: str, temperature: float):
    # Attached here rather than at call sites so no path can opt out. A
    # structured-output chain returns a parsed model and discards the
    # AIMessage, so metering return values missed every agent proposal and the
    # decision — the architecture's own work.
    from token_meter import get_token_callback
    callbacks = [get_token_callback()]
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {}
        if ollama_host := os.getenv("OLLAMA_HOST"):
            kwargs["base_url"] = ollama_host
        return ChatOllama(
            model=MODELS["ollama"][tier],
            temperature=temperature,
            callbacks=callbacks,
            **kwargs,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODELS["openai"][tier], temperature=temperature,
                          callbacks=callbacks)
    elif provider == "vllm":
        # vLLM speaks the OpenAI API, so the OpenAI client works unchanged.
        # api_key is required by the client but ignored by a local server.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=VLLM_MODEL,
            temperature=temperature,
            base_url=VLLM_BASE_URL,
            api_key=os.getenv("PLAYZORK_VLLM_API_KEY", "not-needed"),
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {provider}")


def get_cheap_llm(temperature: float = 0):
    """Get the cheap LLM instance (research, summarization, deduplication)."""
    return _build_llm(LLM_PROVIDER, "cheap", temperature)


def get_expensive_llm(temperature: float = 0):
    """Get the expensive LLM instance (decisions, proposals, observation)."""
    return _build_llm(LLM_PROVIDER, "expensive", temperature)
