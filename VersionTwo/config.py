"""
Configuration for LLM models used throughout the application.

Update these model names in one place to change models globally.
"""

# Ollama model names
CHEAP_MODEL = "llama3.2"      # Used for: research, summarization, deduplication
EXPENSIVE_MODEL = "llama3.3"  # Used for: decision making, agent proposals
