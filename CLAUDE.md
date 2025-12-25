# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PlayZork is an experimental project that uses LLMs to autonomously play the classic 1980s text adventure game Zork. The project addresses the "Mr. Meeseeks problem" - LLMs lack long-term memory across API calls, so this implementation programmatically gives the LLM memory by providing complete game state context with each invocation.

The goal: Can an LLM beat Zork (reach 350 points) if given proper memory and context?

## Repository Structure

- **VersionTwo/** - Current Python implementation (active development)
- **VersionOne/** - Earlier C# implementation (archived)

## Development Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (creates .venv and installs packages)
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run the main Zork playing agent
uv run python VersionTwo/main.py
```

The game will run for 25 turns automatically, with the AI making decisions at each step.

## Architecture

### Core Components

The system follows a three-layer architecture:

1. **GameSession** (VersionTwo/game_session.py:5) - Orchestrates the game loop
   - Manages the main gameplay loop (25 turns by default)
   - Coordinates between ZorkService and AdventurerService
   - Entry point: `play()` method initializes game and processes turns

2. **ZorkService** (VersionTwo/zork/zork_service.py:6) - External Zork API integration
   - Communicates with hosted Zork game API at AWS
   - Endpoint: `https://bxqzfka0hc.execute-api.us-east-1.amazonaws.com/Prod/ZorkOne`
   - Sends player commands, receives game state responses
   - Returns ZorkApiResponse objects containing: Response text, LocationName, Moves, Score

3. **AdventurerService** (VersionTwo/adventurer/adventurer_service.py:10) - AI decision engine
   - Uses LangChain with OpenAI models (GPT-3.5-turbo for history, GPT-4 for decisions)
   - Receives game state and returns structured JSON commands via `with_structured_output()`
   - Returns AdventurerResponse with: command, reason, remember, rememberImportance, item, moved

### Memory System - The Key Innovation

The memory system solves the LLM statelessness problem through two mechanisms:

**HistoryProcessor** (VersionTwo/adventurer/history_processor.py:7)
- Maintains both raw history and LLM-generated summarized history
- Uses a cheaper LLM (GPT-3.5) to continuously summarize past interactions
- On each turn: appends new interaction → generates new summary → passes summary to decision LLM
- This creates "Leonard's tattoos from Memento" - persistent context across stateless API calls

**PromptLibrary** (VersionTwo/adventurer/prompt_library.py:1)
- Stores all prompts as static methods
- `get_adventurer_prompt()`: Main decision prompt with JSON schema for structured output
- `get_system_prompt()`: Game objective and rules (target: 350 points)
- `get_history_processor_*_prompt()`: Templates for summarizing game history

### Data Flow

```
GameSession.play_turn(input)
  → ZorkService.play_turn(input)
    → ZorkApiClient posts to AWS API
    → Returns ZorkApiResponse
  → AdventurerService.handle_user_input(ZorkApiResponse)
    → HistoryProcessor provides summarized context
    → LangChain chain invokes GPT-4 with structured output
    → Returns AdventurerResponse.command
  → Loop continues with new command
```

## Key Implementation Details

### Structured Output Format

The AdventurerResponse schema (VersionTwo/adventurer/adventurer_response.py:5) enforces:
- `command`: Next Zork command (e.g., "NORTH", "TAKE LAMP")
- `reason`: Why this command was chosen
- `remember`: Novel critical information to persist (avoid duplicates - memory is limited)
- `rememberImportance`: 1-1000 score (low-importance items may be forgotten)
- `item`: New items discovered in location
- `moved`: Direction attempted if movement command

### LLM Usage Strategy

- **GPT-3.5-turbo**: History summarization (cheap, frequent)
- **GPT-4**: Decision making (expensive, needs reasoning power)
- Temperature: 0 (deterministic behavior)

### Dependencies

- LangChain (prompts, chains, structured output)
- langchain-openai (ChatOpenAI)
- httpx (async HTTP client for Zork API)
- Pydantic (data models with validation)

## Environment Setup

The project requires OpenAI API credentials. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-actual-api-key
```

## Dependencies

Managed via `pyproject.toml`:
- **langchain** - LLM orchestration and prompt management
- **langchain-openai** - OpenAI integration for LangChain
- **httpx** - Async HTTP client for Zork API calls
- **pydantic** - Data validation and structured outputs
- **python-dotenv** - Environment variable management from .env files

## Two Versions

**VersionTwo** (Python): Current implementation, uses LangChain for LLM orchestration
**VersionOne** (C#): Earlier implementation with similar architecture but .NET stack