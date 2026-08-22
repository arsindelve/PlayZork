# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PlayZork is a research project using LLMs to autonomously play classic text adventure games (Zork I, Planetfall, and a local Escape Room test game). It has evolved from a single-agent memory experiment into a **multi-agent deliberation architecture**: specialist agents advocate for competing objectives, and a separate arbiter chooses one action per turn. The README.md is a working research draft describing this architecture; STATUS.md and NOTES.md are dated development logs.

The core hypothesis: separating proposal generation (advocacy) from action selection (arbitration) improves long-horizon sequential decision making compared to single-shot inference.

## Repository Structure

- **VersionTwo/** - Current Python implementation (active development)
- **VersionOne/** - Earlier C# implementation (archived, do not modify)
- **VersionTwo/docs/WORLD_MODEL_PROPOSAL.md** - Design proposal for the next direction (structured world model)

## Development Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (creates .venv and installs packages)
uv sync

# Set up environment variables
cp .env.example .env

# Install/start the default local LLM and fetch its model
brew install ollama
brew services start ollama
ollama pull qwen2.5:14b

# Run the main playing agent (from the repo root — paths are CWD-relative)
uv run python VersionTwo/main.py
```

`run_playzork.py` at the repo root is a PyCharm play-button entry point that chdirs to the project root and launches `VersionTwo/main.py` with the same paths.

The game runs until interrupted with Ctrl+C, with the AI making decisions at each step. Runtime artifacts (all CWD-relative, so always run from the repo root):

- `data/zork_sessions.db` - SQLite persistence (all state)
- `logs/game_<SESSION_ID>.log` - detailed per-session log (root logger writes here)
- `logs/sessions/<SESSION_ID>/Turn-N.html` + `index.html` - per-turn HTML reports

## Configuration

All runtime configuration lives in `VersionTwo/config.py`, driven by `.env` (loaded from the repo root at import time):

- `PLAYZORK_GAME` - active backend: `zork`, `planetfall`, or `escaperoom` (default `zork`). `GAME_NAME`, `GAME_OBJECTIVE`, and `GAME_OBJECTIVE_SCORE` are derived from the selected backend and interpolated into every prompt.
- `PLAYZORK_SESSION_ID` - session ID for both the game backend and local persistence. **Sessions resume**: reusing an ID continues turn numbering and keeps prior turns/memories/map in the DB.
- `PLAYZORK_LLM_PROVIDER` - `ollama` (default) or `openai`
- `OLLAMA_HOST` / `OPENAI_API_KEY` - provider credentials

Game backends (`GAME_BACKENDS` in config.py): Zork I and Planetfall are hosted AWS APIs; Escape Room expects a server at `localhost:5000` (not part of this repo).

**Model tiers** (`MODELS` in config.py, accessed only via `get_cheap_llm()` / `get_expensive_llm()` — instances are memoized per (provider, tier, temperature)):
- Cheap: research, summarization, deduplication, inventory analysis, death detection
- Expensive: decisions, agent proposals, observation, history summarization
- Ollama uses `qwen2.5:14b` for both tiers (stays warm, no model swap); temperature 0 throughout

**Timeouts (coupled — don't tune one alone):** `LLM_TIMEOUT_SECONDS` (per *attempt*, default 180, `PLAYZORK_LLM_TIMEOUT_SECONDS`) × `LLM_MAX_RETRIES` (default 3, `PLAYZORK_LLM_MAX_RETRIES`) plus exponential backoff = `LLM_RETRY_ENVELOPE_SECONDS`, the worst case for one guarded call (`llm_utils.invoke_with_retry` / `ainvoke_with_retry`). `TURN_BUDGET_SECONDS` (default 1200, `PLAYZORK_TURN_BUDGET_SECONDS`) is the wall-clock cap on the whole per-turn decision graph and must exceed that envelope, or retries 2..N are unreachable — config enforces a floor of `2 ×` the envelope and logs a warning when it raises a configured value. `config.retry_envelope_seconds()` must stay in sync with the backoff schedule in `llm_utils`.

**Failure containment:** a failed turn is not a failed run. `invoke_tool_safely` (`tools/agent_graph/tool_execution.py`) turns any model-supplied tool call error into an `"Error: ..."` string fed back as the tool result; spawn uses `gather(return_exceptions=True)` and neutralizes failed agents; `close_issues`/`observe`/`persist` are individually guarded so post-decision failures can't discard a chosen command; `GameSession.play()` recovers a failed turn with `FALLBACK_COMMAND` and only gives up after `MAX_CONSECUTIVE_TURN_FAILURES`. Memory closures are **staged** by `IssueClosedAgent` and applied by `persist_node` last, so a cancelled turn never half-applies memory state.

## Architecture

### Per-Turn Flow

`GameSession.play()` (VersionTwo/game_session.py) loops indefinitely:

1. **ZorkService** POSTs the command to the game API → `ZorkApiResponse` (Response, LocationName, Score, Moves)
2. **HistoryToolkit.update_after_turn** (`async`, must be awaited) - stores the turn, then regenerates the two LLM summaries (recent + long-running) **concurrently** via `asyncio.gather`. They are independent until `save_both_summaries`; if either fails neither is committed, so the previous turn's summaries stay in place. This still runs *before* the decision graph and so is outside `TURN_BUDGET_SECONDS` — `ainvoke_with_retry`'s per-attempt timeout is what bounds it. Moving it off the critical path entirely is [#24](https://github.com/arsindelve/PlayZork/issues/24) option 2 (M4).
3. **MapperToolkit.update_after_turn** - records the location transition; a movement command with no location change is recorded as `location --[DIR]--> BLOCKED` so the explorer never re-suggests it
4. **AdventurerService.handle_user_input** runs the LangGraph decision graph (below) and returns the next command
5. Display update (Rich terminal UI via **DisplayManager**)
6. Post-turn work (BigPictureAnalyzer, DeathAnalyzer, HTML report, session index) is dispatched as a **background task** (`_dispatch_post_turn_io`) so the next turn doesn't wait; tasks are drained at shutdown

The HTML report for a turn shows the agents/decision that **led to** that turn's command, so `GameSession` carries the previous turn's decision data in a `PendingDecision` dataclass.

### The Decision Graph

`tools/agent_graph/decision_graph.py` builds a LangGraph pipeline:

```
SpawnAgents → Research → Decide → CloseIssues → Observe → Persist → END
```

- **SpawnAgents** - creates specialist agents and runs all their research+proposal passes concurrently (`asyncio.gather`):
  - **IssueAgent** (up to 5): one per tracked strategic issue from the memory DB, top 5 by lazily-decayed importance. Each researches with tools (including pathfinding to its issue's location and inventory checks) and proposes an action with confidence 1-100.
  - **ExplorerAgent** (0 or 1): spawned if the current location has unexplored directions; picks the best direction deterministically (mentioned-in-description > cardinals > diagonals > up/down) and computes confidence heuristically.
  - **InteractionAgent** (always): proposes local object interactions from the current room description + inventory.
  - **LoopDetectionAgent**: **currently disabled** (`loop_detection_agent = None` in spawn_agents_node), but the class and its display/report plumbing remain.
- **Research** - a cheap-LLM research agent (`tool_choice="any"`) gathers turn-level context; its bound tools and the execution map must stay in sync (history + mapper + inventory + analysis tools).
- **Decide** - the **arbiter**. Receives all proposals formatted with expected-value scores (`_format_agent_proposals`) and picks exactly one command; it evaluates, it does not generate. Returns a structured `AdventurerResponse` (command, reason, remember, rememberImportance, item, moved).
- **CloseIssues** - `IssueClosedAgent` marks resolved issues closed in the memory DB.
- **Observe** - `ObserverAgent` scans the game response for new strategic issues to track.
- **Persist** - stores the observer's new issue (after exact + LLM semantic dedup via `MemoryDeduplicator`) and runs `InventoryAnalyzer` on the executed command to update inventory state.

### Toolkits (facades over shared SQLite state)

Each toolkit wraps a state class and exposes LangChain `@tool` functions via module-level initialization (`initialize_*_tools(...)` then `get_*_tools()`):

- **tools/history/** - turns + dual summaries; tools: `get_recent_turns`, `get_full_summary`
- **tools/mapping/** - location graph with BFS pathfinding (`pathfinder.py`); tools: `get_map`, `get_exits_from_location`, `find_path_between_locations`, `get_direction_to_location`
- **tools/inventory/** - item tracking, bootstrapped at game start by sending `INVENTORY` and LLM-parsing the response
- **tools/memory/** - strategic issue storage. **Write-only by design**: issues are flagged via the observer's `remember` field and read only by the spawn node; `memory_tools.py`/`memory_retriever.py` read-tools exist but are not registered anywhere.
- **tools/analysis/** - `get_strategic_analysis` (latest BigPictureAnalyzer output, tolerates one-turn lag), plus `DeathAnalyzer` (detects deaths, analyzes cause, persists lessons-learned)
- **tools/database/db_manager.py** - single DatabaseManager for all tables: sessions, turns, summaries, memories, map_transitions, inventory, strategic_analysis, deaths
- **tools/reporting/turn_report_writer.py** - per-turn HTML reports + session index

### Memory Importance Decay

Issue importance (1-1000) decays **lazily on read** (`MemoryState.get_top_memories` with `current_turn`), not via per-turn UPDATEs. New issues therefore naturally outrank stale ones.

## Key Implementation Details

- **Structured output everywhere**: `AdventurerResponse`, `IssueProposal`, `ExplorerProposal`, etc. are Pydantic models enforced via `with_structured_output()`. Local models sometimes return wrong types — code defensively coerces (e.g., importance int coercion in MemoryState).
- **All prompts live in `adventurer/prompt_library.py`** as static methods, with `GAME_NAME`/`GAME_OBJECTIVE` interpolated from config. Edit prompts there, not inline.
- **Async convention**: hot-path agents use `ainvoke_with_retry`; `ObserverAgent` and `IssueClosedAgent` still use the sync `invoke_with_retry` inside sync graph nodes.
- **LangSmith run names**: LLM invocations use `.with_config(run_name=...)` for traceability — keep this when adding calls.
- **Legacy/dead code** (don't extend, candidates for removal): `adventurer/history_processor.py` (superseded by `tools/history/history_summarizer.py`), `tools/memory/memory_tools.py` + `memory_retriever.py` (unregistered), `PromptLibrary.get_adventurer_prompt`/`get_system_prompt` (pre-multi-agent), `LoopDetectionAgent` (disabled).

## Testing

```bash
# Run from VersionTwo (pytest.ini + conftest.py live there and set up sys.path)
cd VersionTwo && uv run python -m pytest
```

Tests cover the pathfinder and decision-graph persist node. There is no integration-test coverage of the live game loop — verifying agent behavior means running a session and reading `logs/` and the HTML reports.

## Dependencies

Managed via `pyproject.toml`: langchain / langchain-openai / langchain-ollama / **langgraph** (decision graph), httpx (async game API client), pydantic (structured outputs), rich (terminal UI), python-dotenv. Dev: pytest.

## Ollama Troubleshooting

**For VM/Remote Setup**: If Ollama runs on a different host (e.g., macOS host from VM):
1. Start Ollama with network binding:
   ```bash
   OLLAMA_HOST=0.0.0.0:11434 OLLAMA_ORIGINS="*" ollama serve
   ```
2. Use IPv6 address in `.env` (IPv6 often works when IPv4 fails):
   ```
   OLLAMA_HOST=http://[fd9e:f32d:415e:4a47:1037:abbc:765d:3da2]:11434
   ```
   (Replace with your host's actual IPv6 address from `ifconfig`)

**Common Issues**:
- If Ollama connection fails, verify it's listening: `lsof -i :11434` on the host
- IPv6 format requires brackets: `http://[ipv6-address]:11434`
- Test connectivity: `curl -g "http://[your-ipv6]:11434/api/tags"`
