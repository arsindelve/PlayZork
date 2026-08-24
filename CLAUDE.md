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
- `PLAYZORK_LLM_PROVIDER` - `ollama` (default), `openai`, or `vllm`
- `OLLAMA_HOST` / `OPENAI_API_KEY` - provider credentials; vLLM uses `PLAYZORK_VLLM_BASE_URL` + `PLAYZORK_VLLM_MODEL`
- `PLAYZORK_CONDITION` - **the thesis's independent variable**: `multi_agent` (default) or `single_shot`. Both implement the same interface, so everything downstream is unaffected by which arm runs.
- `BIG_PICTURE_HISTORY_TURNS` (`PLAYZORK_BIG_PICTURE_HISTORY_TURNS`, default 20) - how much raw history the BigPictureAnalyzer folds into its expensive-model prompt every turn. It is the dominant term in per-turn latency growth.

Game backends (`GAME_BACKENDS` in config.py): Zork I and Planetfall are hosted AWS APIs; Escape Room expects a server at `localhost:5000` (not part of this repo).

**Model tiers** (`MODELS` in config.py, accessed only via `get_cheap_llm()` / `get_expensive_llm()` — instances are memoized per (provider, tier, temperature)):
- Cheap: deduplication, inventory analysis (fallback only), death detection
- Expensive: decisions, agent proposals, observation, history summarization, big-picture analysis
- Ollama uses `qwen2.5:14b` for both tiers (stays warm, no model swap); temperature 0 throughout

**Timeouts (coupled — don't tune one alone):** `LLM_TIMEOUT_SECONDS` (per *attempt*, default 180, `PLAYZORK_LLM_TIMEOUT_SECONDS`) × `LLM_MAX_RETRIES` (default 3, `PLAYZORK_LLM_MAX_RETRIES`) plus exponential backoff = `LLM_RETRY_ENVELOPE_SECONDS`, the worst case for one guarded call (`llm_utils.invoke_with_retry` / `ainvoke_with_retry`). `TURN_BUDGET_SECONDS` (default 1200, `PLAYZORK_TURN_BUDGET_SECONDS`) is the wall-clock cap on the whole per-turn decision graph and must exceed that envelope, or retries 2..N are unreachable — config enforces a floor of `2 ×` the envelope and logs a warning when it raises a configured value. `config.retry_envelope_seconds()` must stay in sync with the backoff schedule in `llm_utils`.

**Failure containment:** a failed turn is not a failed run. `invoke_tool_safely` (`tools/agent_graph/tool_execution.py`) turns any model-supplied tool call error into an `"Error: ..."` string fed back as the tool result; spawn uses `gather(return_exceptions=True)` and neutralizes failed agents; `close_issues`/`observe`/`persist` are individually guarded so post-decision failures can't discard a chosen command; `GameSession.play()` recovers a failed turn with `FALLBACK_COMMAND` and only gives up after `MAX_CONSECUTIVE_TURN_FAILURES`. Memory closures are **staged** by `IssueClosedAgent` and applied by `persist_node` last, so a cancelled turn never half-applies memory state.

## Architecture

### Per-Turn Flow

`GameSession.play()` (VersionTwo/game_session.py) loops indefinitely:

1. **ZorkService** POSTs the command to the game API → `ZorkApiResponse` (Response, LocationName, Score, Moves)
2. **HistoryToolkit.record_turn** - stores the turn. **Synchronous and on the critical path**, because this turn's agents research against it via `get_recent_turns`. Summarization is *not* here: `refresh_summaries()` is dispatched as a tracked background task and coalesced under a lock (#24 option 2), since folding the newest turn into prose buys almost nothing for the decision being made right now.
3. **MapperToolkit.update_after_turn** - records the location transition. A `BLOCKED` edge is written **only when the response text explicitly refuses the move** (#10/#33) — inferring it from "the room name didn't change" wrote false permanent walls. Death turns are skipped entirely (#12). Rooms are identified by name **+ exits signature**, so same-named rooms don't merge (#15).
4. **AdventurerService.handle_user_input** runs the LangGraph decision graph (below) and returns the next command. `SingleShotService` is the drop-in alternative for the experiment's control arm.
5. Display update (Rich terminal UI via **DisplayManager**)
6. Post-turn work (BigPictureAnalyzer, DeathAnalyzer, HTML report, session index) is dispatched as a **background task** (`_dispatch_post_turn_io`) so the next turn doesn't wait; tasks are drained at shutdown
7. Token accounting is closed out at the **start of the next turn**, so background work is attributed to the turn that spawned it (`_record_previous_turn_tokens` → `turn_tokens` table)

The HTML report for a turn shows the agents/decision that **led to** that turn's command, so `GameSession` carries the previous turn's decision data in a `PendingDecision` dataclass.

### The Decision Graph

`tools/agent_graph/decision_graph.py` builds a LangGraph pipeline. It is a **diamond, not a chain** — `close_issues` and `observe` have no data dependency on the decision (both read the game response), so they run beside the spawn→decide branch and join at persist:

```
build_context ─┬─ spawn_agents → decide ─┐
               ├─ close_issues ──────────┤
               └─ observe ───────────────┴─ persist → END
```

Two invariants make the fan-out safe, and breaking either is subtle:

1. **Nodes return only their own keys.** A node returning the whole state reads to LangGraph as every node writing every key → concurrent-update error. Their write sets are disjoint by design.
2. **`persist` joins via a LIST start_key** — `add_edge(["decide", "close_issues", "observe"], "persist")`. Three separate `add_edge` calls are *not* a join: the branches differ in depth, so persist would run **twice per turn**. This was found in live play, not by the suite.

- **build_context** - assembles `TurnContext` in code (no LLM): inventory, recent turns, both summaries, exits, strategic analysis, precomputed pathfinding per tracked issue, the commands already shown to do nothing here (#18), and the game's own accepted-command list (#16/#30). Also reads the memory snapshot every branch shares.
- **SpawnAgents** - creates specialist agents and runs their proposal passes concurrently (`asyncio.gather(return_exceptions=True)`; a failed agent is neutralized, not fatal):
  - **IssueAgent** (up to 5): one per tracked strategic issue, top 5 by lazily-decayed importance. **One LLM call each** — the old research round-trip is gone (#25); its facts come from `TurnContext`.
  - **ExplorerAgent** (0 or 1): spawned only if the current location is known *and* has unexplored directions. Prose scanning for "mentioned" directions is whole-word (`find_mentioned_directions`), not substring.
  - **InteractionAgent** (always): proposes local interactions. Its deterministic parser is a **hint only and never short-circuits the LLM** (#16).
  - **LoopDetectionAgent**: **disabled** (`loop_detection_agent = None`). Arguably superseded by #18's deterministic repetition suppression, which addresses the cause rather than detecting the symptom.
- **Decide** - the **arbiter**. Receives proposals formatted with expected-value scores (`_format_agent_proposals`), which **zeroes the EV of any command already shown to do nothing here** (#18). Returns a structured `AdventurerResponse` (command, reason, moved).
- **CloseIssues** - `IssueClosedAgent` decides which issues are resolved. **Read-only**: it *stages* closures; persist applies them (#3).
- **Observe** - `ObserverAgent` scans the game response for new strategic issues.
- **Persist** - the graph's **single writer**. Stores the observer's issue (after exact + LLM semantic dedup), applies the staged closures **last** (after all cancellable LLM work), and reconciles inventory. On the hosted backends inventory comes from the game itself, so no LLM call is made (#30); `InventoryAnalyzer` is the fallback.

**Per-turn LLM calls: `5 + N`** for N tracked issues (was `10 + 2N` before #25).

### Toolkits (facades over shared SQLite state)

Each toolkit wraps a state class and exposes LangChain `@tool` functions via module-level initialization (`initialize_*_tools(...)` then `get_*_tools()`):

- **tools/history/** - turns + dual summaries; tools: `get_recent_turns`, `get_full_summary`
- **tools/mapping/** - location graph with BFS pathfinding (`pathfinder.py`); tools: `get_map`, `get_exits_from_location`, `find_path_between_locations`, `get_direction_to_location`
- **tools/inventory/** - item tracking. On the hosted backends the game reports its own inventory every turn, so it is **reconciled, not inferred** (#30); the LLM `InventoryAnalyzer` is the fallback for backends that omit the field. The DB is the source of truth and `get_items()` is a mirror refreshed after every write (#21)
- **tools/mapping/directions.py** - the direction vocabulary. `extract_direction` parses a player COMMAND (strict, token-based); `find_mentioned_directions` scans room PROSE (permissive, whole-word). Keep the distinction — reviewers try to merge them
- **tools/mapping/response_signals.py** - what the prose says about the move: `looks_like_death` (over-detects on purpose) and `is_movement_refusal` (under-detects on purpose). **The opposite biases are deliberate** and are why they are two functions rather than one classifier
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
- **Legacy/dead code** (don't extend, candidates for removal): `adventurer/history_processor.py` (superseded by `tools/history/history_summarizer.py`), `tools/memory/memory_tools.py` + `memory_retriever.py` (unregistered), `PromptLibrary.get_adventurer_prompt`/`get_system_prompt`/`get_research_agent_prompt` (the last is unreferenced since the research node was deleted in #25), `LoopDetectionAgent` (disabled).

## Running the experiment

The thesis compares two arms of the **same model** on the **same information**:

```bash
PLAYZORK_CONDITION=multi_agent  PLAYZORK_SESSION_ID=exp-ma-seed1 uv run python VersionTwo/main.py
PLAYZORK_CONDITION=single_shot  PLAYZORK_SESSION_ID=exp-ss-seed1 uv run python VersionTwo/main.py
```

`SingleShotService` (`adventurer/single_shot_service.py`) makes **one inference per turn** with everything in context. It is deliberately generous — a weak baseline would make the comparison meaningless — which also means it is arguably *too* generous, since it currently sees the map and tracked issues. See PLAN.md for the ablation ladder.

**Always report token counts alongside wall-clock.** `turn_tokens` records per-turn input/output/calls. On fixed serving, wall-clock is a proxy for token volume: this Mac's Ollama was benchmarked at **flat throughput across 1/2/4/8 concurrent requests**, so concurrency changes nothing and only tokens move the number. Without tokens, the multi-agent arm is penalised for verbosity in a way that says nothing about the architecture, and no cross-machine comparison is meaningful. Totals are a **floor** — structured-output calls carry no usage metadata and are skipped rather than estimated.

## Invariants worth not breaking

Each of these was learned by something going wrong, usually in live play rather than in the suite.

- **A false negative beats a false positive, everywhere in the world model.** BLOCKED edges (#11), movement refusals (#10/#33), repetition suppression (#18), room splitting (#15): a wrong "no" silently removes something real and nothing in the game text ever corrects it, whereas a wrong "yes" costs one turn and the game re-teaches it. When in doubt, record nothing.
- **Prompt text is not a mechanism.** A 14B model given a bare prohibition invents its own way around it — told "don't re-add a held item", it emitted a *removal* instead and emptied the inventory (#21). If a rule must hold, enforce it in code; the prompt is a hint.
- **Don't infer what the backend already reports.** Inventory, movement direction, room exits and per-object accepted commands all come from the API (#30). Each replaced an LLM call or a regex that was getting it wrong.
- **`invoke_with_retry` / `ainvoke_with_retry` are the only LLM entry points.** They give timeout, retry and token metering. A bare `.invoke()` is invisible to all three — three of sixteen calls per turn were unmeasurable until #24 fixed that.
- **Run the thing before believing the suite.** Five separate defects this week passed a green test run and were caught by a live session, and most were *silent* rather than loud — error containment (#1) means a broken component logs and continues rather than crashing.

## Testing

```bash
# Run from VersionTwo (pytest.ini + conftest.py live there and set up sys.path)
cd VersionTwo && uv run python -m pytest
```

587 tests. They cover the pathfinder, the decision graph and its topology, direction/location handling, room identity, repetition suppression, inventory state, token metering, provider selection and both experiment arms.

**There is still no integration coverage of the live game loop**, and that gap is real: every substantial refactor this week shipped green and was caught by a live run. Verifying agent *behaviour* means running a session and reading `logs/` and the HTML reports. Prefer tests that **execute** the thing over tests that inspect source text — a source-only assertion missed a deleted import that silently disabled the ObserverAgent for a whole session.

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
