# Multi-Agent Deliberation for Long-Horizon Sequential Decision Making

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18224702.svg)](https://doi.org/10.5281/zenodo.18224702)

> **Status:** Working research draft. Architecture and motivation are stable; results are preliminary.

---

## Abstract

Large language model (LLM) agents continue to struggle with long-horizon sequential decision problems, even when augmented with persistent memory. Using *Zork I* as a challenging interactive fiction testbed, we observe that single-pass inference forces a model to simultaneously track multiple unresolved objectives, manage exploration–exploitation tradeoffs, and arbitrate between competing priorities, leading to looping and incoherent behavior. We propose a multi-agent deliberation architecture in which specialized agents advocate for individual objectives, a dedicated explorer proposes information-gathering actions, and a separate arbitration step selects a single action to execute. This work introduces the architecture and reports preliminary observations from a working prototype; comprehensive empirical evaluation is left to future work.

---

## The Research Problem

Single-shot LLM inference—even when augmented with persistent memory—struggles with long-horizon problems that require sustained attention to multiple competing objectives. When asked:

> "Given everything so far, what should I do next?"

A single model call must simultaneously:

* Track multiple unsolved puzzles
* Maintain spatial awareness across visited locations
* Remember partial progress on distinct threads
* Balance exploitation (solving known puzzles) vs. exploration (discovering new state)
* Arbitrate between competing priorities

This cognitive load increases with problem horizon. We hypothesize that **explicit separation of advocacy and arbitration** can improve decision quality by:

* Distributing reasoning across specialized agents
* Making exploration a visible, accountable voice
* Introducing a dedicated arbitration step that chooses without generating

---

## Architecture: Multi-Agent Deliberation with Explicit Arbitration

### Core Idea

Instead of asking a single model to both generate and choose an action:

```
[ Single LLM ] → Decision
```

we decompose the process into proposal and arbitration:

```
[ Mission Agent 1 ] → Proposal A
[ Mission Agent 2 ] → Proposal B
[ Explorer Agent   ] → Proposal C
            ↓
        [ Arbiter ] → Decision
```

No single agent bears the full burden of choice.

---

## Agent Types

### 1. Mission / Puzzle / Issue Agents

Each Mission Agent represents one concrete unresolved concern, such as:

* An unsolved puzzle (e.g., "How do I cross the river?")
* A partially completed task (e.g., "Find food for the cyclops")
* A blocked hypothesis (e.g., "The brass lantern might need fuel")

**Responsibilities:**

* Read-only access to the current game state
* Propose a single action that advances its specific issue
* Provide a short justification
* Optionally signal confidence or urgency

These agents are intentionally narrow, stubborn, and single-minded. They do not need global coherence. Their role is **advocacy**, not balance. They generate arguments, not decisions.

---

### 2. The Explorer Agent

The Explorer is a first-class agent, not a fallback.

**Mission:** Reduce unknowns, expand the state space, and break local minima.

The Explorer:

* Proposes actions that maximize information gain
* Explicitly competes against mission agents
* May lose arbitration frequently (and that is acceptable)

Crucially:

* Exploration is always present
* Exploration is never implicit
* Exploration can be measured, tuned, and blamed

This makes the exploration–exploitation tradeoff explicit and visible rather than emergent and opaque.

---

### 3. The Arbiter

A separate reasoning step—potentially a distinct LLM call—acts as arbiter.

The Arbiter:

* Sees all proposed actions and justifications
* Sees a compact summary of the current state
* Chooses exactly one action to execute
* Does **not** generate actions itself

**Responsibilities:**

* Resolve conflicts between competing agents
* Prevent thrashing and loops
* Decide when exploration outweighs exploitation
* Decide which issue receives attention at the current step

This is where decision-making is explicitly separated from proposal generation.

---

## Control Loop

Each turn executes the following sequence:

1. Update shared state (facts, inventory, map, memory)
2. Assemble a **deterministic turn context** in code — inventory, map, exits, routing to each tracked issue, and what has already been tried here without effect. No LLM call: the data is deterministic, so fetching it directly is both faster and more reliable than asking a model to request it
3. Mission agents evaluate that context and each propose an action
4. Explorer agent proposes an exploratory action
5. Arbiter receives all proposals, with the expected value of any already-refuted action zeroed, and selects one
6. Environment executes the selected action
7. Results update shared state
8. Repeat

Issue discovery and retirement run **concurrently** with proposal and arbitration, since neither depends on the decision; a single writer commits all state changes at the end of the turn.

No single agent ever holds the full cognitive burden.

---

## Formalization (Pseudocode)

The deliberation loop can be expressed as the following high-level procedure:

```
state ← initialize_state()
issues ← initialize_issue_agents()
explorer ← initialize_explorer_agent()
arbiter ← initialize_arbiter()

while not terminated:
    state ← update_state_from_environment(state)

    proposals ← []

    for issue in issues:
        proposal ← issue.propose_action(state)
        proposals.append(proposal)

    explore_proposal ← explorer.propose_action(state)
    proposals.append(explore_proposal)

    action ← arbiter.select_action(proposals, state)

    observation ← environment.step(action)
    state ← integrate_observation(state, observation)
```

This formulation highlights the explicit separation between **proposal generation** and **decision selection**, as well as the fact that arbitration operates over competing, simultaneously generated action candidates rather than a single linear reasoning trace.

---

## Claims and Non-Claims

### This architecture claims:

* Long-horizon decisions require explicit arbitration
* Competing objectives must be represented simultaneously
* Exploration must be a visible, accountable voice
* Separating advocacy from arbitration reduces cognitive load per inference call

### This architecture does **not** claim:

* Optimality
* Guaranteed success on *Zork I*
* That individual agents exhibit intelligence
* That this is the only viable architecture

It claims only that this structure can represent and arbitrate competing priorities more explicitly than single-shot inference, making the decision process more transparent and potentially more robust over long horizons.

---

## Limitations

This work reports on a preliminary prototype and intentionally limits the scope of its claims.

* **No task completion:** The current system does not solve *Zork I*. No claims are made about end-to-end task completion.
* **Serving, not architecture, dominates measured latency:** on the development machine the local inference server was benchmarked at *flat* throughput across 1, 2, 4 and 8 concurrent requests — i.e. no effective parallelism. Wall-clock per turn is therefore a proxy for total tokens processed, and any architecture comparison must report token counts alongside time or run on serving that genuinely batches.
* **Limited evaluation:** Results are qualitative and based on a small number of runs. No statistical guarantees or comparative benchmarks are provided.
* **Model dependence:** Experiments use a single locally hosted LLM configuration; results may not generalize across models or scales.
* **Unvalidated issue lifecycle:** Issue discovery, semantic de-duplication, and retirement are automated via LLM observers, but the quality of that lifecycle (missed issues, premature closure, duplicate leakage) has not been measured.
* **Unmeasured tradeoffs:** While the architecture makes exploration explicit, optimal weighting between exploration and exploitation remains an open question.
* **Scaffolding correctness is a precondition, not a result:** a systematic audit of the deterministic components (map, inventory, memory) found errors that silently corrupted agent inputs — a mis-named tool that made inventory invisible, successful moves recorded as permanent walls, and same-named rooms merged into one node. These are now fixed and regression-tested, but the episode is itself a finding: a weak model cannot compensate for corrupted scaffolding, and such failures are invisible without deliberate instrumentation.

These limitations are not incidental; they define the boundary of the present contribution, which is architectural and methodological rather than performance-driven.

---

## Related Work (Overview)

This work intersects several existing research directions but differs in its explicit separation of advocacy and arbitration.

### LLM Agents and Tool Use

Prior work on LLM agents emphasizes single-agent reasoning loops augmented with tools and memory (e.g., ReAct-style prompting and tool-augmented agents). While these approaches improve short-horizon reasoning, they place the full burden of proposal generation and decision-making within a single inference step, which can become brittle as task horizons grow.

### Text-Based Games and Interactive Fiction

Text-based games have long served as challenging environments for sequential decision-making, requiring language understanding, spatial reasoning, and long-term planning. Recent benchmarks demonstrate that even strong LLMs struggle to make sustained progress in classic interactive fiction games, highlighting persistent limitations in long-horizon control.

### Planning, Arbitration, and Multi-Agent Systems

Multi-agent systems and planning frameworks often distribute roles across agents or modules. However, many such systems either rely on a centralized planner or implicitly resolve conflicts within a single decision function. In contrast, the present work makes conflict explicit by separating proposal generation from arbitration and by treating exploration as a first-class competing objective.

This paper positions itself as an architectural and methodological contribution, complementing existing benchmarks and agent frameworks rather than competing directly on task completion.

---

## Current Status

See `STATUS.md` for dated development logs. The tag `v0.1-arxiv` (the archived Zenodo deposit) is a **pre–multi-agent baseline**; the architecture described above is now implemented on `main`.

**Working:**

* Multi-agent deliberation per turn: up to 5 IssueAgents + ExplorerAgent + InteractionAgent propose in parallel; a separate Decision Agent arbitrates
* Automatic issue lifecycle: an ObserverAgent discovers new strategic issues, semantic de-duplication merges them, an IssueClosedAgent retires resolved ones, and importance decays lazily over turns
* Map graph with BFS pathfinding and blocked-direction tracking (failed movements are recorded and never re-proposed)
* Inventory tracking, death analysis with lessons-learned persistence, and dual (recent + long-running) history summaries
* SQLite persistence with session resumption; per-turn HTML reports of every proposal and decision
* Structured output enforcement throughout; local LLM inference (Qwen 2.5 14B via Ollama) or OpenAI
* Multiple game backends: *Zork I*, *Planetfall* (hosted APIs), and a local Escape Room test game
* **Both experiment arms implemented:** `PLAYZORK_CONDITION=multi_agent | single_shot` selects the architecture or the single-inference control, which receives the same information and the same model
* **Per-turn token accounting** (`turn_tokens`), so runs are comparable across machines and the architecture is charged for what it costs rather than for how fast the host happens to be

**Not Working:**

* Does not solve *Zork I* (gets stuck early)
* Command loops not fully prevented (a dedicated LoopDetectionAgent exists but is disabled as ineffective)
* Evaluation remains qualitative; no comparative benchmarks yet

---

## Implementation

This project uses **UV** for dependency management and **Ollama** for local LLM inference.

### Prerequisites

Ollama running with the configured Qwen model:

```
brew install ollama
brew services start ollama
ollama pull qwen2.5:14b
```

Create the local runtime configuration:

```bash
cp .env.example .env
```

### Running the System

```
uv sync --locked
uv run python VersionTwo/main.py
```

Runtime is configured via `.env`: `PLAYZORK_GAME` (zork | planetfall | escaperoom), `PLAYZORK_SESSION_ID` (sessions resume), and `PLAYZORK_LLM_PROVIDER` (ollama | openai). The run writes state to `data/zork_sessions.db`, logs to `logs/`, and per-turn HTML reports to `logs/sessions/<session-id>/`.

### Project Structure

```
VersionTwo/
├── adventurer/              # Arbiter chain + all prompts
│   ├── adventurer_service.py    # Builds the decision chain and decision graph
│   ├── single_shot_service.py   # Control arm: one inference per turn, full context
│   ├── adventurer_response.py   # Structured output schema for the arbiter
│   └── prompt_library.py        # Every prompt in the system (static methods)
├── tools/
│   ├── agent_graph/         # LangGraph pipeline + deliberating agents
│   │   ├── decision_graph.py    # BuildContext → (Spawn → Decide | Close | Observe) → Persist
│   │   ├── turn_context.py      # Deterministic per-turn facts, assembled in code
│   │   ├── issue_agent.py       # One advocate per tracked strategic issue
│   │   ├── explorer_agent.py    # Advocates for unexplored directions
│   │   ├── interaction_agent.py # Advocates for local object interactions
│   │   ├── loop_detection_agent.py  # Loop breaker (currently disabled)
│   │   ├── issue_closed_agent.py    # Retires resolved issues
│   │   └── observer_agent.py        # Discovers new issues
│   ├── history/             # Turn history + dual LLM summaries
│   ├── memory/              # Strategic issue store (write-only, with semantic dedup)
│   ├── mapping/             # Location graph, BFS pathfinding, blocked-direction tracking
│   ├── inventory/           # Item tracking with LLM turn analysis
│   ├── analysis/            # Big-picture strategy + death analysis
│   ├── database/            # Shared SQLite persistence
│   └── reporting/           # Per-turn HTML reports
├── zork/                    # Game API client (Zork, Planetfall, Escape Room backends)
├── game_session.py          # Main game loop
├── config.py                # Game/provider/model configuration (.env-driven)
└── display_manager.py       # Rich terminal UI

STATUS.md                    # Dated development logs
NOTES.md                     # Research notes
```

---

## Measurement Notes

Two findings from instrumenting the system are worth stating up front, because they shape how any result here should be read.

**Wall-clock measures token volume, not architecture.** The development machine's local inference server delivers flat throughput regardless of request concurrency (0.26 req/s at 1, 2 and 4 concurrent, with realistic prompts). Restructuring *when* calls happen therefore cannot help; only reducing total tokens can. Consequently `score@wall-clock` is reported alongside token counts, and comparisons across hardware use tokens as the common unit.

**Removing calls works; reordering them does not.** Replacing per-agent LLM "research" round-trips with deterministic context assembly cut per-turn calls from `10 + 2N` to `5 + N` for N tracked issues and halved turn time. A subsequent change that ran independent work concurrently produced no reliable improvement on the same hardware, for the reason above.

---

## Research Questions

* Does explicit arbitration improve long-horizon decision quality compared to single-shot inference?
* How should exploration be weighted against exploitation in adversarial puzzle environments?
* Can specialized advocacy agents improve reasoning transparency without sacrificing performance?
* What is the minimum viable state representation for effective multi-agent coordination?

---

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18224702.svg)](https://doi.org/10.5281/zenodo.18224702)

If you use this work, please cite:

```
@misc{playzork2025,
  title={Multi-Agent Deliberation for Long-Horizon Sequential Decision Making},
  author={Michael Lane},
  orcid={0009-0006-5381-6080},
  year={2025},
  doi={10.5281/zenodo.18224702},
  url={https://zenodo.org/records/18224702},
  howpublished={\url{https://github.com/arsindelve/PlayZork}},
  note={v0.1-arxiv}
}
```

---

## Acknowledgments

Built with LangChain, LangGraph, Ollama, and Qwen 2.5. *Zork I* and *Planetfall* © Infocom.
