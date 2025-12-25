# Multi-Agent Deliberation for Long-Horizon Sequential Decision Making

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
2. Mission agents evaluate state and each propose an action
3. Explorer agent proposes an exploratory action
4. Arbiter receives all proposals and selects one
5. Environment executes the selected action
6. Results update shared state
7. Repeat

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

* **No task completion:** The current system does not solve *Zork I* and becomes stuck early in gameplay. No claims are made about end-to-end task completion.
* **Limited evaluation:** Results are qualitative and based on a small number of runs. No statistical guarantees or comparative benchmarks are provided.
* **Model dependence:** Experiments use a single locally hosted LLM configuration; results may not generalize across models or scales.
* **Manual issue definition:** Issue agents are currently defined manually; automatic discovery, merging, and retirement of issues is not yet implemented.
* **Unmeasured tradeoffs:** While the architecture makes exploration explicit, optimal weighting between exploration and exploitation remains an open question.

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

## Current Status (v0.1-arxiv)

See `STATUS.md` for detailed system status.

**Working:**

* Two-phase agent architecture (research → decision)
* Tool-based memory access (history and memory toolkits)
* Local LLM inference (Llama 3.3 via Ollama)
* Structured output enforcement
* State persistence across turns

**Not Working:**

* Does not solve *Zork I* (gets stuck early)
* Shallow tool usage in research phase
* Memory tools underutilized
* Command loops not fully prevented

This tag represents a **pre–multi-agent baseline**: tool infrastructure in place, single-agent architecture stable, but reasoning quality insufficient for sustained progress.

---

## Implementation

This project uses **UV** for dependency management and **Ollama** for local LLM inference.

### Prerequisites

Ollama running with Llama 3.3:

```
ollama pull llama3.3
ollama serve
```

Optional `.env` configuration:

```
OLLAMA_HOST="http://[your-host]:11434"
```

### Running the System

```
uv sync
uv run python VersionTwo/main.py
```

```
Project Structure
```

```
VersionTwo/
├── adventurer/           # Agent decision logic
│   ├── adventurer_service.py
│   └── prompt_library.py
├── tools/
│   ├── history/         # History toolkit with summarization
│   └── memory/          # Memory toolkit with importance scoring
├── zork/                # Zork API client
├── game_session.py      # Main game loop
└── display_manager.py   # Terminal UI

STATUS.md
NOTES.md
```

---

## Research Questions

* Does explicit arbitration improve long-horizon decision quality compared to single-shot inference?
* How should exploration be weighted against exploitation in adversarial puzzle environments?
* Can specialized advocacy agents improve reasoning transparency without sacrificing performance?
* What is the minimum viable state representation for effective multi-agent coordination?

---

## Citation

If you use this work, please cite:

```
@misc{playzork2025,
  title={Multi-Agent Deliberation for Long-Horizon Sequential Decision Making},
  author={Michael Lane},
  year={2025},
  howpublished={\url{https://github.com/arsindelve/PlayZork}},
  note={v0.1-arxiv}
}
```

---

## Acknowledgments

Built with LangChain, Ollama, and Llama 3.3. *Zork I* © Infocom, 1980.
