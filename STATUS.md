# Status: v0.1-arxiv (2025-12-25)

## What Works

• Two-phase agent architecture (research → decision)
• History toolkit with tool-based access
• Memory toolkit with importance scoring
• Llama 3.3 local model for all reasoning
• Tool calling functional (get_recent_turns, get_full_summary, memory tools)
• Structured output enforced (AdventurerResponse schema)
• State persists across turns (history, memory)
• Dual summary system (recent + long-running)
• Rich terminal display with live updates
• 100% local except Zork API (Ollama via IPv6 to Mac host)

## What Does NOT Work

• Does not solve Zork (gets stuck early)
• Memory tools never called by agent
• Research agent shortcuts (only calls get_recent_turns)
• Gets stuck in command loops
• No multi-step reasoning (single tool call per research phase)
• Agent ignores get_full_summary and memory tools
• Exploration shallow and repetitive
• No progress beyond starting area in most runs

## What This Tag Represents

• First working two-phase architecture
• Tool-based memory infrastructure in place but underutilized
• Pre-LangGraph baseline
• Agent has tools but doesn't use them effectively
• Architecture stable, reasoning quality insufficient

---

# Development Log: 2025-12-26

## Major Architectural Changes

### 1. Multi-Agent System with Parallel Execution

**Replaced:** Single decision agent making all choices independently
**New Architecture:** Specialist agent system with coordinated decision-making

#### IssueAgent System
- **Purpose**: Each IssueAgent focuses on solving ONE specific strategic puzzle/obstacle
- **Data Source**: Memory database (persistent strategic issues flagged from previous turns)
- **Spawning**: Up to 5 IssueAgents spawn per turn (top 5 by importance score)
- **Research Phase**: Each agent uses history tools to gather context about their specific issue
- **Proposal Phase**: Each generates structured proposal (action, reason, confidence 1-100)
- **Lifespan**: Persistent - same issue tracked across multiple turns until resolved
- **Files**: `tools/agent_graph/issue_agent.py`

#### ExplorerAgent System
- **Purpose**: Single agent per turn advocating for systematic map exploration
- **Data Source**: Live map state from MapperToolkit (not persistent memory)
- **Spawning**: ONE ExplorerAgent spawns per turn IF unexplored cardinal directions exist
- **Direction Selection**: Intelligent priority (mentioned in description > cardinals > diagonals > up/down)
- **Confidence Calculation**:
  - Base score (45-75) from number of unexplored directions
  - +20 bonus if chosen direction mentioned in location description
  - Capped at 95 (never 100% certain)
- **Research Phase**: Uses mapper tools to understand known geography
- **Proposal Phase**: Proposes best unexplored direction with rationale
- **Lifespan**: Ephemeral - recreated each turn based on current location
- **Files**: `tools/agent_graph/explorer_agent.py`

#### Parallel Agent Execution
- **Implementation**: Direct threading with `threading.Thread`
- **Why Threading**: Initial attempts with `asyncio.run()` failed - cannot call from within existing event loop (game's async `play()` method)
- **Execution**: All agents (IssueAgents + ExplorerAgent) run in parallel, each calling LangChain tools independently
- **Synchronization**: `thread.join()` waits for all agents to complete before proceeding
- **Error Handling**: Each agent has try/catch with fallback proposals if research fails
- **Files**: `tools/agent_graph/decision_graph.py` (spawn_agents_node)

### 2. Decision Agent as Evaluator (Not Independent Decider)

**Critical Paradigm Shift**: Decision Agent no longer makes independent decisions. It now **evaluates and chooses** from specialist agent proposals.

#### New Decision Agent Responsibilities
1. **Evaluate Proposals**: Receive proposals from IssueAgents + ExplorerAgent
2. **Calculate Expected Value**:
   - IssueAgent EV = (importance/1000) × (confidence/100) × 100
   - ExplorerAgent EV = (unexplored_count/10) × (confidence/100) × 50
3. **Apply Heuristics**:
   - High-value puzzles first (importance 800+ AND confidence 80+ = top priority)
   - Avoid loops (reject proposals matching recent failures from research context)
   - Exploration when stuck (same location 3+ turns with no progress)
   - Consensus signal (multiple agents suggest same action)
4. **Choose Best Action**: Select proposal with highest EV unless heuristics override
5. **Identify New Issues**: Watch game response for NEW strategic puzzles to track (this was the bug fixed later)

#### Decision Agent Prompt Architecture
- **System Prompt**: Defines role as judge/evaluator, explains specialist agents, lists decision criteria
- **Human Prompt**: Presents formatted agent proposals with EV calculations, game state, research context
- **Input Format**:
  ```
  IssueAgent #1: [Importance: 800/1000, Confidence: 85/100, EV: 68.0]
    Issue: Locked grating blocks path east
    Proposed Action: UNLOCK GRATING WITH KEY
    Reason: Research shows we have brass key, likely fits this lock

  ExplorerAgent: [Confidence: 75/100, EV: 37.5]
    Best Direction: NORTH
    Proposed Action: GO NORTH
    Reason: Unexplored cardinal direction, 5 total unexplored
  ```
- **Output**: AdventurerResponse with `command` (chosen from proposals), `reason` (which agent chosen and WHY), `remember`, `rememberImportance`, `item`, `moved`
- **Files**: `adventurer/prompt_library.py` (get_decision_agent_evaluation_prompt, get_decision_agent_human_prompt)

### 3. MapperToolkit - Failed Direction Tracking

**Critical Bug Fixed**: Mapper only recorded successful movements, so ExplorerAgent would suggest blocked directions infinitely.

#### Previous Behavior
```python
if self.previous_location != current_location:
    # Only record if location changed (successful movement)
    record_movement(from_location, to_location, direction)
```

#### New Behavior
```python
if self.previous_location != current_location:
    # Successful movement
    record_movement(from_location, to_location, direction)
elif direction:
    # Location SAME but direction command issued = BLOCKED
    record_movement(from_location, "BLOCKED", direction)
```

#### How It Works
1. Player at "Clearing" tries "GO NORTH"
2. Game responds "You cannot go that way" or "The windows are boarded"
3. Location stays "Clearing" (previous_location == current_location)
4. Mapper detects: direction command issued but no movement occurred
5. Records: `Clearing --[NORTH]--> BLOCKED` in database
6. ExplorerAgent spawning: Calls `get_exits_from("Clearing")` which returns `[("NORTH", "BLOCKED")]`
7. NORTH added to `known_directions`, excluded from `unexplored_directions`
8. ExplorerAgent never suggests NORTH again

**Files**: `tools/mapping/mapper_state.py` (update_from_turn method)

### 4. LangSmith Trace Naming

**Problem**: All LangChain traces showed generic "RunnableSequence" names, making debugging impossible.

**Solution**: Added `.with_config(run_name="...")` to all LLM invocations:

- **IssueAgent Research**: `"IssueAgent Research: {issue_content[:60]}"`
- **IssueAgent Proposal**: `"IssueAgent Proposal: {issue_content[:60]}"`
- **ExplorerAgent Research**: `"ExplorerAgent Research: {direction} from {location}"`
- **ExplorerAgent Proposal**: `"ExplorerAgent Proposal: {direction} from {location}"`
- **Decision Agent**: `"Decision Agent"`
- **Summary Generation**: `"Summary Generation: Turn {N} @ {location}"`
- **Long-Running Summary**: `"Long-Running Summary: Turn {N} @ {location}"`

**Files**:
- `tools/agent_graph/issue_agent.py`
- `tools/agent_graph/explorer_agent.py`
- `tools/agent_graph/decision_graph.py`
- `tools/history/history_summarizer.py`

### 5. Display System Updates

#### Decision Agent Reasoning Display
**New Feature**: Game I/O panel now shows Decision Agent's reasoning BEFORE each command

**Format**:
```
[Clearing]
🤖 Decision: Chose IssueAgent #2 (importance 800, confidence 85, EV 68.0)
because solving the grating puzzle is critical for winning. Research shows
we have the key. ExplorerAgent suggested NORTH (confidence 75, EV 37.5)
but solving this puzzle takes priority.

> UNLOCK GRATING WITH KEY

The grating unlocks with a satisfying click...
```

**Implementation**:
- Modified `DisplayManager.add_turn()` to accept `reasoning` parameter
- Modified `DisplayManager._build_io_content()` to display reasoning in yellow before command
- Modified `GameSession.__play_turn()` to pass `player_response.reason` to display
- **Files**: `display_manager.py`, `game_session.py`

#### Single Responsibility Principle (SRP) Refactoring
**Violation**: GameSession had 33 lines of display formatting logic (lines 145-170)

**Fix**: Moved all formatting into DisplayManager
- **Old**: GameSession formatted agent strings, passed formatted strings to DisplayManager
- **New**: GameSession passes raw objects, DisplayManager handles all formatting

**New DisplayManager Methods**:
```python
def update_agents(self, issue_agents: list, explorer_agent):
    """Accepts raw agent objects, formats internally"""
    # Sorting by confidence
    # Type detection (IssueAgent vs ExplorerAgent)
    # String formatting with proposals, reasons, confidence

def update_map_from_transitions(self, transitions: list):
    """Accepts raw LocationTransition objects, formats internally"""
    # Formats: "Location1 --[DIRECTION]--> Location2 (T5)"
```

**GameSession Now**:
```python
# Clean - just passes raw data
display.update_agents(issue_agents, explorer_agent)
display.update_map_from_transitions(transitions)
```

**Files**: `display_manager.py` (new methods), `game_session.py` (simplified)

### 6. Database Session Persistence Fix

**Critical Bug**: History summaries were stale, showing old data from previous runs with same session ID.

**Root Cause Chain**:
1. `create_session()` used `INSERT OR IGNORE` - if session "v7" already existed, did nothing
2. Old turns 5-9 remained in database from previous run
3. New game started, added turns 1-4
4. Database now had BOTH old (5-9) and new (1-4) turns
5. `get_latest_summary()` used `ORDER BY turn_number DESC LIMIT 1` - returned turn 9's summary (OLD)
6. Summary said "pile of leaves" but leaves were already disturbed in old turn 2

**Fix**: Session resumption instead of deletion
```python
def create_session(self, session_id: str):
    # Don't delete old data - sessions should persist!
    # Just allow resumption
    cursor.execute("INSERT OR IGNORE INTO sessions ...")
```

**Additional Fix**: Turn number continuity
```python
# GameSession.__init__
last_turn = self.db.get_latest_turn_number(session_id)
self.turn_number = last_turn if last_turn is not None else 0
```

**New Method**:
```python
def get_latest_turn_number(self, session_id: str) -> Optional[int]:
    """Get the highest turn number for this session"""
    cursor.execute("SELECT MAX(turn_number) FROM turns WHERE session_id = ?")
    return result[0] if result and result[0] is not None else None
```

**How It Works Now**:
1. Run 1: Session "v7" plays turns 1-10, stops
2. Run 2: Session "v7" resumes at turn 10, continues with turn 11+
3. Summaries are continuous, no stale data

**Files**: `tools/database/db_manager.py`, `game_session.py`

### 7. Decision Agent "Identify New Issues" Bug Fix

**Critical Oversight**: When switching Decision Agent to evaluator role, forgot to tell it to identify new strategic issues.

**Broken Prompt** (lines 27-28):
```
Your role: Evaluate proposals from specialist agents and choose the best action.
```

**No mention** of watching game response for new puzzles/obstacles to track.

**Result**: Decision Agent never populated `remember` or `rememberImportance` fields. No new IssueAgents spawned. Same 4 agents every turn.

**Fix**: Added explicit dual responsibility and detailed guidance

**New Prompt**:
```
YOUR TWO RESPONSIBILITIES:
1. **CHOOSE ACTION**: Evaluate proposals from specialist agents and choose the best one
2. **IDENTIFY NEW ISSUES**: Watch the game response for new strategic puzzles/obstacles to track

IDENTIFYING NEW STRATEGIC ISSUES (for 'remember' field):
After choosing your action, read the Game Response carefully for NEW strategic issues:

What to track (use 'remember' field):
- NEW unsolved puzzles ("locked door", "troll demands payment", "need key")
- NEW obstacles blocking progress ("chasm too wide to cross", "darkness prevents movement")
- NEW opportunities to try ("found a ladder", "discovered a mechanism")

What NOT to track (leave 'remember' empty):
- Items/observations already in existing IssueAgent proposals
- General descriptions or flavor text
- Temporary states that will change
- Things you're handling this turn with your chosen action

Importance scoring (1-1000):
- 800-1000: Major puzzle blocking core progress (locked gate to treasury, troll blocking bridge)
- 500-700: Promising lead or secondary puzzle (mysterious mechanism, locked chest)
- 100-400: Minor puzzle or optional challenge (decorative statue, sealed jar)
```

**Files**: `adventurer/prompt_library.py`

## Technical Debt Addressed

### Agent Limit (Top 5 Only)
- Changed from spawning up to 100 IssueAgents to only top 5 by importance
- Prevents performance issues with too many parallel LLM calls
- **File**: `tools/agent_graph/decision_graph.py` (line 74: `limit=5`)

### Summary Model Field Correction
- Fixed long-running summary using `history_state.previous_command` instead of `latest_turn.player_command`
- Ensured both summaries use correct turn data
- **File**: `tools/history/history_summarizer.py` (line 133)

## Files Modified Today

### New Files Created
1. `tools/agent_graph/explorer_agent.py` - ExplorerAgent class with direction selection and confidence calculation
2. None others - all other work was modifications

### Existing Files Modified
1. `tools/agent_graph/decision_graph.py` - Spawn logic, parallel execution, proposal formatting, decision evaluation
2. `tools/agent_graph/issue_agent.py` - LangSmith naming
3. `tools/agent_graph/__init__.py` - Export ExplorerAgent
4. `tools/mapping/mapper_state.py` - Failed direction tracking
5. `adventurer/adventurer_service.py` - New decision chain using evaluation prompts, return ExplorerAgent
6. `adventurer/prompt_library.py` - Decision Agent evaluation prompts, new issue identification
7. `display_manager.py` - Reasoning display, agent formatting, map formatting (SRP fix)
8. `game_session.py` - Turn resumption, display updates (SRP fix), ExplorerAgent handling
9. `tools/database/db_manager.py` - get_latest_turn_number method
10. `tools/history/history_summarizer.py` - LangSmith naming, correct field usage

## Current System Architecture

```
TURN FLOW:
1. GameSession increments turn_number (resumes from database max)
2. ZorkService sends command to game, receives response
3. HistoryToolkit updates (adds turn, generates summaries)
4. MapperToolkit updates (records movement or blocked direction)
5. LangGraph Decision Flow:

   a. SPAWN_AGENTS_NODE:
      - Query memory DB → get top 5 issues by importance
      - Create 5 IssueAgents (one per issue)
      - Query mapper → get unexplored directions from current location
      - Create 1 ExplorerAgent (if unexplored directions exist)
      - ALL agents research in parallel (threading)
      - Each agent generates proposal (action, confidence, reason)

   b. RESEARCH_NODE:
      - Call history tools (get_full_summary, get_recent_turns)
      - Detect loops/failures in history
      - Return research context

   c. DECISION_NODE:
      - Format agent proposals with EV calculations
      - Pass to Decision Agent with game state + research context
      - Decision Agent evaluates proposals, chooses best
      - Decision Agent identifies new strategic issues in game response
      - Returns AdventurerResponse

   d. PERSIST_NODE:
      - If Decision Agent populated 'remember' field → save to memory DB
      - New issue becomes IssueAgent next turn

6. DisplayManager updates all panels:
   - Game I/O: Shows reasoning + command + game response
   - Summary: Recent + long-running summaries
   - Issues/Agents: Sorted list of all active agents with proposals
   - Map: All discovered location transitions

7. Return chosen command, loop continues
```

## Known Issues / Technical Debt Remaining

1. **Model**: Still using "gpt-5-nano-2025-08-07" which shouldn't exist but apparently does work
2. **Parallel Execution Verification**: Need to verify in LangSmith that agents are truly running concurrently
3. **Agent Limit Hardcoded**: Top 5 limit is hardcoded, should be configurable
4. **No Agent Cleanup**: IssueAgents persist forever, no mechanism to mark issues as "solved" and remove them
5. **ExplorerAgent Turn Number**: Currently hardcoded to 0 in spawning (line 137 decision_graph.py)

## Testing Needed

1. Verify agents run in parallel (check LangSmith traces for overlapping timestamps)
2. Verify Decision Agent identifies new issues (check logs for "MEMORY STORED")
3. Verify blocked directions prevent repeated suggestions
4. Verify session resumption works correctly across runs
5. Verify display shows reasoning for all decisions
6. Verify ExplorerAgent stops spawning when all directions explored from a location

---

# Development Log: 2025-01-18

## Major New Feature: Death Analyzer Tool

### Overview

Implemented a comprehensive death tracking and analysis system that uses LLMs to detect player deaths, analyze their causes, and generate recommendations for avoiding similar deaths in future playthroughs. This creates a persistent "lessons learned" database that accumulates wisdom across the session.

### Problem Statement

When the player dies in a text adventure game:
1. The game resets to a previous state (often losing progress)
2. The same death can happen repeatedly if the system doesn't learn from it
3. There was no mechanism to remember WHY deaths occurred or HOW to avoid them
4. Deaths were silent events with no analysis or tracking

### Solution: DeathAnalyzer

A new tool that runs after every turn and:
1. **Detects death** using LLM analysis of the game response
2. **Analyzes cause** with context from recent game history
3. **Generates recommendations** for avoiding this death in the future
4. **Persists to database** for permanent memory
5. **Displays in HTML reports** so all deaths are visible with their lessons

### Implementation Details

#### 1. Database Schema Addition (`tools/database/db_manager.py`)

**New Table: `deaths`**
```sql
CREATE TABLE IF NOT EXISTS deaths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    location TEXT,
    score INTEGER DEFAULT 0,
    moves INTEGER DEFAULT 0,
    cause_of_death TEXT NOT NULL,
    events_leading_to_death TEXT NOT NULL,
    recommendations TEXT NOT NULL,
    game_response TEXT,
    player_command TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
```

**New Index:**
```sql
CREATE INDEX IF NOT EXISTS idx_deaths_session
ON deaths(session_id, turn_number DESC)
```

**New Methods:**
- `add_death()` - Record a death with full analysis
- `get_all_deaths()` - Retrieve all deaths for a session as list of dicts
- `get_death_count()` - Get total number of deaths in session

#### 2. DeathAnalyzer Class (`tools/analysis/death_analyzer.py`)

**New File Created** with two classes:

**DeathAnalysis (Pydantic Model)**
```python
class DeathAnalysis(BaseModel):
    died: bool = Field(description="Whether the player died this turn")
    cause_of_death: str = Field(default="", description="What killed the player")
    events_leading_to_death: str = Field(default="", description="The sequence of events that led to death")
    recommendations: str = Field(default="", description="How to avoid this death in the future")
```

**DeathAnalyzer Class**

Core method: `analyze_turn()`
```python
def analyze_turn(
    self,
    turn_number: int,
    game_response: str,
    player_command: str,
    location: str,
    score: int,
    moves: int
) -> Optional[DeathAnalysis]:
```

**Two-Phase LLM Analysis:**

1. **Quick Detection Phase** (`_analyze_for_death`)
   - Uses cheap LLM with structured output
   - Checks for death indicators in game response:
     - "You have died"
     - "You are dead"
     - "Your adventure is over"
     - "You have been killed/slain"
     - Score resetting with death message
   - Returns immediately if no death detected (saves LLM calls)

2. **Full Analysis Phase** (`_analyze_death_with_context`)
   - Only runs if death detected
   - Gathers last 10 turns of history for context
   - Uses LLM to analyze:
     - **Cause**: What specifically killed the player
     - **Events**: The sequence of decisions that led here
     - **Recommendations**: Actionable advice for avoidance
   - Persists to database automatically

**LLM Prompts:**

Detection prompt focuses on identifying death markers:
```
Look for death indicators such as:
- "You have died"
- "You are dead"
- Score resetting to 0 with death message
- Game over messages
- Being eaten, drowned, crushed, etc.
```

Analysis prompt focuses on lessons learned:
```
Your job is to:
1. Identify the CAUSE of death - what specifically killed the player
2. Trace the EVENTS leading to death - what decisions or circumstances led to this outcome
3. Provide RECOMMENDATIONS - specific, actionable advice for avoiding this death in future playthroughs
```

#### 3. Integration into Game Loop (`game_session.py`)

**New Step 7c** added after big picture analysis:
```python
# Step 7c: Analyze for death (saved to database if death detected)
death_analyzer = DeathAnalyzer(
    self.history_toolkit,
    self.session_id,
    self.db
)
death_analysis = death_analyzer.analyze_turn(
    turn_number=self.turn_number,
    game_response=zork_response.Response,
    player_command=input_text,
    location=zork_response.LocationName,
    score=zork_response.Score,
    moves=zork_response.Moves
)
# Get all deaths for the report
all_deaths = death_analyzer.get_all_deaths()
```

**Report Writer Call Updated:**
```python
report_writer.write_turn_report(
    ...
    all_deaths=all_deaths  # New parameter
)
```

#### 4. HTML Report Display (`tools/reporting/turn_report_writer.py`)

**New CSS Styles:**
- `.death-log-section` - Red gradient background with skull emoji
- `.death-count-badge` - Shows total death count
- `.death-entry` - Individual death card with red left border
- `.death-header` - Turn number and location
- `.death-cause` - Red background for cause of death
- `.death-events` - Events leading to death
- `.death-recommendations` - Green background for avoidance tips
- `.no-deaths` - Celebratory message if no deaths yet

**New HTML Section** (after Agent Analysis, before Decision Agent):
```html
<section class="section">
    <h2 class="section-title">Death Log</h2>
    <div class="death-log-section">
        <div class="death-log-title">
            <span>💀</span> Deaths This Session
            <span class="death-count-badge">N deaths</span>
        </div>
        <!-- For each death: -->
        <div class="death-entry">
            <div class="death-header">
                <span class="death-turn">Turn N</span>
                <span class="death-location">📍 Location</span>
            </div>
            <div class="death-cause">
                <div class="death-cause-label">Cause of Death</div>
                <div>Eaten by a grue</div>
            </div>
            <div class="death-events">
                <div class="death-events-label">Events Leading to Death</div>
                <div>Entered dark area without lamp...</div>
            </div>
            <div class="death-recommendations">
                <div class="death-recommendations-label">How to Avoid</div>
                <div>Always carry a light source when...</div>
            </div>
        </div>
    </div>
</section>
```

#### 5. Module Exports (`tools/analysis/__init__.py`)

Updated to export new classes:
```python
from .death_analyzer import DeathAnalyzer, DeathAnalysis

__all__ = [
    'BigPictureAnalyzer',
    'DeathAnalyzer',
    'DeathAnalysis',
    ...
]
```

### Files Created

1. `tools/analysis/death_analyzer.py` - DeathAnalyzer class and DeathAnalysis model

### Files Modified

1. `tools/database/db_manager.py` - Added deaths table, index, and methods
2. `tools/analysis/__init__.py` - Added exports for DeathAnalyzer
3. `game_session.py` - Integrated death analyzer into turn loop
4. `tools/reporting/turn_report_writer.py` - Added CSS and HTML for death log section

---

## Bug Fix: None Agents in Parallel Execution

### Problem

When running the game, an exception was thrown:
```
AttributeError: 'NoneType' object has no attribute 'issue_content'
```

The error occurred in `decision_graph.py` line 239 when trying to process agents in parallel.

### Root Cause

The `loop_detection_agent` was explicitly disabled (set to `None` on line 172):
```python
loop_detection_agent = None  # DISABLED - not useful in practice
```

But it was being added to the agent list unconditionally:
```python
all_agents = issue_agents + ([explorer_agent] if explorer_agent else []) + [loop_detection_agent, interaction_agent]
```

This caused `None` to be in the list, which then failed when the parallel execution tried to access `agent.issue_content`.

### Fix

Changed the agent list construction to filter out None values:
```python
# Old (broken):
all_agents = issue_agents + ([explorer_agent] if explorer_agent else []) + [loop_detection_agent, interaction_agent]

# New (fixed):
all_agents = [a for a in issue_agents + [explorer_agent, loop_detection_agent, interaction_agent] if a is not None]
```

### File Modified

- `tools/agent_graph/decision_graph.py` (line 262-263)

---

## Bug Fix: Wrong Reasoning Displayed in HTML Reports

### Problem

The HTML turn report was showing mismatched command/reasoning pairs:
- **Command shown**: "look"
- **Reasoning shown**: "Chose InteractionAgent proposing OPEN BULKHEAD..."

The system was executing the correct command, but the report displayed the reasoning for the NEXT command instead of the current one.

### Root Cause

In `game_session.py`, the reasoning was being captured AFTER it was updated:

```python
# Step 4: Display uses self.pending_reasoning (CORRECT - old value)
display.add_turn(..., reasoning=self.pending_reasoning, ...)

# Then we UPDATE pending_reasoning to new value
self.pending_reasoning = player_response.reason

# ... later ...

# Step 8: Report uses self.pending_reasoning (WRONG - now has new value!)
report_writer.write_turn_report(..., player_reasoning=self.pending_reasoning, ...)
```

The display got the correct (old) reasoning, but the report writer got the wrong (new) reasoning because it was called after the update.

### Fix

Captured the reasoning before updating it:
```python
# Capture reasoning BEFORE updating so we can use it in the report later
reasoning_for_this_command = self.pending_reasoning

display.add_turn(..., reasoning=reasoning_for_this_command, ...)

self.pending_reasoning = player_response.reason  # Update for next turn

# ... later ...

# Report now uses the captured value
report_writer.write_turn_report(..., player_reasoning=reasoning_for_this_command, ...)
```

### File Modified

- `game_session.py` (lines 146-147, 216)

---

## New Game Backend: Escape Room

### Addition

Added a new game backend configuration for local testing:

```python
"escaperoom": {
    "base_url": "http://localhost:5000",
    "endpoint": "/EscapeRoom",
    "name": "Escape Room",
    "objective": "Escape the room",
    "target_score": 100
}
```

### Configuration

Set as active game:
```python
ACTIVE_GAME = "escaperoom"  # Options: "zork", "planetfall", or "escaperoom"
```

Session ID updated to "E1" for the new game.

### File Modified

- `config.py` (lines 35-41, 45)

---

## Summary of All Changes Today

### New Features
1. **Death Analyzer Tool** - LLM-powered death detection, analysis, and recommendation system
2. **Death Log in HTML Reports** - Visual display of all deaths with causes and avoidance tips
3. **Escape Room Game Backend** - New localhost game for testing

### Bug Fixes
1. **None Agent Crash** - Fixed parallel execution crashing on disabled agents
2. **Wrong Reasoning Display** - Fixed HTML reports showing next turn's reasoning instead of current

### Files Created
- `tools/analysis/death_analyzer.py`

### Files Modified
- `tools/database/db_manager.py`
- `tools/analysis/__init__.py`
- `game_session.py`
- `tools/reporting/turn_report_writer.py`
- `tools/agent_graph/decision_graph.py`
- `config.py`

---

## Architecture After Today's Changes

```
TURN FLOW (Updated):

1. GameSession increments turn_number
2. ZorkService sends command, receives response
3. HistoryToolkit updates (adds turn, generates summaries)
4. MapperToolkit updates (records movement or blocked)
5. LangGraph Decision Flow (spawn → research → decide → persist)
6. DisplayManager updates all panels
7. BigPictureAnalyzer generates strategic analysis
8. **NEW: DeathAnalyzer checks for death**
   - If death detected:
     a. LLM analyzes cause with recent history context
     b. LLM generates avoidance recommendations
     c. Death record persisted to database
   - All deaths retrieved for report
9. TurnReportWriter generates HTML with:
   - Game state, context, strategic overview
   - Agent analysis
   - **NEW: Death log section showing all session deaths**
   - Decision agent reasoning and tool calls
10. Session index updated
11. Return chosen command, loop continues
```

## Testing Needed for New Features

1. **Death Detection** - Verify LLM correctly identifies death markers in various game responses
2. **Death Analysis Quality** - Verify LLM provides useful cause/events/recommendations
3. **Database Persistence** - Verify deaths survive session restarts
4. **HTML Display** - Verify death log renders correctly with styling
5. **Performance** - Verify death analysis doesn't add significant latency (cheap LLM should be fast)
6. **Edge Cases** - Test with:
   - Multiple deaths in same session
   - Deaths at different locations
   - Deaths from different causes (combat, environment, puzzles)

---

# Development Log: 2026-08-21

## Research Reframing (Thesis Direction)

The project's original motivation — external memory scaffolding for small-context models — is obsolete: 1M-context models fit an entire game run in context. The standing research question is now:

> **Can the multi-agent deliberation architecture (advocacy agents + arbiter + deterministic toolkits) let much LESS powerful models (qwen2.5:14b local) solve the game?**

Architecture as a substitute for model capability. This is a candidate **Master's in AI thesis**. Key implication: correctness of the deterministic scaffolding (map, inventory, memory) is a *precondition* for interpretable experiments — a weak model cannot compensate for corrupted state, so any negative result from a buggy platform is uninterpretable.

## Full Code Audit → 27 GitHub Issues

Ran a nine-agent parallel audit (one reviewer per agent, plus the mapper and the orchestration layer). Every finding was verified by **executing the actual code** — real parsers driven with real game prose, throwaway SQLite DBs, installed library sources inspected. Unverified claims were dropped.

**Result: [27 GitHub issues](https://github.com/arsindelve/PlayZork/issues), all labeled `bug`.**

### Correctness (#1–#22) — headline findings per component

- **Systemic (#1–#5)**: no exception handling anywhere in the turn path (one malformed LLM response ends the session); the dedup call has no retry at all; `TURN_BUDGET_SECONDS` (600s) < one call's retry envelope (~1530s), so retries are dead code and mid-graph timeouts can half-commit memory state; `tool_choice="any"` is silently ignored by ChatOllama; research LLM bound with 8 tools but Observer can execute only 2.
- **IssueAgent (#6–#7)**: instructs the model to call `get_current_inventory()` — a tool that does not exist — so inventory renders as "empty" every turn on every code path (the high-confidence "I have the item" branch is unreachable); `"Unknown"`-location memories are pathfound to a nonexistent node → forced confidence 0 forever.
- **Explorer/Mapper (#8–#15)**: substring alias matching fabricates "mentioned" directions ("NE" in CORNER → Attic proposes NORTHEAST at confidence 95); `N` never matches `NORTH` in the explored-check; `MOVE RUG` records direction "E" (the verb MOVE contains E); BLOCKED records are permanent (UNIQUE constraint silently rejects corrections — the trap door route is unreachable forever after one early failed DOWN); death/respawn writes a fabricated edge BFS then routes through; lookups are case-sensitive end to end; CLIMB/ENTER unrecognized → unreachable orphan rooms.
- **InteractionAgent (#16–#18)**: deterministic regex parser bypasses the LLM with `TAKE NOTHING` (fires on Zork's standard EXAMINE reply), `OPEN YOU`, negation-blind `PRESS BUTTON` at confidence 80–90; unknown inventory presented as authoritatively empty; no history → re-proposes the same failed action every turn.
- **IssueClosedAgent (#19–#20)**: returned IDs never validated — the prompt's own worked example `[5, 12]` can close real unrelated issues, invisibly and irreversibly (dedup matches closed rows); closer ranks by undecayed importance while the spawner uses decayed, so actively-worked issues can become un-closable.
- **Inventory (#21)**: cache and DB permanently diverge (phantom items feed every prompt for the rest of the session).
- **LoopDetectionAgent (#22)**: disabled safely today; confirmed broken five ways if re-enabled (nonexistent tool name → can only ever propose INVENTORY; parser can't read the real tool format; fabricated score data; proposes the very command it flags; turn parser corrupts location-less turns).

### Orchestration (#23–#27) — with measured timings

Turn 1 of today's smoke run (`logs/game_codex-smoke-20260821.log`) took **7m25s with only 2 subagents and zero IssueAgents**:

| Phase | Time | Share |
|---|---|---|
| History summaries (serial, blocking, expensive model) | 86s | 19% |
| Spawn agents (4 LLM calls) | 146s | 33% |
| Research node (redundant) | 27s | 6% |
| Decision | 74s | 17% |
| Post-decision bookkeeping (observe + persist) | 112s | 25% |

Findings: 25% of the turn happens **after the command is chosen** (#23); summaries block every turn's start and grow with game length — 113s by turn 2 (#24); ~40% of turn time is LLM round-trips for deterministic data fetches (#25); LangGraph is wired as a pure linear chain — its fan-out concurrency is unused, `research` waits behind spawn despite zero data dependency (#26); "parallel" fan-out achieved only ~1.9× against the single Ollama server, and the sync retry's timeout abandons-but-doesn't-cancel in-flight requests, piling retries onto the queue that caused the timeout (#27).

## Plan of Attack (PLAN.md, committed)

Milestone-ordered roadmap: **M1** runs survive (exception handling, budget coherence) → **M2** five-minute fixes (#6, #20, #9, #19, #7) → **M3** trustworthy world state (map upsert first — permanence makes every other mapper bug permanent) → **M4** turn engine restructure (deterministic TurnContext closes #4/#5/#17 for free; graph ends at `decide`) → **M5** honest proposals → **M6** LoopDetectionAgent decision point → thesis experiment protocol (seeded runs: single-shot-with-full-history baseline vs. architecture, plus ablations; score@turns and score@wall-clock). Estimated 8–12 focused days to a runnable experiment.

## Documentation Overhaul (commit 8a56668, pushed)

- **CLAUDE.md**: rewritten for the actual multi-agent architecture (decision graph, agent roster, toolkits, `.env`-driven config, session resumption) — the old text still described the pre-multi-agent single-decider design.
- **README**: Current Status now describes `main` (v0.1-arxiv noted as the archived pre-multi-agent Zenodo baseline); accurate project structure tree; "manual issue definition" limitation replaced with the true one (automated lifecycle, unvalidated quality); Qwen 2.5 acknowledgments.
- **PLAN.md**: new.

## Uncommitted Working-Tree Changes (same day, separate session)

- Config moved to `.env`-driven (`PLAYZORK_GAME` / `PLAYZORK_SESSION_ID` / `PLAYZORK_LLM_PROVIDER`); `GAME_NAME`/`GAME_OBJECTIVE` now derived from the active backend (fixes the prompts-say-Planetfall-while-playing-EscapeRoom inconsistency)
- `run_playzork.py` PyCharm entry point
- Research agent now binds history + mapper + inventory + analysis tools; research node execution map widened to match
- Persist node analyzes the executed `player_command` instead of the next decision's command, with new test `tests/test_decision_graph.py`
- `.env.example` rewritten

## Testing

All tests pass: 48 pathfinder + 1 new persist-node test.


# Development Log: 2026-08-22 — Milestones 2 & 3, and the M3 checkpoint run

## Shipped

- **Milestone 2** (`7c7266b`) — #6, #7, #9, #19, #20 closed.
- **Milestone 3** (`f03856f`) — #8, #10, #11, #12, #13, #14, #21 closed; #15 half-fixed and updated.
- **Inventory analyzer fix** (`17a4354`) — found by the checkpoint run, three turns in.
- Tests: **145 → 463**. New issues filed: #28–#32.

Both milestones were investigated by parallel read-only subagents (one per issue) and applied serially, because the issues overlap heavily on the same files.

## The checkpoint run

Session `m3-checkpoint-20260822`, 15 turns, Zork I, qwen2.5:14b via local Ollama.
Stopped at 15 rather than 30 — see "the deadlock" below; the information saturated.

### What the run validated

| Issue | Evidence from live play |
|---|---|
| #1 | **15 turns, 0 exceptions, 0 turn failures.** |
| #3 | `Attempt 1/3 (timeout: 180s)` — the coherent retry envelope. |
| #8 | West Of House logged `['WEST']`. The 2026-08-21 log for the same room: `['WEST', 'SOUTHEAST']` (fabricated from HOU**SE**). |
| #10 | `GO SOUTH` → canonical `SOUTH`. `MOVE RUG` wrote **no** map row (the old code extracted `E` from the verb). |
| #12 | Probed a real grue/troll death: died in Cellar, respawned Forest, **no edge recorded**, `previous_location` correctly kept as the respawn room. |
| #13 | `West **Of** House` and `North **of** House` both appeared in one map — the exact inconsistency that made this corrective rather than defensive. |
| #14 | Two raw-command edges recorded: `Forest Path --[CLIMB TREE]--> Up A Tree`, `Behind House --[ENTER WINDOW]--> Kitchen`. Both were previously dropped, orphaning those rooms. |
| #24 (opt 1) | `Both summaries generated in 13.9s (concurrent)`. |

The mapper produced 5 edges, all correct Zork geography, and **zero** BLOCKED rows.

Probing also confirmed #10's deliberate *exclusion*: "The door is boarded and you can't remove the boards." contains "can't" but is object-specific — a temporary puzzle state — and correctly wrote nothing, while "You cannot go that way." correctly wrote BLOCKED. A naive deny-list would have burned the boarded door permanently.

### Finding 1 — a regression the tests could not reach

Turn 3, `TAKE LEAFLET` → `"Taken."` → **`Items removed: ['leaflet']`**. Successfully picking something up emptied it from inventory. The model's reasoning named the cause: *"it was already in their inventory, so it was removed... despite the action being a take command."*

Two chained defects: a pre-existing one (`"reveals a leaflet"` recorded as acquired) put the leaflet in inventory early, and then the M3 rule *"never list an item in items_added if it already appears in CURRENTLY CARRYING"* — which said only what **not** to do — left the model to invent an alternative, and it chose removal.

Fixed in `17a4354` by stating the no-op explicitly. **This is the checkpoint's whole justification**: 463 unit tests and seven parallel investigations missed it, because it exists only in the interaction between a prompt rule and one model's reasoning. Three turns of real play found it.

### Finding 2 — turn latency grows superlinearly, and summaries drive it

```
turn   1    3    5    7    9   11   13
secs  79   83  148  187  194  208  225
summ  13.9 26.6 52.4 67.0 65.9  ...
```

Turn time **more than doubled in nine turns**, and roughly half the growth is the summary phase (13.9s → 65.9s, ~34% of a turn), on a fixed model and machine with the concurrency fix already applied.

This is #24 option 3 confirmed and quantified. Option 1 halved a constant; it did nothing to the growth *rate*. Extrapolating ~6s/turn, the summary phase alone reaches minutes by turn 50 — and Zork is hundreds of turns.

**This changes M4's ordering.** #24 options 2 and 3 should precede #25's TurnContext: moving summarization off the critical path removes it from turn latency entirely, which is a larger and cheaper win than shaving research round-trips. For a thesis where latency is experimental throughput and the protocol needs N seeded runs per condition, this is the binding constraint on whether the experiment is runnable.

### Finding 3 — the deadlock

```
turn 11  EXAMINE PILE OF LEAVES  -> "There is nothing special about the pile of leaves."
turn 12  NORTH                   -> "The forest becomes impenetrable to the north."
turn 13  EXAMINE PILE OF LEAVES  -> (repeat)
turn 14  NORTH                   -> (repeat)
turn 15  EXAMINE PILE OF LEAVES  -> (repeat)
```

Two known-refused actions, alternating. Three gaps compound:

1. **The map never learned the wall.** "The forest becomes impenetrable to the north." is a genuine topological refusal outside #10's allow-list, so no BLOCKED row was written and the explorer still counts NORTH as unexplored. The under-detect bias is working as designed (no false wall) but the explorer learns nothing. Filed as a follow-up.
2. **Nothing suppresses repetition** — both negative results were in recent history. This is #18 (M5), now with a real trace.
3. **The loop detector is disabled** (#22).

The #22 "keep disabled?" decision now has data: the capability is needed. #18 looks like the better vehicle — deterministic, no LLM call, and it addresses the cause rather than detecting the symptom.

Encouraging counter-observation: before deadlocking, the agent broke a wander loop on its own to issue `EXAMINE PILE OF LEAVES`, acting on a strategic issue the ObserverAgent had stored eight turns earlier. The memory → IssueAgent → arbiter path closes end to end.

## Caveats

- The 78s turn-1 time is **not** comparable to the 2026-08-21 baseline of 445s. The summary phase alone went 86s → 13.9s, a 6x change where parallelising two calls can buy at most 2x, so the rest is machine load or history length. M4's before/after must be measured on one machine in one session.
- This session's *inventory* data is poisoned from turn 3 by Finding 1 and should not be used. The fix was verified directly against the model instead.

## Next

1. #24 options 2 and 3 (summaries off the critical path, bound the growth) — promoted ahead of #25 on Finding 2.
2. #30 — consume `LastMovementDirection` / `exits`, before TurnContext is designed around command parsing.
3. Then the rest of M4, and M5's #18 with Finding 3 as its justification.


## Correction to Finding 2 (same day, before implementing)

Finding 2 above attributed the latency growth to the summaries' own text
growing. **That attribution is wrong**, and the fix it implied (#24 option 3,
"bound the long-running summary") would have targeted a non-cause. Correcting
it here rather than editing it away, because the reasoning matters.

Measured from the same run:

| | turn 1 | turn 14 |
|---|---|---|
| LLM calls per turn | 16 | 16 (**constant**) |
| Total LLM call-seconds | 98s | 359s |
| Wall clock | 79s | 228s |
| Effective parallelism | 1.24x | 1.57x |
| Stored summary text | 374-832 chars | never larger |

The call **count** never changes; each call gets slower (~10s -> ~28s mean).
The summaries are tiny — 832 characters at their largest, a few seconds of
generation — so their own size cannot explain a 4-9x slowdown.

What actually grows is the *history-shaped* prompt content: `get_recent_turns`
output, the map, and tool results threaded into research and decision prompts.
The dominant single contributor is `BigPictureAnalyzer`, which pulled
`get_recent_turns(50)` into a prompt it runs on the **expensive** model **every
turn**. It was also invisible: it and `DeathAnalyzer` called `.invoke()`
directly with no retry wrapper and no log markers, so 3 of the 16 calls per
turn could not be measured at all.

The summaries merely *looked* worst because they run first in a turn and
therefore collide head-on with the previous turn's background analyzers. That
also explains the plateau: turn times flatten near 225s once the 50-turn
window fills.

**Lesson for the thesis measurement discipline:** the first plausible story
fit the shape of the data and was still wrong. Every LLM call must be
instrumented, or analysis silently reasons about two-thirds of the work.

## Acted on

- `BigPictureAnalyzer` and `DeathAnalyzer` now go through `invoke_with_retry`
  — log markers, timeout and retry, and no more measurement blind spot.
- The big-picture window is bounded and configurable:
  `BIG_PICTURE_HISTORY_TURNS` (default 20, was a hardcoded 50).
- **#24 option 2 implemented.** `HistoryToolkit.record_turn()` stays synchronous
  on the critical path (this turn's agents research against it via
  `get_recent_turns`); `refresh_summaries()` is dispatched as a tracked
  background task, coalesced under a lock so overlapping turns cannot let an
  older summary overwrite a newer one.
- **#24 option 3 deliberately NOT implemented** — it targets a non-cause.
- `OLLAMA_NUM_PARALLEL` is confirmed unset on this machine, exactly as #27
  warned. Effective parallelism measured at ~1.6x against 16 calls per turn.

The strategic conclusion also flips back: with a constant 16 calls per turn
against a saturated single server, **reducing the number of calls (#25) is the
dominant lever**, not reordering them. Moving summaries off the critical path
still shortens the turn, but on a saturated server it relocates work rather
than removing it.


# Development Log: 2026-08-24 — M4 continued, and the finding that reframes it

## Shipped

- **#28** (`9b3f621`) — prompt JSON examples. Narrower than filed: only the plain-string path was over-escaped; the template-rendered prompts were already correct. Introduced by my own #19 rewrite.
- **#23 + #26** (`5d97771`, follow-up) — the graph fans out; nodes return partial state; `build_context` hoisted; `close_issues`/`observe` async; `turn_number` moved from a mutable side-channel dict into graph state.
- Tests: 496 → 506.

## THE FINDING: this machine's Ollama has no useful parallelism at all

Benchmarked directly, warm model, identical prompts:

| concurrency | wall | per-call | throughput | speedup |
|---|---|---|---|---|
| 1 | 3.8s | 3.8s | 0.26/s | 1.00x |
| 2 | 7.7s | 5.7s | 0.26/s | 0.99x |
| 4 | 15.3s | 9.6s | 0.26/s | **0.99x** |

At ~1700-token prompts (realistic for this system). The small-prompt run is identical: 0.79 / 0.78 / 0.78 / 0.76 req/s at 1 / 2 / 4 / 8.

**Throughput is flat. Concurrency buys nothing; it only divides the same tokens/sec across more requests, so per-call latency scales linearly.**

### What this invalidates

- **#27 understates it.** The audit measured "~1.9x, not 4x" and read that as *degraded* parallelism. It is not degraded — it is **absent**. The 1.9x was measurement overlap of queued requests, not concurrent service.
- **#23/#26 cannot produce a throughput win on this hardware**, and the measurement says so: turns 1–2 improved (49→29s, 87→68s) but turns 3–4 got *worse* (62→72s, 68→78s), because the diamond raises peak in-flight requests from 6 to 8 and mean per-call latency from 24.8s to **31.1s**. Net over four turns: 7%, which is inside the noise.
- **#25's win was real precisely because it removed calls** (12 → 6), not because it reordered them.

### What actually governs turn time

Turn time = total tokens processed per turn ÷ ~fixed tokens/sec. Nothing else. The levers are:

1. **Fewer LLM calls** — #25 did this, and it is the only lever that has produced a measured win.
2. **Shorter prompts** — the `BIG_PICTURE_HISTORY_TURNS` bound, and anything else that trims history-shaped content.
3. **Fewer output tokens** — untouched so far; structured outputs are small but proposals and reasoning are not.
4. **Different serving** — vLLM with real continuous batching, or a smaller model.

**For the thesis this is a hardware constraint that belongs in the methodology, not a bug.** `score@wall-clock` on this rig is a measure of total tokens per turn. Any architecture comparison must either report token counts alongside wall-clock, or run on serving that actually batches — otherwise the multi-agent arm is penalised for token volume in a way that says nothing about the architecture.

## Was the #23/#26 refactor still worth doing?

Yes, but for correctness rather than speed, and the write-up should say so:

- The graph now expresses the real dependency structure, which is what #26 asked for and what "I wanted to use LangGraph for fun and learning" was supposed to deliver. It was previously a straight line that would have behaved identically as sequential awaits.
- `close_issues` and `observe` are async, so a timeout **cancels** the request instead of leaking a thread and retrying alongside it (#27's amplifier).
- Nodes return only their own keys, which makes the disjointness of their writes explicit and checkable.
- It positions the system to benefit immediately if serving is ever changed.

## Four for four

Every substantial refactor this week has had a defect that the unit suite missed and a live run caught:

1. M2 — inventory analyzer inverted a TAKE into a removal (prompt × model interaction).
2. #25 — deleted import left the Observer silently disabled; #1's containment hid it.
3. #23/#26 — separate fan-in edges are not a join; `persist` ran twice per turn.
4. #26 follow-up — `create_decision_graph()` signature change crashed on startup; **no test constructed the real `AdventurerService`**, so 504 tests passed against code that could not boot.

Each of the four was invisible-by-design rather than loud. A wiring test that constructs the real service now exists (#4 above), and the standing rule stands: **run the thing before believing the suite**.


## 2026-08-24, later — M5 correctness, experiment scaffolding, GPU prep

Context: the PC with the 5070 Ti arrives at the weekend, so this block is deliberately all hardware-independent work.

**Shipped:** #28, #23+#26, #30 follow-through, #18, #33, #16, #15, plus token accounting, a vLLM provider and the experiment's control arm. Tests 496 → 587. Issues 23 → 27 closed, 6 open.

### The deadlock is fixed (#18, #33)

The M3 checkpoint ended with the agent alternating two already-refused commands for five turns. Promoted ahead of remaining latency work on the grounds that faster hardware only deadlocks faster, and a run that flatlines at turn 11 cannot produce `score@turns` data.

`TurnContext` now tracks commands already shown to do nothing *in this room*, and the arbiter sees any repeat at **EV 0.0** with the prior response quoted. The demotion is in code, not prompt text — the #21 inventory bug established that a 14B model handed a bare prohibition invents its own way around it.

#33 widened the refusal allow-list from *observed* backend phrasings. A drafted `impassable + mountains` pattern was written and deleted: Zork's Forest room *description* reads "revealing impassible mountains" — scenery on a **successful** move. It escaped the first check only on a spelling coincidence.

### Two fixes that turned out to be "ask the server, not the model"

- **#16** — the InteractionAgent's regex emitted `TAKE NOTHING` at confidence 90 on Zork's standard EXAMINE reply, and short-circuited the LLM on exactly those turns. The real fix was #30's `ActionsAvailableFromLocation`: the game reports, per object, which commands it will accept. The parser is now a hint that can never bypass the LLM.
- **#15** — same-named rooms merged into one node. #30's `exits` array discriminates them: verified live, `Forest [2,0,1]` and `Forest #2 [3,2,1]` are now separate nodes.

Both follow the pattern established by #30's inventory work: stop inferring what the backend already knows.

### Instrumentation and scaffolding

- **Per-turn token accounting** → `turn_tokens`. Metered in `llm_utils`, the choke point both retry helpers pass through. Totals are a **floor**: structured-output calls carry no usage metadata and are skipped rather than estimated.
- **vLLM provider** — `PLAYZORK_LLM_PROVIDER=vllm`, so Saturday is a config change.
- **Control arm** — `PLAYZORK_CONDITION=single_shot`. One inference, full context, same model tier. Live smoke run played sensibly at ~20s/turn vs the treatment's ~70s.

### Baseline for the hardware move

Apple M5, 24GB, qwen2.5:14b: **generation 14 tok/s, prefill 237 tok/s**. A 3.5k-token call takes 14.4s, of which **10.4s is prefill** — which is why a GPU should help disproportionately here, prefill being compute-bound and parallel where generation is bandwidth-bound and sequential.

### Five for five

Every substantial change this week shipped with a green suite and had a defect found by a live run:

1. M2 — inventory analyzer inverted a TAKE into a removal (prompt × model interaction).
2. #25 — a deleted import left the ObserverAgent silently disabled.
3. #23/#26 — separate fan-in edges are not a join; `persist` ran twice per turn.
4. #26 follow-up — a signature change crashed on startup; no test built the real service.
5. #16 — a greedy quantifier turned "a small mailbox here." into `TAKE HERE`.

Four of the five were *silent*, which is the direct cost of #1's error containment: a broken component logs and continues. The standing rule is now in CLAUDE.md — prefer tests that **execute** over tests that inspect source, and run a session before believing the suite.

### Docs

CLAUDE.md was materially wrong (it still documented the deleted research node and the pre-#30 inventory path) and has been rewritten: new graph topology, the experiment setup, and an **Invariants** section recording the false-negative-beats-false-positive rule that now governs the whole world model. README gained a Measurement Notes section stating plainly that wall-clock on this rig measures token volume.

---

# Development Log: 2026-08-24 (later) — Planetfall, and six defects Zork could not show

Playing a second game was worth more than any amount of further work on the
first. Zork exercises one shape of failure; Planetfall exercised six others,
and one of them had been silently costing every Zork run as well.

## The run that made it obvious

`pf-20260824`. Planetfall opens on a doomed ship: escape via the pod or die
with it. The game hands over the escape route in the **starting room**:

```json
"actionsAvailableFromLocation": {"escape pod bulkhead": ["open bulkhead", "close bulkhead"]}
```

Every stage handled it correctly. The parser read it; it reached the prompt
verbatim; the ObserverAgent stored it at importance 900 with the right
location; the InteractionAgent proposed `OPEN escape pod bulkhead` at
confidence 70 on turn 2.

Then the arbiter chose `GO UP` and walked away from it.

```
InteractionAgent: [Confidence: 70/100]             OPEN escape pod bulkhead
ExplorerAgent:    [Confidence: 95/100, EV: 47.5]   GO UP                     ← chosen
```

**The InteractionAgent was never given an expected value**, while the decision
prompt ranks by expected value. The only agent that proposes object
interactions was structurally unrankable. And the explorer's EV scales with
`unexplored/10`, which is maximal at the start of every game — precisely when
the pod mattered. This had been quietly shaping Zork too, where ExplorerAgent
won 16 of 26 contested turns.

## Six fixes

1. **InteractionAgent EV.** Evidence-weighted rather than tuned to win this
   case: a command the *backend* listed for an object present here is
   guaranteed to parse (#30/#16) — strictly stronger evidence than an
   advertised exit, which is sometimes refused — so it scores 100, mirroring
   the explorer's +3 for a game-confirmed exit. A model-invented interaction
   scores 50, so at confidence 70 it gets 35 and **still loses** to
   exploration's 47.5. A test pins that, because otherwise this would just be
   handing the agent a blanket win.
2. **Its repeat/undo multiplier was discarded** — `note, _ = repeat_note(...)`
   kept the warning line and threw away the zeroing, making #18 a prohibition
   in prose for this agent. The #21 lesson, re-learned.
3. **The game clock was parsed and never read.** `Time` has been on the
   response model since #30; the only matches in the codebase were `timeout`
   and `setTimeout`. On a timed objective the agents could not see the deadline
   they were being judged against. Zork returns 0 by probe, so it renders only
   where a clock exists.
4. **Planetfall's objective was "Complete the mission"** — interpolated into
   every prompt as the arbiter's only statement of what it plays for.
5. **IssueAgent now walks to its issue.** It returned `nothing` at confidence 0
   for the 900-importance pod two rooms away, with the route already in its
   prompt. Confidence 70 describes the reliability of the *action* — one step
   along a BFS path over edges we recorded — while worth is priced by the
   importance term: 900 gives EV 63 and outranks exploration, a decayed 300
   gives 21 and does not.
6. **Movement was three commands and had no inverse.** `GO WEST` / `WEST` / `W`
   were distinct keys, and `_INVERSES` held no directions while `inverse_of`
   required an object after the verb. So nothing detected that EAST reverses
   WEST. In Zork `frontier3-20260824` the agent reached **Behind House — the
   room containing the window into the house** — and oscillated
   `GO WEST → EAST → GO WEST` off it, with **zero** suppressions in 16 turns.
   Planetfall reproduced it vertically.

   Ship directions (port/starboard/fore/aft) were added as **aliases**, not new
   canonical directions: EAST and STARBOARD are one passage, and since the
   explorer's EV scales with the unexplored count, aliasing them separately
   would inflate its EV and let it re-walk a passage under the other name.
   Mapping verified by live probe: starboard→E, port→W, fore→N, aft→S.

## Verified in play, not just in the suite

| | turn 2 decision |
|---|---|
| control `pf-20260824` | `GO UP` |
| fixed `pf2-20260824` | **`OPEN escape pod bulkhead`** |

Tests 661 → 736.

## What this cost, and the lesson

Three consecutive Zork runs were spent chasing an empty frontier that turned
out to be two *different* wiring bugs, and both times the available conclusion
was "milestone 5b doesn't help" — which would have charged a wiring bug to the
architecture. The rule that keeps earning its place: **run the thing, and read
the raw rows rather than guessing.** The third attempt found it in the DB in
about a minute.

A second rule was added after nearly reporting a 27% token saving that was
entirely a turn-count artifact (26-turn control vs 11-turn treatment; matched
on turns 1–10 the honest figure is +4.1%): **never compare a per-turn mean
across runs of unequal length.**
