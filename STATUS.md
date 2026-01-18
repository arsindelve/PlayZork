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
