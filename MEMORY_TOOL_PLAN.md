# Memory Tool Implementation Plan

## Overview

Build an in-memory memory system that automatically captures what the LLM flags as important via the `remember` field, then provides intelligent query tools for retrieving relevant memories during decision-making.

## Core Concept

**Automatic Capture:**
- Every turn, if `AdventurerResponse.remember` is not empty, store it as a Memory
- Track: content, importance, turn_number, location, timestamp

**Intelligent Retrieval:**
- 3 query tools that use a cheap LLM to find and format relevant memories
- LLM does semantic matching (not just keyword search)
- Returns context-aware summaries

## Architecture

### Memory State Structure

```
tools/
├── history/                    # EXISTING
│   ├── __init__.py
│   ├── history_state.py
│   ├── history_tools.py
│   └── history_summarizer.py
└── memory/                     # NEW
    ├── __init__.py            # MemoryToolkit facade
    ├── memory_state.py        # Memory storage & state
    ├── memory_tools.py        # @tool decorated query functions
    └── memory_retriever.py    # LLM-powered semantic search
```

## Data Models

### Memory Model

```python
# tools/memory/memory_state.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Memory(BaseModel):
    """A single memory flagged by the LLM as important"""

    turn_number: int           # When this was remembered
    content: str               # What to remember (from AdventurerResponse.remember)
    importance: int            # 1-1000 priority (from AdventurerResponse.rememberImportance)
    location: str              # Where we were when we learned this
    score: int                 # Game score at the time
    moves: int                 # Total moves at the time
    timestamp: str             # ISO timestamp

    def __str__(self) -> str:
        return f"[Turn {self.turn_number} @ {self.location}] {self.content} (importance: {self.importance})"

class MemoryState:
    """In-memory storage for all memories"""

    def __init__(self):
        self.memories: List[Memory] = []
        self._next_id: int = 1

    def add_memory(
        self,
        content: str,
        importance: int,
        turn_number: int,
        location: str,
        score: int,
        moves: int
    ) -> Memory:
        """
        Add a new memory.

        Args:
            content: What to remember
            importance: Priority 1-1000
            turn_number: Current turn
            location: Current location name
            score: Current score
            moves: Current move count

        Returns:
            The created Memory
        """
        # Don't add empty or duplicate memories
        if not content or not content.strip():
            return None

        # Check for near-duplicates (fuzzy match on content)
        for existing in self.memories:
            if self._is_similar(content, existing.content):
                # Update importance if new one is higher
                if importance > existing.importance:
                    existing.importance = importance
                return None  # Don't add duplicate

        memory = Memory(
            turn_number=turn_number,
            content=content.strip(),
            importance=importance,
            location=location,
            score=score,
            moves=moves,
            timestamp=datetime.now().isoformat()
        )

        self.memories.append(memory)

        # Keep sorted by importance (highest first)
        self.memories.sort(key=lambda m: m.importance, reverse=True)

        return memory

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Simple similarity check to avoid duplicates"""
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Exact match
        if t1 == t2:
            return True

        # One contains the other (80% threshold)
        if len(t1) > len(t2):
            return t2 in t1 and len(t2) / len(t1) > 0.8
        else:
            return t1 in t2 and len(t1) / len(t2) > 0.8

    def get_top_memories(self, limit: int = 10) -> List[Memory]:
        """Get the N most important memories"""
        return self.memories[:limit]

    def get_memories_by_location(self, location: str) -> List[Memory]:
        """Get all memories associated with a specific location"""
        return [m for m in self.memories if m.location.lower() == location.lower()]

    def get_all_memories(self) -> List[Memory]:
        """Get all memories (sorted by importance)"""
        return self.memories

    def get_memory_count(self) -> int:
        """Total number of memories stored"""
        return len(self.memories)
```

## Memory Retriever (LLM-Powered Search)

```python
# tools/memory/memory_retriever.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from .memory_state import Memory

class MemoryRetriever:
    """Uses a cheap LLM to intelligently search and summarize memories"""

    def __init__(self, llm: ChatOpenAI):
        """
        Args:
            llm: A cheap LLM for semantic search (e.g., GPT-3.5-turbo)
        """
        self.llm = llm

    def query_memories(
        self,
        query: str,
        memories: List[Memory],
        max_results: int = 5
    ) -> str:
        """
        Semantic search: find memories relevant to a query.

        Args:
            query: Natural language question (e.g., "Where did I see a key?")
            memories: List of all memories to search
            max_results: Max memories to include in response

        Returns:
            Formatted string with relevant memories and answer
        """
        if not memories:
            return "No memories recorded yet."

        # Format all memories for the LLM
        memories_text = "\n".join([
            f"{i+1}. [Turn {m.turn_number} @ {m.location}, importance={m.importance}] {m.content}"
            for i, m in enumerate(memories)
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are helping someone play Zork by searching their memory notes.

            You will be given:
            1. A list of memories (things they flagged as important)
            2. A question they're asking

            Your job:
            - Find the most relevant memories that answer the question
            - Return ONLY the relevant memory numbers and a brief answer
            - If no memories are relevant, say "No relevant memories found."

            Format:
            Relevant memories: #3, #7, #12
            Answer: [1-2 sentence answer based on those memories]
            """),
            ("human", """Memories:
{memories}

Question: {query}

Which memories are relevant and what's the answer?""")
        ])

        response = self.llm.invoke(
            prompt.format_messages(memories=memories_text, query=query)
        )

        return response.content

    def summarize_location_memories(
        self,
        location: str,
        memories: List[Memory]
    ) -> str:
        """
        Summarize all memories about a specific location.

        Args:
            location: Location name (e.g., "West Of House")
            memories: Memories filtered to this location

        Returns:
            Summary of what we learned at this location
        """
        if not memories:
            return f"No memories recorded for {location}."

        memories_text = "\n".join([
            f"- Turn {m.turn_number}: {m.content} (importance: {m.importance})"
            for m in memories
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are summarizing memories about a specific location in Zork. Be concise."),
            ("human", """Location: {location}

Memories from this location:
{memories}

Provide a 2-3 sentence summary of what we learned here.""")
        ])

        response = self.llm.invoke(
            prompt.format_messages(location=location, memories=memories_text)
        )

        return response.content

    def get_top_insights(
        self,
        memories: List[Memory],
        limit: int = 10
    ) -> str:
        """
        Format the top N most important memories.

        Args:
            memories: Top memories (already sorted)
            limit: How many to include

        Returns:
            Formatted string of top memories
        """
        if not memories:
            return "No memories recorded yet."

        result = f"Top {len(memories[:limit])} Most Important Memories:\n\n"

        for i, m in enumerate(memories[:limit], 1):
            result += f"{i}. [Turn {m.turn_number} @ {m.location}] {m.content}\n"
            result += f"   Importance: {m.importance}/1000\n\n"

        return result.strip()
```

## Memory Tools (LangChain @tool decorators)

```python
# tools/memory/memory_tools.py
from langchain_core.tools import tool
from typing import Optional
from .memory_state import MemoryState
from .memory_retriever import MemoryRetriever

# Module-level state (initialized by toolkit)
_memory_state: Optional[MemoryState] = None
_memory_retriever: Optional[MemoryRetriever] = None

def initialize_memory_tools(memory_state: MemoryState, retriever: MemoryRetriever):
    """
    Initialize the module-level memory state and retriever.
    Called by MemoryToolkit on creation.
    """
    global _memory_state, _memory_retriever
    _memory_state = memory_state
    _memory_retriever = retriever

@tool
def get_top_memories(limit: int = 10) -> str:
    """
    Get the most important memories (up to 10).
    Use this to recall critical information flagged during the game.

    Args:
        limit: Number of top memories to retrieve (default: 10, max: 20)

    Returns:
        Formatted list of top memories sorted by importance
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    limit = max(1, min(limit, 20))  # Clamp to 1-20

    top_memories = _memory_state.get_top_memories(limit)

    if not top_memories:
        return "No memories recorded yet. As you play, important discoveries will be remembered here."

    return _memory_retriever.get_top_insights(top_memories, limit)

@tool
def query_memories(question: str) -> str:
    """
    Search memories for information relevant to a specific question.
    Uses semantic search to find the best matches.

    Args:
        question: Natural language question (e.g., "Where did I see a rusty key?" or "What puzzles are unsolved?")

    Returns:
        Relevant memories and an answer to the question

    Examples:
        query_memories("Where is the brass lantern?")
        query_memories("What items do I need for the puzzle in the attic?")
        query_memories("Which doors are still locked?")
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    if not question or not question.strip():
        return "Error: Please provide a specific question."

    all_memories = _memory_state.get_all_memories()

    if not all_memories:
        return "No memories to search. Nothing important has been flagged yet."

    return _memory_retriever.query_memories(question, all_memories, max_results=5)

@tool
def get_location_memories(location: str) -> str:
    """
    Get all memories associated with a specific location.
    Use this to recall what you learned at a particular place.

    Args:
        location: The location name (e.g., "West Of House", "Living Room")

    Returns:
        Summary of memories from that location

    Examples:
        get_location_memories("West Of House")
        get_location_memories("Kitchen")
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    if not location or not location.strip():
        return "Error: Please provide a location name."

    location_memories = _memory_state.get_memories_by_location(location)

    if not location_memories:
        return f"No memories recorded for '{location}'. Either you haven't been there or nothing important was noted."

    return _memory_retriever.summarize_location_memories(location, location_memories)

def get_memory_tools() -> list:
    """
    Get all memory tools for use by the agent.

    Returns:
        List of LangChain tools
    """
    return [
        get_top_memories,
        query_memories,
        get_location_memories
    ]
```

## Memory Toolkit (Facade)

```python
# tools/memory/__init__.py
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from typing import List
from .memory_state import MemoryState, Memory
from .memory_retriever import MemoryRetriever
from .memory_tools import initialize_memory_tools, get_memory_tools

class MemoryToolkit:
    """
    Facade for the memory system.
    Manages memory state and provides tools for the agent.
    """

    def __init__(self, retriever_llm: ChatOpenAI):
        """
        Initialize the memory toolkit.

        Args:
            retriever_llm: A cheap LLM for semantic search (e.g., GPT-3.5-turbo)
        """
        self.state = MemoryState()
        self.retriever = MemoryRetriever(retriever_llm)

        # Initialize module-level tools
        initialize_memory_tools(self.state, self.retriever)

    def add_memory(
        self,
        content: str,
        importance: int,
        turn_number: int,
        location: str,
        score: int,
        moves: int
    ) -> bool:
        """
        Add a new memory (called by game loop after each turn).

        Args:
            content: What to remember (from AdventurerResponse.remember)
            importance: Priority 1-1000 (from AdventurerResponse.rememberImportance)
            turn_number: Current turn number
            location: Current location
            score: Current game score
            moves: Current move count

        Returns:
            True if memory was added, False if skipped (empty or duplicate)
        """
        memory = self.state.add_memory(
            content=content,
            importance=importance,
            turn_number=turn_number,
            location=location,
            score=score,
            moves=moves
        )

        return memory is not None

    def get_tools(self) -> List[Tool]:
        """
        Get all memory tools for the agent to use.

        Returns:
            List of LangChain tools: get_top_memories, query_memories, get_location_memories
        """
        return get_memory_tools()

    def get_memory_count(self) -> int:
        """Get total number of stored memories"""
        return self.state.get_memory_count()

    def get_summary_stats(self) -> dict:
        """Get summary statistics about memories"""
        memories = self.state.get_all_memories()

        if not memories:
            return {
                "total_memories": 0,
                "avg_importance": 0,
                "locations_covered": 0
            }

        return {
            "total_memories": len(memories),
            "avg_importance": sum(m.importance for m in memories) / len(memories),
            "locations_covered": len(set(m.location for m in memories)),
            "top_location": max(set(m.location for m in memories), key=lambda loc: sum(1 for m in memories if m.location == loc))
        }

# Export main classes
__all__ = ['MemoryToolkit', 'Memory', 'MemoryState']
```

## Integration into Game Loop

### Update AdventurerService

```python
# adventurer/adventurer_service.py (MODIFIED)

from tools.memory import MemoryToolkit

class AdventurerService:

    def __init__(self, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit):
        """
        Args:
            history_toolkit: For accessing game history
            memory_toolkit: For accessing and storing memories (NEW)
        """
        self.history_toolkit = history_toolkit
        self.memory_toolkit = memory_toolkit  # NEW
        self.logger = GameLogger.get_instance()

        # Get ALL tools (history + memory)
        all_tools = self.history_toolkit.get_tools() + self.memory_toolkit.get_tools()

        # Create research agent (Phase 1) with all tools
        self.research_agent = self._create_research_agent(all_tools)
        self.decision_chain = self._create_decision_chain()

    def _create_research_agent(self, tools: list) -> Runnable:
        """
        Create the research agent with ALL available tools.

        Args:
            tools: Combined list of history + memory tools
        """
        llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0)

        # Bind all tools, force at least one call
        llm_with_tools = llm.bind_tools(tools, tool_choice="any")

        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_research_agent_prompt()),
            ("human", "{input}"),
        ])

        return prompt | llm_with_tools
```

### Update GameSession

```python
# game_session.py (MODIFIED)

from tools.memory import MemoryToolkit

class GameSession:

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turn_number = 0

        # Initialize logger
        self.logger = GameLogger.get_instance(session_id)

        # Initialize Zork service
        self.zork_service = ZorkService(session_id=session_id)

        # Create history toolkit with cheap LLM
        cheap_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.history_toolkit = HistoryToolkit(cheap_llm)

        # Create memory toolkit with same cheap LLM (NEW)
        self.memory_toolkit = MemoryToolkit(cheap_llm)

        # Pass BOTH toolkits to adventurer service
        self.adventurer_service = AdventurerService(
            self.history_toolkit,
            self.memory_toolkit
        )

    async def __play_turn(self, input_text: str, display: DisplayManager) -> str:
        try:
            self.turn_number += 1
            self.logger.log_turn_start(self.turn_number, input_text)

            # Step 1: Send input to Zork
            zork_response = await self.zork_service.play_turn(input_text=input_text)
            self.logger.log_game_response(zork_response.Response)

            # Step 2: Update history BEFORE decision
            self.history_toolkit.update_after_turn(
                game_response=zork_response.Response,
                player_command=input_text,
                location=zork_response.LocationName,
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            # Step 3: Make decision (research + decide)
            player_response = self.adventurer_service.handle_user_input(zork_response)

            # Step 4: Store memory if LLM flagged something (NEW)
            if player_response.remember and player_response.remember.strip():
                memory_added = self.memory_toolkit.add_memory(
                    content=player_response.remember,
                    importance=player_response.rememberImportance,
                    turn_number=self.turn_number,
                    location=zork_response.LocationName,
                    score=zork_response.Score,
                    moves=zork_response.Moves
                )

                if memory_added:
                    self.logger.logger.info(
                        f"MEMORY STORED: [{player_response.rememberImportance}/1000] {player_response.remember}"
                    )

            # Step 5: Update display
            display.add_turn(
                location=zork_response.LocationName,
                game_text=zork_response.Response,
                command=input_text,
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            # Step 6: Update display summary
            summary = self.history_toolkit.state.get_full_summary()
            display.update_summary(summary)
            self.logger.log_summary_update(summary)

            return player_response.command

        except Exception as e:
            self.logger.log_error(str(e))
            raise
```

## Updated Research Prompt

```python
# adventurer/prompt_library.py (MODIFIED)

@staticmethod
def get_research_agent_prompt():
    return """You are an assistant helping someone play Zork I.

    You have access to tools that let you query game history and memories:

    HISTORY TOOLS:
    - get_recent_turns(n): Get the last N turns of detailed game history
    - get_full_summary(): Get a complete narrative summary of all game history

    MEMORY TOOLS (NEW):
    - get_top_memories(limit): Get the most important flagged memories (up to 10)
    - query_memories(question): Search memories with a specific question
    - get_location_memories(location): Get memories about a specific place

    Current game state:
    - Score: {score}
    - Location: {locationName}
    - Moves: {moves}
    - Game Response: {game_response}

    CRITICAL INSTRUCTIONS:
    1. ALWAYS call get_full_summary() to understand the overall game state
    2. ALWAYS call get_recent_turns(5) to see recent actions
    3. Call get_top_memories() to recall important discoveries
    4. If you need specific information, use query_memories("your question")
    5. If revisiting a location, use get_location_memories(locationName)
    6. ANALYZE the data - identify loops, failures, and what to try next
    7. If stuck in a loop, FLAG THIS CLEARLY

    After calling tools, respond with:
    RESEARCH_COMPLETE: [Your analysis in 3-4 sentences]

    Example:
    RESEARCH_COMPLETE: The summary shows we already have the leaflet in inventory. Recent turns reveal a LOOP: we keep trying OPEN MAILBOX and TAKE MAILBOX which both fail. Top memories indicate there's a brass lantern somewhere we haven't explored yet. We need to STOP interacting with the mailbox and try moving NORTH or EAST to find new areas.
    """
```

## Logger Enhancement

```python
# game_logger.py (ADD METHOD)

def log_memory_stored(self, content: str, importance: int):
    """Log when a memory is stored"""
    self.logger.info(f"MEMORY STORED [importance: {importance}/1000]: {content}")
```

## Data Flow

```
Turn Execution:
1. Execute command → Get Zork response
2. Update history state
3. AdventurerService.handle_user_input()
   ├─ Research Phase:
   │  ├─ Calls get_full_summary() ✓
   │  ├─ Calls get_recent_turns(5) ✓
   │  ├─ Calls get_top_memories() ✓ (NEW)
   │  ├─ Calls query_memories("...") ✓ (if needed, NEW)
   │  └─ Returns analysis
   └─ Decision Phase:
      └─ Returns AdventurerResponse with .remember field
4. If .remember is not empty:
   └─ memory_toolkit.add_memory() → Store in MemoryState
5. Update display
```

## Example Memory Flow

**Turn 5:**
```
Command: OPEN MAILBOX
Response: "Opening the small mailbox reveals a leaflet."
Decision: TAKE LEAFLET
Remember: "Mailbox at West of House contains a leaflet"
RememberImportance: 300
→ Memory stored
```

**Turn 12:**
```
Location: Kitchen
Research Phase:
  - get_top_memories(10) called
  - Returns: "1. Mailbox at West of House contains a leaflet..."
  - query_memories("Where did I find items?") called
  - Returns: "Relevant memories: #1, #5. Answer: Found leaflet in mailbox at West of House, found brass lantern in Living Room."
Decision: Based on memories, go back to Living Room to get lantern
```

## Testing Strategy

### Manual Testing

1. **Run 10 turns** and check logs for:
   - "MEMORY STORED" entries when LLM flags something
   - Memories have importance scores
   - No duplicate memories

2. **At turn 15**, check if research phase calls:
   - `get_top_memories()`
   - `query_memories()` (if relevant)

3. **Revisit a location** and verify:
   - `get_location_memories()` is called
   - Returns relevant memories from that location

### Success Criteria

- ✅ Memories stored when `remember` field is populated
- ✅ No duplicate memories (fuzzy matching works)
- ✅ Top memories sorted by importance
- ✅ LLM calls memory tools during research
- ✅ Agent uses memories to make better decisions
- ✅ Agent recalls where items were found

## File Changes Summary

### New Files
1. `VersionTwo/tools/memory/__init__.py` - MemoryToolkit facade
2. `VersionTwo/tools/memory/memory_state.py` - Memory model and state
3. `VersionTwo/tools/memory/memory_tools.py` - @tool decorators
4. `VersionTwo/tools/memory/memory_retriever.py` - LLM-powered search

### Modified Files
1. `VersionTwo/adventurer/adventurer_service.py` - Accept memory_toolkit
2. `VersionTwo/game_session.py` - Create memory_toolkit, store memories
3. `VersionTwo/adventurer/prompt_library.py` - Update research prompt
4. `VersionTwo/game_logger.py` - Add log_memory_stored method

## Implementation Timeline

**Phase 1: Core Models (30 min)**
- Create `memory_state.py` with Memory and MemoryState classes
- Implement add_memory, get_top_memories, get_by_location

**Phase 2: Retriever (30 min)**
- Create `memory_retriever.py` with MemoryRetriever class
- Implement query_memories, summarize_location_memories

**Phase 3: Tools (30 min)**
- Create `memory_tools.py` with @tool decorators
- Implement get_top_memories, query_memories, get_location_memories

**Phase 4: Integration (30 min)**
- Create `__init__.py` with MemoryToolkit
- Update game_session.py and adventurer_service.py

**Phase 5: Testing (30 min)**
- Run game for 20 turns
- Verify memories stored
- Verify tools called

**Total: ~2.5 hours**

## Future: SQLite Migration

Once this works, we migrate to SQLite with:

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL,
    location TEXT NOT NULL,
    score INTEGER,
    moves INTEGER,
    timestamp TEXT NOT NULL
);

CREATE INDEX idx_session_importance ON memories(session_id, importance DESC);
CREATE INDEX idx_session_location ON memories(session_id, location);
```

**Same tool interface, different backend!**

## Questions Before Implementation

1. Should memories be deduplicated aggressively or allow similar ones?
2. Max number of memories to keep in memory (if we cap it before SQLite)?
3. Should the cheap LLM for retrieval be the same as history summarizer?

Ready to implement?
