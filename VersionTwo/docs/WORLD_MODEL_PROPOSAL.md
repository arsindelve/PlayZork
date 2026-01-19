# Structured World Model Proposal

## The Problem

Currently, knowledge in the IF solver lives in **text**:
- LLM-generated summaries (lossy, unstructured)
- Flat issue list (strings with importance scores)
- Map graph (locations + transitions)
- Inventory (list of item names)

Text summaries are lossy. The summarizer might drop details. The decision LLM might miss things. We're playing telephone with ourselves.

### What's Missing

1. **Object-Location Memory** - "There's a brass key in the Kitchen" - we don't track items we've SEEN but not taken.

2. **World State Beyond Location** - Is the door open or closed? Is the lamp on or off? The world changes, but we don't model it.

3. **Failed Action Memory** - "I tried UNLOCK DOOR and it said 'You don't have the key'" - we're not recording what DIDN'T work.

4. **Puzzle Dependency Graph** - Issues are independent, but puzzles chain: `get_treasure → open_case → get_key → kill_thief → get_sword`

5. **Hypothesis Tracking** - "I think the key opens the grate" with confidence levels, updated based on outcomes.

---

## What IF Games Actually Are

Interactive fiction is a **state machine**:

```
WorldState {
  locations: Map<LocationId, Location>
  objects: Map<ObjectId, Object>
  player: { location, inventory }
}

Action + WorldState → NewWorldState + Response
```

The game has perfect knowledge of this. We're trying to reconstruct it from prose descriptions.

---

## Structured World Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WORLD MODEL                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LOCATIONS                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Kitchen     │───▶│ Living Room │───▶│ Attic       │     │
│  │             │    │             │    │             │     │
│  │ contains:   │    │ contains:   │    │ contains:   │     │
│  │ - table     │    │ - lamp      │    │ - rope      │     │
│  │ - key ✓taken│    │ - rug       │    │             │     │
│  │             │    │             │    │ state:      │     │
│  │ state:      │    │ state:      │    │ - dark      │     │
│  │ - lit       │    │ - lit       │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
│  OBJECTS                                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ brass_key                                          │    │
│  │   location: INVENTORY                              │    │
│  │   properties: takeable                             │    │
│  │   hypotheses:                                      │    │
│  │     - unlocks:grate (confidence: 0.8)              │    │
│  │     - unlocks:door (confidence: 0.3, DISPROVEN)    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  FACTS (certain)                    HYPOTHESES (uncertain)  │
│  - key is in inventory              - key unlocks grate     │
│  - grate is locked                  - treasure in case      │
│  - lamp is off                      - thief has egg         │
│  - troll is at bridge               - sword kills troll     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Three Layers

### Layer 1: Extraction (after each turn)

Parse game response into structured updates:

```python
class TurnExtraction(BaseModel):
    objects_mentioned: List[ObjectMention]  # [{name, properties, location}]
    state_changes: List[StateChange]        # [{object, property, old, new}]
    action_result: ActionResult             # success/failure/partial + reason
    new_information: List[str]              # facts learned

class ObjectMention(BaseModel):
    name: str
    location: str  # where it was seen
    properties: List[str]  # takeable, openable, readable, etc.
    state: Dict[str, str]  # open/closed, on/off, etc.

class StateChange(BaseModel):
    object: str
    property: str
    old_value: Optional[str]
    new_value: str
```

Example LLM extraction prompt:
```
Game response: "You take the brass key. The door is locked."

Extract:
- objects_mentioned: [
    {name: "brass key", location: "inventory", properties: ["takeable"]},
    {name: "door", location: "current", state: {locked: "true"}}
  ]
- action_result: {action: "TAKE KEY", result: "success"}
- state_changes: [{object: "brass key", property: "location", new: "inventory"}]
```

### Layer 2: Storage

```python
@dataclass
class GameObject:
    id: str
    name: str
    aliases: List[str]  # "key", "brass key", "small key"
    location: str  # location_id or "inventory" or "unknown"
    properties: Dict[str, Any]  # takeable, openable, readable, etc.
    state: Dict[str, Any]       # open/closed, on/off, locked/unlocked
    last_seen_turn: int
    might_solve: List[str]  # patterns like "locked*", "dark*"

@dataclass
class Hypothesis:
    id: str
    subject: str      # "brass_key"
    predicate: str    # "unlocks"
    object: str       # "grate"
    confidence: float # 0.0-1.0
    evidence: List[str]  # why we believe this
    tested: bool
    disproven: bool

@dataclass
class FailedAction:
    turn: int
    command: str
    error_message: str
    location: str
    inference: str  # "door requires key we don't have"
```

### Layer 3: Query Tools

```python
@tool
def get_objects_at_location(location: str) -> List[str]:
    """What objects have we seen at this location?"""

@tool
def where_is_object(object_name: str) -> str:
    """Where did we last see this object? Returns location, 'inventory', or 'unknown'."""

@tool
def get_object_state(object_name: str) -> Dict:
    """What do we know about this object's current state?"""

@tool
def get_objects_with_property(property: str) -> List[str]:
    """Find all objects with a given property (e.g., 'key-like', 'light-source')."""

@tool
def get_hypotheses_about(object_name: str) -> List[Hypothesis]:
    """What hypotheses involve this object?"""

@tool
def get_failed_actions_at(location: str) -> List[FailedAction]:
    """What have we tried here that didn't work?"""

@tool
def record_hypothesis(subject: str, predicate: str, obj: str, confidence: float, evidence: str):
    """Record a new hypothesis about the world."""

@tool
def update_hypothesis(hypothesis_id: str, tested: bool, disproven: bool, new_confidence: float):
    """Update a hypothesis based on test results."""
```

---

## The Agent Query Problem

**Problem**: Agents need to know what questions to ask. An IssueAgent tracking "locked door" might not think to query "are there any keys?"

**Solution**: Don't make them ask - inject relevance automatically.

### Approach 1: Auto-Enrich Context

Before any IssueAgent runs, automatically retrieve and inject relevant objects/facts:

```python
def enrich_issue_context(issue: str, world_model: WorldModel) -> str:
    """Automatically find relevant objects/facts for an issue."""

    relevant_objects = []
    relevant_hypotheses = []

    # Keyword/semantic matching
    if any(word in issue.lower() for word in ["locked", "lock", "unlock", "door", "gate", "chest"]):
        relevant_objects.extend(world_model.get_objects_with_property("key-like"))

    if any(word in issue.lower() for word in ["dark", "darkness", "can't see", "pitch black"]):
        relevant_objects.extend(world_model.get_objects_with_property("light-source"))

    if any(word in issue.lower() for word in ["troll", "thief", "monster", "combat", "fight"]):
        relevant_objects.extend(world_model.get_objects_with_property("weapon"))

    # Get hypotheses that mention any word in the issue
    relevant_hypotheses = world_model.get_hypotheses_matching(issue)

    return f"""
POTENTIALLY RELEVANT (auto-retrieved):
Objects: {format_objects(relevant_objects)}
Hypotheses: {format_hypotheses(relevant_hypotheses)}
Failed attempts: {world_model.get_failed_actions_matching(issue)}
"""
```

### Approach 2: Bidirectional Tagging

When objects are discovered, tag them with what issues they might solve:

```python
# Object tagging rules
OBJECT_TAGS = {
    "key": ["locked*", "unlock*", "door*", "gate*", "chest*", "case*"],
    "lamp": ["dark*", "light*", "can't see*", "pitch black*"],
    "lantern": ["dark*", "light*", "can't see*", "pitch black*"],
    "sword": ["troll*", "thief*", "combat*", "fight*", "kill*"],
    "rope": ["climb*", "descend*", "tie*", "canyon*", "pit*"],
}

def tag_object(obj: GameObject) -> None:
    for keyword, tags in OBJECT_TAGS.items():
        if keyword in obj.name.lower():
            obj.might_solve.extend(tags)
```

### Approach 3: Objects Become Issues

When significant objects are found, automatically create issues for them:

```python
SIGNIFICANT_OBJECT_TYPES = ["key", "lamp", "sword", "rope", "bottle", "book"]

def maybe_create_object_issue(obj: GameObject, turn: int) -> Optional[Issue]:
    for obj_type in SIGNIFICANT_OBJECT_TYPES:
        if obj_type in obj.name.lower():
            return Issue(
                content=f"{obj.name} found at {obj.location} - determine use",
                importance=600,
                turn_number=turn,
                location=obj.location
            )
    return None
```

Now you have:
- "Locked door" issue → IssueAgent looks for keys
- "brass key found - determine use" issue → IssueAgent looks for locked things

They find each other through research.

---

## Backward Chaining with World Model

With structured data, you can reason backward from goals:

```
GOAL: Get treasure (worth 100 points)
  └─ treasure is in jeweled_case
      └─ case is LOCKED
          └─ need key that unlocks case
              └─ hypothesis: brass_key unlocks case (0.6 confidence)
                  └─ brass_key is in INVENTORY ✓
                      └─ ACTION: UNLOCK CASE WITH KEY
```

```python
def backward_chain(goal: str, world_model: WorldModel, depth: int = 0) -> List[str]:
    """Find action sequence to achieve goal via backward chaining."""

    if depth > 5:
        return ["MAX_DEPTH_REACHED"]

    # Check if goal is already satisfied
    if world_model.is_satisfied(goal):
        return ["ALREADY_DONE"]

    # Find what blocks the goal
    blockers = world_model.get_blockers(goal)

    for blocker in blockers:
        # Find what might resolve the blocker
        resolvers = world_model.get_hypotheses_for_blocker(blocker)

        for resolver in resolvers:
            # Check if we can do the resolving action
            if world_model.can_do(resolver.action):
                return [resolver.action]
            else:
                # Recurse: what do we need to do the resolving action?
                sub_plan = backward_chain(resolver.precondition, world_model, depth + 1)
                if sub_plan:
                    return sub_plan + [resolver.action]

    return []  # No plan found
```

---

## Implementation Phases

### Phase 1: Object-Location Tracking (~200 lines)
- Add `WorldModel` class with object storage
- Add extraction LLM call after each turn
- Add basic query tools
- Inject object context into IssueAgent prompts

### Phase 2: State Tracking (~100 lines)
- Extend objects with state dict
- Track state changes (open/closed, on/off, etc.)
- Add state query tools

### Phase 3: Failed Action Memory (~100 lines)
- Record failed commands with error messages
- Query "what have we tried at this location"
- Prevent re-trying known failures

### Phase 4: Hypothesis System (~200 lines)
- Hypothesis CRUD
- Confidence updates based on outcomes
- Auto-generate hypotheses from object + issue matching

### Phase 5: Dependency Reasoning (~300 lines)
- Backward chaining from goals
- Puzzle dependency graph
- Plan generation

---

## Risks and Mitigations

### Risk: Extraction Errors
IF prose is literary and ambiguous:
- "A glint of metal catches your eye" = metal object exists

**Mitigation**:
- Confidence levels on extracted facts
- Reconciliation when observations don't match model
- Fall back to text summaries for ambiguous cases

### Risk: State Divergence
If we miss an update, our model diverges from game reality.

**Mitigation**:
- Periodic LOOK commands to re-sync location state
- INVENTORY commands to verify inventory
- Mark old observations as "stale" after N turns

### Risk: Over-Engineering
Could model everything but what actually helps?

**Mitigation**:
- Start with Phase 1 only
- Measure: does it improve puzzle-solving?
- Only add complexity that demonstrably helps

---

## Key Insight

**Don't rely on agents to ask the right questions.**

The world model should be **proactive context**, not a passive database:
1. Auto-enrich agent prompts with relevant objects/facts
2. Significant items spawn their own issues
3. Agents evaluate and act on information, not discover it

This is the difference between:
- "Here's your issue. Use tools to research." (agent might miss things)
- "Here's your issue. Here's everything relevant." (agent can focus on decisions)

---

## Files to Create

```
VersionTwo/
  tools/
    world_model/
      __init__.py
      world_state.py      # WorldModel class, GameObject, Hypothesis
      extractor.py        # LLM extraction after each turn
      world_tools.py      # LangChain @tool definitions
      enrichment.py       # Auto-enrich issue context
      tagging.py          # Object → issue pattern matching
```

---

## Next Steps

1. Prototype Phase 1 (object-location tracking)
2. Test on Escape Room game (simpler)
3. Measure: do agents make better decisions?
4. Iterate based on failure modes
