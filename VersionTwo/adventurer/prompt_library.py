from config import GAME_NAME, GAME_OBJECTIVE, GAME_OBJECTIVE_SCORE


class PromptLibrary:
  """
  A static class to store and manage prompts.
  All LLM prompts used throughout the application are centralized here.
  """

  # ═══════════════════════════════════════════════════════════
  # HISTORY PROCESSOR PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_history_processor_human_prompt():
    return """
Previous summary:
{summary}

Latest interaction:
Player: {player_response}
Game: {game_response}

Update the summary with this new interaction. Output ONLY the updated summary.
"""

  @staticmethod
  def get_history_processor_system_prompt():
    return f"""You are a game state logger for {GAME_NAME}. Produce FACTUAL, TERSE summaries.

FORMAT - Use this structure:
```
CURRENT: [location] | Score: X | Inventory: [items]

RECENT ACTIONS:
- Turn N: [COMMAND] → [result in 5-10 words]
- Turn N-1: [COMMAND] → [result]
...

KEY FACTS:
- [Important discoveries, unlocked doors, solved puzzles]
- [Obstacles encountered, items found]
```

RULES:
1. FACTS ONLY - No creative writing, no "you feel", no "your hand brushes"
2. TERSE - Each action = one line, 10 words max
3. STATE CHANGES - When something changes (door unlocked, item taken), UPDATE the record
4. NO NARRATIVE - This is a log, not a story
5. CURRENT STATE - Always reflect what is TRUE NOW, not what was true before

BAD: "You approach the desk, feeling a sense of relief as your hand brushes against a flashlight"
GOOD: "Turn 5: TAKE FLASHLIGHT → Success. Now in inventory."

Output ONLY the summary. No meta-commentary."""

  # ═══════════════════════════════════════════════════════════
  # DECISION AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_decision_agent_evaluation_prompt():
    return f"""You are the Decision Agent in a {GAME_NAME}-playing AI system.

YOUR SINGLE RESPONSIBILITY: Choose the best action from specialist agent proposals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIALIST AGENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- IssueAgents: Each solves a specific tracked puzzle (importance = value for winning)
- InteractionAgent: Identifies and proposes interactions with objects in current location
- ExplorerAgent: Discovers new areas/items through systematic exploration
- LoopDetectionAgent: Detects stuck/oscillating patterns and proposes actions to break loops

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION CRITERIA (in priority order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. FILTER NON-ACTIONABLE PROPOSALS FIRST
   - IGNORE any agent with confidence = 0 (they have NO solution)
   - IGNORE any agent with proposed_action = "NOTHING" or "nothing" or empty
   - These agents determined they CANNOT help right now - skip them completely

1. HIGH-VALUE PUZZLES FIRST
   - IssueAgent with importance 800-1000 + confidence 80+ = TOP PRIORITY
   - Solving major puzzles = points = winning (goal: {GAME_OBJECTIVE})

2. AVOID LOOPS (check research context)
   - Never repeat actions that just failed
   - Reject proposals that match recent failures

3. EXPLORATION WHEN STUCK
   - Same location 3+ turns? → Prefer ExplorerAgent
   - No IssueAgent confidence >70? → Explore to find new puzzles

4. CONSENSUS SIGNAL
   - Multiple agents suggest same action? → Strong signal

5. CONFIDENCE LEVELS
   - 80-100: Strong, likely to succeed
   - 50-79: Worth trying if important
   - 1-49: Last resort only
   - 0: IGNORE COMPLETELY (see rule 0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED VALUE CALCULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- IssueAgent EV = (importance/1000) × (confidence/100) × 100
- ExplorerAgent EV = (unexplored_count/10) × (confidence/100) × 50
- Choose highest EV unless heuristics override

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL COMMAND RULES:
- NEVER use semicolons (;) in commands
- NEVER combine multiple commands - only ONE command at a time
- Use the SIMPLEST possible version of each command
- Examples: 'OPEN DOOR', 'TAKE LAMP', 'NORTH', 'EXAMINE KEY'
- NOT allowed: 'OPEN DOOR; TAKE KEY', 'OPEN SURVIVAL KIT; TAKE ROPE'

Output JSON with:
- command: The action to execute (from chosen agent proposal) - MUST be a single simple command
- reason: Which agent you chose and WHY (explain your decision)
- moved: Direction if movement command (or empty string)
"""

  @staticmethod
  def get_decision_agent_human_prompt():
    return f"""=== GAME STATE ===
Location: {{locationName}}
Score: {{score}} | Moves: {{moves}}
Game Response: {{game_response}}

=== RESEARCH CONTEXT ===
{{research_context}}

=== AGENT PROPOSALS ===
{{agent_proposals}}

=== YOUR TASK ===
Evaluate the proposals above and choose the best action.

Consider:
1. Expected Value: Which proposal has highest (importance × confidence)?
2. Are we stuck in a loop? (prefer exploration)
3. Any consensus among agents?
4. What does research context warn against?

Choose the best proposal and explain your reasoning clearly in the 'reason' field.

CRITICAL COMMAND RULES:
- NEVER use semicolons (;) in commands
- NEVER combine multiple commands - only ONE command at a time
- Use the SIMPLEST possible version of each command
- If an agent proposes a multi-command (e.g., 'OPEN KIT; TAKE ROPE'), choose ONLY the first part ('OPEN KIT')
- Examples of valid commands: 'OPEN DOOR', 'TAKE LAMP', 'NORTH', 'EXAMINE KEY'
- Examples of INVALID commands: 'OPEN DOOR; TAKE KEY', 'OPEN SURVIVAL KIT; TAKE ROPE'

Instructions: Provide a JSON output without backticks:

{{{{
    "command": "The command from the chosen proposal (or LOOK if uncertain) - MUST be a single simple command with NO semicolons",
    "reason": "Explain which agent's proposal you chose and WHY. Example: 'Chose IssueAgent #2 (importance 800, confidence 85, EV 68.0) because solving the grating puzzle is critical for winning. Research shows we have the key. ExplorerAgent suggested NORTH (confidence 75, EV 37.5) but solving this puzzle takes priority.'",
    "moved": "if you chose a movement command, list the direction you tried to go. Otherwise, leave this empty."
}}}}
"""

  # ═══════════════════════════════════════════════════════════
  # ADVENTURER PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_adventurer_prompt():
    return f"""
    You have played {{moves}} moves and have a score of {{score}}.

    You are currently in this location: {{locationName}}

    The game just responded: {{game_response}}

    === RESEARCH ANALYSIS ===
    {{research_context}}
    === END RESEARCH ANALYSIS ===

    CRITICAL DECISION-MAKING RULES:
    1. READ THE RESEARCH ANALYSIS ABOVE CAREFULLY - it tells you what to avoid and what to try
    2. If research says you're in a LOOP - BREAK OUT OF IT immediately by trying something completely different
    3. If the game says "already open", "already have", "securely anchored" - NEVER try that action again
    4. If you've been in the same location for 3+ turns without progress, TRY MOVING: NORTH, SOUTH, EAST, WEST, UP, DOWN
    5. Exploration is KEY - if stuck, try directional movement to find new areas
    6. Inventory management: Use INVENTORY to check what you have, DROP items if needed
    7. Every command must be NEW and PRODUCTIVE - no wasted turns

    CRITICAL COMMAND RULES:
    - NEVER use semicolons (;) in commands - they are FORBIDDEN
    - NEVER combine multiple commands - only ONE command at a time
    - Use the SIMPLEST possible version of each command
    - Examples: 'OPEN DOOR', 'TAKE LAMP', 'NORTH', 'EXAMINE KEY'
    - NOT allowed: 'OPEN DOOR; TAKE KEY', 'OPEN SURVIVAL KIT; TAKE ROPE', 'TAKE LAMP AND EXAMINE IT'

    Instructions: Provide a JSON output without backticks:

    {{{{
        "command": "Your NEXT command. Must be a SINGLE SIMPLE command with NO semicolons. Must NOT be a failed action from research analysis. If stuck, try directional movement.",
        "reason": "brief explanation based on research analysis and game state",
        "remember": "Record STRATEGIC ISSUES only: (1) UNSOLVED PUZZLES you discovered (e.g., 'locked grating blocks path east', 'need to cross the river somehow'), (2) OBVIOUS THINGS TO TRY that could unlock progress (e.g., 'get inside the white house', 'find a light source for dark areas'), (3) MAJOR OBSTACLES preventing advancement (e.g., 'troll demands payment to pass', 'cyclops is hostile and blocking path'). Do NOT record observations, items, or general notes. Memory is limited. Leave empty if no strategic issue discovered this turn.",
        "rememberImportance": "Score 1-1000 based on: How much will SOLVING/OVERCOMING this issue help us WIN the game ({GAME_OBJECTIVE})? Major blocking puzzles/obstacles = 800-1000. Promising leads = 500-700. Minor puzzles = 100-400. This score determines priority when making decisions.",
        "item": "any new, interesting items you have found in this location, along with their locations, which are not already mentioned above. For example 'there is a box and a light bulb in the maintenance room'. Omit if there is nothing here.",
        "moved": "if you attempted to move in a certain direction, list the direction you tried to go. Otherwise, leave this empty."
    }}}}
    """

  @staticmethod
  def get_system_prompt():
    return f"""
    You are playing {GAME_NAME} with the goal of winning the game by achieving {GAME_OBJECTIVE}.
    Play as if for the first time, without relying on any prior knowledge of the game.

    Objective: {GAME_OBJECTIVE}.

    CRITICAL COMMAND RULES:
    - NEVER use semicolons (;) - they are forbidden
    - NEVER combine multiple commands - one command at a time ONLY
    - Use the SIMPLEST possible version of each command
    - Examples: 'OPEN DOOR', 'TAKE LAMP', 'GO NORTH', 'READ BOOK'
    - NOT allowed: 'OPEN DOOR; GO NORTH', 'TAKE LAMP AND EXAMINE IT'

    Input Style: Use simple commands with one verb and one or two nouns, such as 'OPEN DOOR' or 'TURN SCREW WITH SCREWDRIVER.'
    Type "INVENTORY" to check items you're carrying and "SCORE" to view your current score (so you know if you're winning).
    Type "LOOK" to see where you are, and what is in the current location. Use this liberally.
    Progression: Use the recent interactions with the game provided to avoid repeating actions you've already completed and going around in circles.
    Focus on new, logical actions that progress the game and explore new opportunities or areas based on the current context and past interactions.
    """

  @staticmethod
  def get_research_agent_prompt():
    return f"""You are an assistant helping someone play {GAME_NAME}.

    You have access to tools that let you query game history:

    HISTORY TOOLS:
    - get_recent_turns(n): Get the last N turns of detailed game history
    - get_full_summary(): Get a complete narrative summary of all game history

    Current game state:
    - Score: {{score}}
    - Location: {{locationName}}
    - Moves: {{moves}}
    - Game Response: {{game_response}}

    CRITICAL INSTRUCTIONS:
    1. ALWAYS call get_full_summary() to understand the overall game state
    2. ALWAYS call get_recent_turns(5) to see recent actions
    3. ANALYZE the data - identify loops, failures, and what to try next
    4. If stuck in a loop, FLAG THIS CLEARLY
    5. Look for patterns of repeated failed commands
    6. Check if we've been in the same location multiple turns without progress

    After calling tools, respond with:
    RESEARCH_COMPLETE: [Your analysis in 3-4 sentences]

    Example:
    RESEARCH_COMPLETE: The summary shows we already have the leaflet in inventory. Recent turns reveal a LOOP: we keep trying OPEN MAILBOX and TAKE MAILBOX which both fail. We need to STOP interacting with the mailbox and try moving NORTH or EAST to find new areas.
    """

  # ═══════════════════════════════════════════════════════════
  # ISSUE AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_issue_agent_system_prompt():
    return f"""You are an IssueAgent tasked with solving ONE SPECIFIC puzzle/obstacle in {GAME_NAME}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL QUESTION YOU MUST ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Does the action I'm proposing DIRECTLY solve MY SPECIFIC issue?"

If YES → Give high confidence (70-100)
If NO → Give confidence 0 or very low (1-20)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR RESPONSIBILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You ONLY propose actions that solve YOUR specific issue.
You do NOT propose actions for other issues, exploration, or general progress.

EXAMPLES OF CORRECT BEHAVIOR:

Your Issue: "Locked door in Kitchen - need key"
Action: "UNLOCK DOOR WITH KEY" → Confidence 90 ✓ (solves YOUR issue)
Action: "GO NORTH" → Confidence 0 ✗ (doesn't solve YOUR issue)
Action: "EXAMINE ROOM" → Confidence 0 ✗ (doesn't solve YOUR issue)

Your Issue: "Troll blocking Bridge - need to defeat or bypass"
Action: "KILL TROLL WITH SWORD" → Confidence 85 ✓ (solves YOUR issue)
Action: "TAKE LAMP" → Confidence 0 ✗ (doesn't solve YOUR issue)
Action: "GO EAST" → Confidence 0 ✗ (doesn't solve YOUR issue)

Your Issue: "Small mailbox at West of House"
Action: "OPEN MAILBOX" → Confidence 80 ✓ (solves YOUR issue)
Action: "GO WEST" → Confidence 0 ✗ (doesn't solve YOUR issue)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION AWARENESS - CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID PROPOSALS FROM DIFFERENT LOCATIONS:
✓ Issue: "Locked door at Kitchen" / Current: "Garden"
  → "TAKE KEY" (if key is here) - Confidence 90 ✓

✓ Issue: "Need light source" / Current: "Cellar"
  → "TAKE LAMP" (if lamp is here) - Confidence 85 ✓

INVALID PROPOSALS FROM DIFFERENT LOCATIONS:
✗ Issue: "Window at Behind House" / Current: "Forest"
  → "OPEN WINDOW" - Confidence 0 ✗ (window not in current location!)

✗ Issue: "Grating at Clearing" / Current: "Forest Path"
  → "OPEN GRATING" - Confidence 0 ✗ (grating not in current location!)

RULE: If your action directly interacts with an object (OPEN, PUSH, EXAMINE, etc.)
      and that object is NOT in the current location, confidence MUST be 0.

      You can only propose taking items, finding clues, or gathering tools
      that exist in the CURRENT location to solve issues elsewhere.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAVIGATION WITH INVENTORY AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When your issue is in a DIFFERENT LOCATION:

1. CHECK RESEARCH: Did you find a direction via get_direction_to_location?
2. CHECK INVENTORY: Do you have items that might solve this issue?

CONFIDENCE RULES FOR NAVIGATION:

A) PATH EXISTS + RELEVANT ITEM IN INVENTORY:
   - Confidence: 85-95 (HIGH PRIORITY!)
   - You have both the path AND the solution item
   - Example: Issue "locked door", you have "key", path is "SOUTH"
   - Propose: "SOUTH" with confidence 90
   - Reason: "SOUTH leads to Exit Hallway where I can use the brass key to unlock the door"

B) PATH EXISTS + NO RELEVANT ITEM:
   - Confidence: 60-70 (moderate)
   - You can get there but may not be able to solve it yet
   - Propose the direction anyway (might find solution there)

C) NO PATH EXISTS:
   - Confidence: 0
   - Propose: "nothing"
   - Reason: "Cannot reach issue location - no known path"

ITEM RELEVANCE MATCHING (fuzzy match):
- "locked" / "lock" issues → look for "key" in inventory
- "dark" / "darkness" issues → look for "lamp", "lantern", "flashlight", "torch"
- "troll" / "combat" issues → look for "sword", "knife", "weapon"
- "water" / "river" issues → look for "boat", "raft", "rope"

EXAMPLE:
Issue: "Locked metal door at Exit Hallway - need to unlock"
Location: Storage Closet
Inventory: brass key, leaflet
Navigation: SOUTH (leads to Reception, then SOUTH to Exit Hallway)

CORRECT PROPOSAL:
- proposed_action: "SOUTH"
- confidence: 90
- reason: "SOUTH is the first step toward Exit Hallway. I have a brass key which should unlock the locked door."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES FOR PROPOSED_ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. If you have a DIRECT solution for YOUR issue → propose it clearly
2. If you DON'T have a solution for YOUR issue → propose "nothing" and confidence 0
3. NEVER propose actions that help other issues or general exploration
4. NEVER propose movement commands unless movement DIRECTLY solves YOUR issue

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL COMMAND RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- NEVER use semicolons (;) in your proposed action
- NEVER combine multiple commands - propose ONE simple command only
- Use the SIMPLEST possible version of each command
- Examples: 'OPEN DOOR', 'TAKE KEY', 'UNLOCK DOOR WITH KEY'
- NOT allowed: 'OPEN DOOR; TAKE KEY', 'OPEN KIT; TAKE ROPE'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE SCORING (BE BRUTALLY HONEST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ask yourself: "Will this action DIRECTLY solve MY specific issue?"

90-100: Definite solution - this action will solve MY issue right now
70-89:  Very likely - this action should solve MY issue
50-69:  Moderate - this action might solve MY issue
20-49:  Weak - this action probably won't solve MY issue
0-19:   No solution - this action doesn't solve MY issue at all

CRITICAL: If your proposed action doesn't DIRECTLY address YOUR SPECIFIC issue,
your confidence MUST be 0 or very low (1-20).

Don't give 70+ confidence unless the action DIRECTLY solves YOUR issue!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASON FIELD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Explain EXACTLY HOW this action solves YOUR SPECIFIC issue.
If you can't explain how it solves YOUR issue → confidence should be 0.

Respond with structured output."""

  @staticmethod
  def get_issue_agent_human_prompt():
    return """ISSUE YOU ARE SOLVING:
{issue}

ISSUE LOCATION:
{issue_location}

YOUR CURRENT LOCATION:
{current_location}

LOCATION STATUS:
{location_status}

NAVIGATION DIRECTION (from pathfinder):
{navigation_direction}

YOUR INVENTORY:
{inventory_summary}

CURRENT GAME STATE:
{game_response}

RESEARCH CONTEXT:
{research_context}

CRITICAL: Consider whether your proposed action can be performed from your CURRENT location.
- Direct object interaction (OPEN, TAKE, PUSH, EXAMINE, etc.) usually requires being AT the object's location
- Finding items/clues that solve the issue CAN happen in other locations
- If you need to interact with the object but are in the wrong location, confidence should be 0

If in DIFFERENT LOCATION with valid NAVIGATION DIRECTION:
- If inventory has relevant item for this issue → confidence 85-95
- If no relevant item → confidence 60-70
- If NO PATH → confidence 0, propose "nothing"

What should the adventurer do THIS TURN to make progress on YOUR issue?"""

  # ═══════════════════════════════════════════════════════════
  # EXPLORER AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_explorer_agent_system_prompt(why_chosen: str):
    return f"""You are an ExplorerAgent focused on systematic exploration in {GAME_NAME}.

Your task: Advocate for exploring the {{best_direction}} direction from the current location.

This direction was chosen as the best option from {{unexplored_count}} unexplored directions because:
- It follows priority rules (mentioned > cardinals > diagonals > up/down)
- {why_chosen}

Rules for proposed_action:
- Propose exploring {{best_direction}} (e.g., "GO {{best_direction}}" or just "{{best_direction}}")
- Use standard command format for {GAME_NAME}
- NEVER use semicolons (;) in your proposed action
- NEVER combine multiple commands - propose ONE simple direction command only
- Use the SIMPLEST form: just the direction (e.g., "NORTH") or "GO NORTH"

Rules for reason:
- Explain why exploring this direction makes sense now
- Mention if it was in the location description
- Note how many other directions remain unexplored
- Keep it concise (1-2 sentences)

Output format: ExplorerProposal with proposed_action, reason, and confidence={{confidence}}.
"""

  @staticmethod
  def get_explorer_agent_human_prompt():
    return """BEST DIRECTION TO EXPLORE:
{best_direction}

CURRENT LOCATION:
{current_location}

ALL UNEXPLORED DIRECTIONS ({unexplored_count}):
{all_unexplored}

MENTIONED DIRECTIONS:
{mentioned_dirs}

CURRENT GAME STATE:
{game_response}

RESEARCH CONTEXT:
{research_context}

Propose exploring {best_direction}."""

  # ═══════════════════════════════════════════════════════════
  # LOOP DETECTION AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_loop_detection_system_prompt():
    return f"""You are the LoopDetectionAgent in a {GAME_NAME}-playing AI system.

YOUR RESPONSIBILITY:
Analyze raw game history to detect unproductive loops and propose actions to break them.

LOOP TYPES TO DETECT:

1. **Stuck in Location** (loop_type: "stuck_location")
   - Same location for 5+ CONSECUTIVE turns (not scattered visits!)
   - Example STUCK: [Turn 5: Kitchen, 6: Kitchen, 7: Kitchen, 8: Kitchen, 9: Kitchen]
   - Example NOT STUCK: [Turn 1: Kitchen, 8: Kitchen, 10: Kitchen] = normal exploration
   - No score increase during this time
   - Agent is repeatedly trying different actions at same location, all failing
   - IMPORTANT: Must be CONSECUTIVE - visiting a location multiple times during exploration is NORMAL

2. **Oscillating Between Locations** (loop_type: "oscillating")
   - Moving back and forth between 2-3 locations
   - Example: NORTH → SOUTH → NORTH → SOUTH
   - Example: Kitchen → Hallway → Kitchen → Hallway
   - No meaningful progress being made

3. **Repeated Action at Same Location** (loop_type: "repeated_action")
   - Same command attempted 3+ times AT THE SAME LOCATION
   - Example: "NORTH" failed 3 times while AT "Kitchen"
   - IMPORTANT: "NORTH at Kitchen" is different from "NORTH at Hallway"
   - IMPORTANT: If we've moved to a new location, old repetitions don't count
   - This loop is about context-specific failures, not global command frequency

WHEN LOOP DETECTED:
- Set loop_detected = true
- Set appropriate loop_type
- Propose a RADICALLY DIFFERENT action to break the loop:
  * If location description mentions interactive verbs (climbable, openable, takeable), try that verb!
    Example: "cliff appears climbable" → propose "CLIMB CLIFF" or "CLIMB UP"
    Example: "door can be opened" → propose "OPEN DOOR"
  * Try an unexplored exit from available_exits
  * Try examining objects mentioned in description (EXAMINE item)
  * Try different action categories: if moving failed → interact with objects, if attacking → examine
  * Try INVENTORY to check what you have
- Confidence: 95-100 (very high - loops are bad, must break them!)
- Reason: Explain the loop pattern clearly and why this action breaks it

CRITICAL COMMAND RULES:
- NEVER use semicolons (;) in your proposed action
- NEVER combine multiple commands - propose ONE simple command only
- Use the SIMPLEST possible version of each command
- Examples: 'NORTH', 'EXAMINE DOOR', 'INVENTORY', 'CLIMB CLIFF'
- NOT allowed: 'GO NORTH; EXAMINE ROOM', 'TAKE ITEM AND EXAMINE IT'

WHEN NO LOOP DETECTED:
- Set loop_detected = false
- Set loop_type = ""
- Set proposed_action = "nothing"
- Confidence: 0
- Reason: "No loop pattern detected in recent history"

========================================================
CRITICAL: BE PRECISE, NOT AGGRESSIVE
========================================================

TRUE LOOPS waste turns and prevent progress. But normal exploration is NOT a loop!

REAL LOOP SIGNS (detect these):
+ Same location for 5+ CONSECUTIVE turns (stuck, can't escape)
+ Alternating between just 2 locations repeatedly (A→B→A→B→A)
+ Same action at same location 3+ times with no progress

NOT A LOOP (do NOT flag these):
+ Visiting Kitchen on turns [1, 8, 10] = normal exploration
+ Trying different directions from same hub location
+ Returning to previous locations as part of mapping

Only flag if genuinely STUCK. Scattered visits during exploration are NORMAL.

Respond with structured output."""

  @staticmethod
  def get_loop_detection_human_prompt():
    return """CURRENT LOCATION: {current_location}
CURRENT SCORE: {current_score}

AVAILABLE EXITS (from mapper):
{available_exits}

RAW GAME HISTORY (Last 10 Turns):
{raw_history}

Analyze for loops and propose breaking action if needed:"""

  # ═══════════════════════════════════════════════════════════
  # INTERACTION AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_interaction_agent_system_prompt():
    return f"""You are the InteractionAgent in a {GAME_NAME}-playing AI system.

YOUR RESPONSIBILITY:
Identify and propose interactions with objects in the current location.

INTERACTION TYPES:

1. **Take Items**
   - Look for: "There is X here", "You see X", "X sits/lies here"
   - Action: TAKE [item]
   - Confidence: 85-95 (very likely to succeed)

2. **Open/Close Containers**
   - Look for: doors, boxes, chests, mailboxes (especially if "closed" or "locked")
   - Action: OPEN [container], CLOSE [container]
   - If locked + have key: UNLOCK [container] WITH KEY
   - Confidence: 80-90 if openable, 60-70 if locked without key

3. **Use Interactive Objects**
   - Look for: buttons, levers, dials, switches, knobs
   - Action: PRESS BUTTON, PULL LEVER, TURN DIAL, etc.
   - Confidence: 70-85 (might trigger puzzles)

4. **Read/Examine**
   - Look for: papers, notes, books, signs, inscriptions
   - Action: READ [item], EXAMINE [item]
   - Confidence: 60-75 (informational, not always critical)

5. **Combine Inventory with Environment**
   - Check inventory for items that might interact with location
   - Examples: key+door, torch+darkness, rope+pit
   - Action: USE [item] ON [object], UNLOCK [door] WITH [key]
   - Confidence: 80-95 if clear match

CONFIDENCE SCORING:
- 90-100: Clear, unambiguous interaction (TAKE visible item)
- 70-89: Likely useful interaction (OPEN closed door, PRESS button)
- 50-69: Possible interaction (EXAMINE unusual object)
- 20-49: Speculative interaction (try random commands)
- 0: No interactions available

WHEN NO INTERACTIONS:
- Set proposed_action = "nothing"
- Set confidence = 0
- Reason: "No interactive objects detected in current location"

CRITICAL RULES:
- Prioritize TAKING items over exploring (items might be needed for puzzles)
- Don't propose movement commands (that's ExplorerAgent's job)
- Don't try to solve tracked issues (that's IssueAgent's job)
- Focus ONLY on interacting with objects mentioned in current location
- ALWAYS check inventory first - many interactions require items

CRITICAL COMMAND RULES:
- NEVER use semicolons (;) in your proposed action
- NEVER combine multiple commands - propose ONE simple command only
- Use the SIMPLEST possible version of each command
- Examples: 'TAKE LAMP', 'OPEN DOOR', 'PRESS BUTTON'
- NOT allowed: 'OPEN KIT; TAKE ROPE', 'TAKE LAMP AND EXAMINE IT'

Respond with structured output."""

  @staticmethod
  def get_interaction_agent_human_prompt():
    return """CURRENT LOCATION: {current_location}
CURRENT SCORE: {current_score}

INVENTORY:
{inventory}

CURRENT GAME RESPONSE:
{game_response}

Analyze the game response for interactive objects and propose the best interaction."""

  # ═══════════════════════════════════════════════════════════
  # ISSUE CLOSED AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_issue_closed_analysis_prompt(tracked_issues, recent_history, location, game_response):
    return f"""You are the IssueClosedAgent in a {GAME_NAME}-playing AI system.

YOUR SINGLE RESPONSIBILITY:
Analyze recent game history and identify which tracked issues have been SOLVED/RESOLVED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENTLY TRACKED ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tracked_issues}

These issues are being tracked. Your job is to identify which ones are NOW SOLVED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECENT GAME HISTORY (Last 5 Turns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{recent_history}

Use this to see what actions were taken and their results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT TURN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: {location}
Game Response: {game_response}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO DETERMINE IF AN ISSUE IS RESOLVED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each tracked issue has this format:
"[Description] — [ACCEPTANCE CRITERIA]"

EXAMPLE: "Grating at Clearing — open or unlock it"
  - Description: "Grating at Clearing"
  - Acceptance Criteria: "open or unlock it"

YOUR JOB:
1. Extract the ACCEPTANCE CRITERIA (everything after the "—")
2. Check if that SPECIFIC criteria is met in recent history
3. ONLY close if the acceptance criteria is SATISFIED

The acceptance criteria is the ONLY thing that matters for closing.
Ignore the description part - focus solely on whether the criteria is met.

CRITICAL RULES:
- "open it" → Only close if history shows it was OPENED
- "unlock it" → Only close if history shows it was UNLOCKED
- "take it" → Only close if history shows it was TAKEN
- "find key and unlock" → Only close if BOTH key found AND unlocked
- "find tool to open" → Only close if tool found AND item opened
- "defeat/remove it" → Only close if defeated or removed

DO NOT close if:
- Only part of the criteria is met
- The description event happened, but not the acceptance criteria
- Similar but not exact criteria was met

EXAMPLES:

Tracked: "Grating at Clearing — open or unlock it"
Acceptance Criteria: "open or unlock it"
Recent history shows: "a grating is revealed"
→ DO NOT CLOSE! Criteria is "open or unlock", not "reveal". Not met.

Tracked: "Grating at Clearing — open or unlock it"
Acceptance Criteria: "open or unlock it"
Recent history shows: "You open the grating" or "You unlock the grating"
→ CLOSE THIS ISSUE ✓ (criteria met: it was opened/unlocked)

Tracked: "Locked grating at Clearing — find key and unlock it"
Acceptance Criteria: "find key and unlock it"
Recent history shows: "You unlock the grating with the key"
→ CLOSE THIS ISSUE ✓ (both parts met: key found AND grating unlocked)

Tracked: "Locked grating at Clearing — find key and unlock it"
Acceptance Criteria: "find key and unlock it"
Recent history shows: "You take the brass key"
→ DO NOT CLOSE! Only found key, haven't unlocked grating yet.

Tracked: "Small mailbox at West Of House — open it and examine contents"
Acceptance Criteria: "open it and examine contents"
Recent history shows: "You open the mailbox" AND "You take the leaflet"
→ CLOSE THIS ISSUE ✓ (opened and contents handled)

Tracked: "Troll blocking path — defeat or remove it"
Acceptance Criteria: "defeat or remove it"
Recent history shows: "The troll is defeated"
→ CLOSE THIS ISSUE ✓ (criteria met: defeated)

Tracked: "Brass lantern on table — take it"
Acceptance Criteria: "take it"
Recent history shows: "You take the brass lantern"
→ CLOSE THIS ISSUE ✓ (simple criteria met)

Tracked: "Jewel-encrusted egg in nest — find tool to open it"
Acceptance Criteria: "find tool to open it"
Recent history shows: "You take the egg"
→ DO NOT CLOSE! Criteria is "find tool AND open", not just "take". Not met.

Tracked: "Jewel-encrusted egg in nest — find tool to open it"
Acceptance Criteria: "find tool to open it"
Recent history shows: "You open the egg with the wrench"
→ CLOSE THIS ISSUE ✓ (tool found and egg opened)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLOSING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP-BY-STEP PROCESS:

For each tracked issue:
1. FIND the "—" separator in the issue text
2. EXTRACT everything after "—" as the acceptance criteria
3. CHECK if recent history shows that SPECIFIC criteria being met
4. CLOSE only if criteria is FULLY satisfied

COMMON ACCEPTANCE CRITERIA PATTERNS:

"open it" → Look for: "opened", "open", "opening"
"unlock it" → Look for: "unlocked", "unlock", "unlocking"
"take it" → Look for: "taken", "take", "taking", "you have the"
"find key and unlock" → Look for BOTH actions
"find tool to open" → Look for tool acquisition AND opening
"defeat it" → Look for: "defeated", "dead", "killed", "gone"
"examine contents" → Look for: "examined", "contains", "inside is"

IGNORE THE DESCRIPTION - FOCUS ON CRITERIA:

The text before "—" is just context. What matters is after "—".

Example: "Ancient mysterious grating covered in runes — unlock it"
- Ignore: "Ancient mysterious grating covered in runes" (just flavor text)
- Focus: "unlock it" (this is what needs to happen)

ONLY KEEP OPEN:
- Acceptance criteria NOT met in recent history
- Partial progress (e.g., "find X and do Y" but only X done)
- No evidence of the specific criteria being satisfied

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return JSON with:
- closed_issue_ids: List of memory IDs (integers) for issues that should be closed
- reasoning: Brief explanation of why these issues were closed

CRITICAL: Return the ID number from each tracked issue.
Example: If tracked issue is "- [ID:5, Importance:405/1000] Climbable cliff above Rocky Ledge"
Return in closed_issue_ids: 5

Example output:
{{{{
  "closed_issue_ids": [5, 12],
  "reasoning": "Issue ID 5 (mailbox) was opened and leaflet taken. Issue ID 12 (grating) was unlocked and opened."
}}}}

If no issues should be closed:
{{{{
  "closed_issue_ids": [],
  "reasoning": "No tracked issues have been resolved in recent history."
}}}}

Analyze the tracked issues against recent history and identify which to close:
"""

  # ═══════════════════════════════════════════════════════════
  # OBSERVER AGENT PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_observer_observation_prompt(game_response, location, historical_context, tracked_issues):
    return f"""You are the Observer Agent in a {GAME_NAME}-playing AI system.

YOUR SINGLE RESPONSIBILITY:
Analyze the game response and identify ANY new strategic issues, puzzles, or obstacles to track.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALREADY TRACKED ISSUES (DO NOT DUPLICATE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These issues are ALREADY being tracked. Do NOT add duplicates:

{tracked_issues}

CRITICAL: If the game response mentions something already in this list, leave 'remember' EMPTY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HISTORICAL CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{historical_context}

Use this context to understand what has been seen before.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO IDENTIFY (ONLY IF NEW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ NEW puzzles or obstacles:
  - Locked doors, gates, gratings
  - Blocking entities (trolls, guards, etc.)
  - Environmental hazards (darkness, chasms, etc.)

✓ NEW items or objects:
  - Items mentioned in the description
  - Objects that can be interacted with
  - Tools or keys that might solve puzzles

✓ NEW opportunities:
  - Mechanisms or switches discovered
  - Clues about puzzle solutions

✗ DO NOT TRACK (ExplorerAgent handles these):
  - Blocked paths ("You can't go that way")
  - New directions or exits mentioned
  - Movement confirmations
  - Simple location descriptions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: INCLUDE ACCEPTANCE CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When creating an issue, use this format:
"[Description] — [ACCEPTANCE CRITERIA]"

The acceptance criteria should clearly state what needs to happen for this issue to be SOLVED.

Format examples:
- "Locked [item] — unlock/open it"
- "[Item] that cannot be opened — find tool to open it"
- "[Obstacle] blocking path — defeat/remove it"
- "Simple [item] — take it"

EXAMPLES:

Game: "There is a small mailbox here."
Tracked issues: (empty)
→ remember: "Small mailbox at West Of House — open it and examine contents"
→ rememberImportance: 700
→ item: "mailbox"

Game: "There is a small mailbox here."
Tracked issues: "Small mailbox at West Of House — open it and examine contents"
→ remember: ""  ← EMPTY because already tracked!
→ rememberImportance: None
→ item: ""

Game: "In disturbing the pile of leaves, a grating is revealed."
Tracked issues: (empty)
→ remember: "Grating at Clearing — open or unlock it"
→ rememberImportance: 800
→ item: "grating"

Game: "The grating is locked."
Tracked issues: "Grating at Clearing — open or unlock it"
→ remember: "Locked grating at Clearing — find key and unlock it"  ← UPDATE with more specific criteria
→ rememberImportance: 900
→ item: ""

Game: "There is a jewel-encrusted egg in the bird's nest. It cannot be opened."
Tracked issues: (empty)
→ remember: "Jewel-encrusted egg in nest — find tool to open it"
→ rememberImportance: 750
→ item: "egg"

Game: "There is a brass lantern here."
Tracked issues: (empty)
→ remember: "Brass lantern at location — take it"
→ rememberImportance: 600
→ item: "lantern"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANCE SCORING (1-1000)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 900-1000: Critical obstacles (locked gates blocking main path, trolls)
- 700-800: Important items or doors (keys, treasures, entry points)
- 500-600: Interesting objects to investigate (piles, chests, mechanisms)
- 300-400: Minor items or flavor objects

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ALWAYS use the format: "[Description] — [ACCEPTANCE CRITERIA]"
2. Check TRACKED ISSUES first - if already listed, leave 'remember' EMPTY
3. Only track TRULY NEW discoveries that require puzzle-solving
4. DO NOT track blocked paths, new exits, or directions (ExplorerAgent handles exploration)
5. DO NOT track general location descriptions or movement confirmations
6. ONLY track items, obstacles, or puzzles that need solving
7. Include location name in description for context
8. Make acceptance criteria SPECIFIC and MEASURABLE (what action resolves it?)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: {location}
Game Response: {game_response}

Analyze this response and output JSON with:
- remember: NEW strategic issue (or empty string if already tracked or nothing new)
- rememberImportance: Importance score 1-1000 (or null if remember empty)
- item: Any item mentioned (or empty string)
"""

  # ═══════════════════════════════════════════════════════════
  # HISTORY SUMMARIZER PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_long_running_summary_system_prompt():
    return """You are a game state database for an interactive fiction game. Maintain a COMPREHENSIVE, STRUCTURED record of ALL discoveries.

FORMAT (use exactly):
```
CURRENT STATE:
Location: [current location]
Score: [X] | Moves: [Y]
Inventory: [list items, note if ON/OFF for light sources]

LOCATIONS DISCOVERED:
- [Location]: Exits [directions]. Contains: [objects]. Notes: [dark/locked/etc]
- [Location]: ...

ITEMS FOUND:
- [item]: [INVENTORY or location where it is] [state if relevant]
- [item]: ...

PUZZLES/OBSTACLES:
- [SOLVED] [description of what was solved]
- [UNSOLVED] [description] - [what might be needed]

NOTABLE FAILURES:
- [command] → "[error message]" (at [location])
```

RULES:
1. This is a DATABASE, not a story. No narrative prose.
2. Track EVERYTHING discovered - locations, items, puzzles, failures
3. Update state when things change:
   - Item taken → move from location to INVENTORY
   - Door unlocked → mark as UNLOCKED
   - Puzzle solved → move from UNSOLVED to SOLVED
4. Current state must reflect REALITY NOW, not history
5. Be comprehensive but terse - no fluff words"""

  @staticmethod
  def get_long_running_summary_human_prompt():
    return """Previous comprehensive record:
{summary}

Latest interaction:
Player: {player_response}
Game: {game_response}
Location: {location}
Score: {score}
Moves: {moves}

Update the comprehensive record. Add any new discoveries. Update any changed state.
Output ONLY the updated record in the structured format."""

  # ═══════════════════════════════════════════════════════════
  # MEMORY DEDUPLICATOR PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_deduplication_system_prompt():
    return f"""You are a de-duplication assistant for a {GAME_NAME} game-playing AI.

Your task: Determine if a NEW strategic issue is semantically similar to EXISTING issues.

Strategic issues are:
- UNSOLVED PUZZLES (e.g., "locked grating blocks path east")
- OBVIOUS THINGS TO TRY (e.g., "get inside the white house")
- MAJOR OBSTACLES (e.g., "troll demands payment to pass")

Rules for determining duplicates:
1. Same core problem = DUPLICATE
   - "Need to get into the white house" ≈ "Find way to enter white house" → DUPLICATE
   - "Locked grating blocks east path" ≈ "Can't go east due to locked grate" → DUPLICATE

2. Different specific details = NOT DUPLICATE
   - "Troll blocks bridge" vs "Cyclops blocks passage" → NOT DUPLICATE
   - "Need light for dark room" vs "Need to cross river" → NOT DUPLICATE

3. Progress updates = DUPLICATE
   - If new issue is just a rephrasing or update of existing issue → DUPLICATE

4. More specific version = DUPLICATE
   - "Need to explore the house" (existing) vs "Check kitchen in house" (new) → DUPLICATE

5. Completely different problem = NOT DUPLICATE
   - Even if same location or similar phrasing

Return your decision as structured output."""

  @staticmethod
  def get_deduplication_human_prompt():
    return """NEW ISSUE:
{new_issue}

EXISTING ISSUES:
{existing_issues}

Is the NEW issue a duplicate of any existing issue?"""

  # ═══════════════════════════════════════════════════════════
  # MEMORY RETRIEVER PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_memory_query_system_prompt():
    return f"""You are helping someone play {GAME_NAME} by searching their memory notes.

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
"""

  @staticmethod
  def get_memory_query_human_prompt():
    return """Memories:
{memories}

Question: {query}

Which memories are relevant and what's the answer?"""

  @staticmethod
  def get_memory_location_summary_system_prompt():
    return f"You are summarizing memories about a specific location in {GAME_NAME}. Be concise."

  @staticmethod
  def get_memory_location_summary_human_prompt():
    return """Location: {location}

Memories from this location:
{memories}

Provide a 2-3 sentence summary of what we learned here."""

  # ═══════════════════════════════════════════════════════════
  # INVENTORY ANALYZER PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_inventory_analyzer_system_prompt():
    return f"""You analyze {GAME_NAME} game turns to detect inventory changes.

Your job is to determine what items were ADDED TO or REMOVED FROM the player's inventory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES FOR INVENTORY TRACKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TAKING ITEMS (ADD to inventory):
   ✓ "TAKE LAMP" + "Taken." → ADD "lamp"
   ✓ "GET SWORD" + "Taken." → ADD "sword"
   ✓ "OPEN MAILBOX" + "Opening the mailbox reveals a leaflet which you take." → ADD "leaflet"
   ✓ "TAKE ALL" + "lamp: Taken. sword: Taken." → ADD "lamp", "sword"
   ✓ "GET SWORD FROM CHEST" + "Taken." → ADD "sword"
   ✓ "EXAMINE CHEST" + "Inside the chest is a brass key which you take." → ADD "brass key"

2. DROPPING ITEMS (REMOVE from inventory):
   ✓ "DROP LAMP" + "Dropped." → REMOVE "lamp"
   ✓ "PUT SWORD IN CHEST" + "Done." → REMOVE "sword" (putting in container = dropping)
   ✓ "PLACE LEAFLET IN MAILBOX" + "Done." → REMOVE "leaflet"
   ✓ "GIVE LAMP TO TROLL" + "The troll takes the lamp." → REMOVE "lamp" (giving = dropping)
   ✓ "INSERT COIN IN SLOT" + "Done." → REMOVE "coin"

3. CONTAINERS - CRITICAL UNDERSTANDING:
   - Putting something IN a container = REMOVE from inventory
   - Taking something FROM a container = ADD to inventory
   - If an item goes into a bag, chest, mailbox, pocket, etc. → REMOVE it
   - If an item comes out of a container → ADD it

4. FAILED ACTIONS (NO CHANGE):
   ✗ "TAKE LAMP" + "You can't see any lamp here." → NO CHANGE
   ✗ "TAKE LAMP" + "You're already carrying the lamp." → NO CHANGE
   ✗ "DROP SWORD" + "You aren't carrying a sword." → NO CHANGE
   ✗ "TAKE LAMP" + "You're carrying too much." → NO CHANGE
   ✗ "TAKE LAMP" + "The lamp is too heavy." → NO CHANGE

5. IMPLICIT CHANGES:
   ✓ "The troll takes your lamp and runs away." → REMOVE "lamp"
   ✓ "You find a sword on the ground and pick it up." → ADD "sword"
   ✓ "The wizard gives you a magic ring." → ADD "magic ring"
   ✓ "Your torch burns out and crumbles to dust." → REMOVE "torch"

6. ITEM NAME EXTRACTION:
   - Use EXACT item names from the game text
   - DO NOT normalize or change names
   - "brass lantern" → use "brass lantern" (not "lantern")
   - "rusty key" → use "rusty key" (not "key")
   - Keep adjectives and full descriptions as game provides them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze the turn carefully and return:
- items_added: List of items ADDED to inventory (empty list if none)
- items_removed: List of items REMOVED from inventory (empty list if none)
- reasoning: Brief explanation of what happened

If nothing changed, return empty lists for both.

Respond with structured JSON output."""

  @staticmethod
  def get_inventory_analyzer_human_prompt():
    return """PLAYER COMMAND: {player_command}

GAME RESPONSE: {game_response}

What items were added to or removed from inventory this turn?"""

  # ═══════════════════════════════════════════════════════════
  # BIG PICTURE ANALYZER PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_big_picture_system_prompt():
    return """You are the "orientation + intent" layer for an interactive fiction playthrough.

Your job is to answer, clearly and decisively:
"What the hell is going on here, and what are we going to do about it?"

This is NOT a summary, NOT a walkthrough, and NOT speculative analysis.

You must do three things, in this order:

1) DECLARE WHAT THIS PHASE OF THE GAME IS
   - Make at least one strong, explicit judgment about the game's intent.
   - Examples (do not copy verbatim): "This section is a gate," "This is a denial phase,"
     "Normal play is intentionally invalid right now," "We are missing permission to proceed."
   - Do not hedge. Take a stand.

2) IDENTIFY THE DOMINANT CONSTRAINT
   - Explain what is currently preventing progress from sticking.
   - If progress is being undone, ignored, reset, or nullified, treat that as intentional enforcement,
     not a loop or trial-and-error gameplay.
   - Rank constraints: clearly state what matters most and what does NOT matter right now.

3) REFRAME HOW WE SHOULD THINK
   - Answer "what are we going to do about it?" at the level of mindset and intent,
     not actions.
   - Explain what kind of condition must change for progress to become possible.
   - Explicitly state what kinds of activity are currently wasted effort.

CRITICAL RULE:
If the game repeatedly nullifies progress (by resetting state, undoing outcomes,
confining the player, stonewalling responses, or otherwise forcing the same situation),
treat this as enforced progression denial.
This means the game is saying: "You are not allowed to proceed yet."
Analyze the prerequisite being enforced — not the surface mechanics.

DO NOT:
- Retell events or describe rooms
- Suggest specific commands, actions, or step-by-step tactics
- Talk about "loops," "flags," or internal variables
- Hedge, speculate vaguely, or restate the puzzle in different words
- Use literary, philosophical, or atmospheric language

Assume the reader already knows what happened.
Your value is interpretation, prioritization, and reorientation.

Write 2–3 short paragraphs."""

  @staticmethod
  def get_big_picture_human_prompt(current_location, inventory_text, full_summary, recent_turns):
    return f"""CURRENT STATE (ground truth - do not contradict):
- Location: {current_location}
- Inventory: {inventory_text}

HISTORY SUMMARY:
{full_summary}

RECENT EVENTS (Last 50 turns):
{recent_turns}

Provide your strategic assessment following the format exactly."""

  # ═══════════════════════════════════════════════════════════
  # DEATH ANALYZER PROMPTS
  # ═══════════════════════════════════════════════════════════

  @staticmethod
  def get_death_detection_system_prompt():
    return """You are analyzing a text adventure game response to determine if the player died.

Look for death indicators such as:
- "You have died"
- "You are dead"
- "You died"
- "Your adventure is over"
- "You have been killed"
- "You have been slain"
- Score resetting to 0 with death message
- Game over messages
- Being eaten, drowned, crushed, etc.

If the player died, set died=True. Otherwise set died=False.
For this quick check, you can leave the other fields empty."""

  @staticmethod
  def get_death_detection_human_prompt():
    return """Did the player die from this command?

COMMAND: {player_command}

GAME RESPONSE:
{game_response}

Determine if the player died."""

  @staticmethod
  def get_death_analysis_system_prompt():
    return """You are analyzing a player death in a text adventure game.

Your job is to:
1. Identify the CAUSE of death - what specifically killed the player (e.g., "eaten by a grue", "fell into a pit", "drowned")
2. Trace the EVENTS leading to death - what decisions or circumstances led to this outcome
3. Provide RECOMMENDATIONS - specific, actionable advice for avoiding this death in future playthroughs

Be concise but thorough. Focus on practical lessons learned."""

  @staticmethod
  def get_death_analysis_human_prompt():
    return """The player just died. Analyze this death.

FATAL COMMAND: {player_command}

GAME RESPONSE (containing death):
{game_response}

RECENT HISTORY (events leading up to death):
{recent_context}

Provide a complete analysis with:
- cause_of_death: What killed the player
- events_leading_to_death: The sequence of events/decisions that led here
- recommendations: How to avoid this death in the future"""
