class PromptLibrary:
  """
  A static class to store and manage prompts.
  """

  @staticmethod
  def get_history_processor_human_prompt():
    return """
    Previous summary:
    {summary}

    Latest interaction:
    Player: {player_response}
    Game: {game_response}

    Provide an updated narrative summary that incorporates this new interaction. Output ONLY the summary text.
    """

  @staticmethod
  def get_history_processor_system_prompt():
    return "You are assisting someone playing ZORK I. You will summarize the game interaction history narratively, in a way that is most useful for helping them understand what has happened so far. Only summarize past interactions, do not provide advice or strategy. Do not provide a heading or title. IMPORTANT: Only output the summary itself, nothing else. Do not include any meta-commentary, instructions, or explanations - just the narrative summary."

  @staticmethod
  def get_decision_agent_evaluation_prompt():
    return """You are the Decision Agent in a Zork-playing AI system.

YOUR SINGLE RESPONSIBILITY: Choose the best action from specialist agent proposals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIALIST AGENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- IssueAgents: Each solves a specific tracked puzzle (importance = value for winning)
- ExplorerAgent: Discovers new areas/items through systematic exploration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION CRITERIA (in priority order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. HIGH-VALUE PUZZLES FIRST
   - IssueAgent with importance 800-1000 + confidence 80+ = TOP PRIORITY
   - Solving major puzzles = points = winning (goal: 350 points)

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
   - <50: Last resort only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED VALUE CALCULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- IssueAgent EV = (importance/1000) × (confidence/100) × 100
- ExplorerAgent EV = (unexplored_count/10) × (confidence/100) × 50
- Choose highest EV unless heuristics override

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output JSON with:
- command: The action to execute (from chosen agent proposal)
- reason: Which agent you chose and WHY (explain your decision)
- moved: Direction if movement command (or empty string)

Note: You do NOT identify new issues - that's handled by a separate Observer Agent.
"""

  @staticmethod
  def get_decision_agent_human_prompt():
    return """=== GAME STATE ===
Location: {locationName}
Score: {score} | Moves: {moves}
Game Response: {game_response}

=== RESEARCH CONTEXT ===
{research_context}

=== AGENT PROPOSALS ===
{agent_proposals}

=== YOUR TASK ===
Evaluate the proposals above and choose the best action.

Consider:
1. Expected Value: Which proposal has highest (importance × confidence)?
2. Are we stuck in a loop? (prefer exploration)
3. Any consensus among agents?
4. What does research context warn against?

Choose the best proposal and explain your reasoning clearly in the 'reason' field.

Instructions: Provide a JSON output without backticks:

{{
    "command": "The command from the chosen proposal (or LOOK if uncertain)",
    "reason": "Explain which agent's proposal you chose and WHY. Example: 'Chose IssueAgent #2 (importance 800, confidence 85, EV 68.0) because solving the grating puzzle is critical for winning. Research shows we have the key. ExplorerAgent suggested NORTH (confidence 75, EV 37.5) but solving this puzzle takes priority.'",
    "remember": "Record STRATEGIC ISSUES only: (1) UNSOLVED PUZZLES you discovered, (2) OBVIOUS THINGS TO TRY that could unlock progress, (3) MAJOR OBSTACLES preventing advancement. Do NOT record observations, items, or general notes. Memory is limited. Leave empty if no strategic issue discovered this turn.",
    "rememberImportance": "Score 1-1000 based on: How much will SOLVING/OVERCOMING this issue help us WIN the game (reach 350 points)? Major blocking puzzles/obstacles = 800-1000. Promising leads = 500-700. Minor puzzles = 100-400.",
    "item": "any new, interesting items you have found in this location, along with their locations. For example 'there is a box and a light bulb in the maintenance room'. Omit if there is nothing here.",
    "moved": "if you chose a movement command, list the direction you tried to go. Otherwise, leave this empty."
}}
"""

  @staticmethod
  def get_adventurer_prompt():
    return """
    You have played {moves} moves and have a score of {score}.

    You are currently in this location: {locationName}

    The game just responded: {game_response}

    === RESEARCH ANALYSIS ===
    {research_context}
    === END RESEARCH ANALYSIS ===

    CRITICAL DECISION-MAKING RULES:
    1. READ THE RESEARCH ANALYSIS ABOVE CAREFULLY - it tells you what to avoid and what to try
    2. If research says you're in a LOOP - BREAK OUT OF IT immediately by trying something completely different
    3. If the game says "already open", "already have", "securely anchored" - NEVER try that action again
    4. If you've been in the same location for 3+ turns without progress, TRY MOVING: NORTH, SOUTH, EAST, WEST, UP, DOWN
    5. Exploration is KEY - if stuck, try directional movement to find new areas
    6. Inventory management: Use INVENTORY to check what you have, DROP items if needed
    7. Every command must be NEW and PRODUCTIVE - no wasted turns

    Instructions: Provide a JSON output without backticks:

    {{
        "command": "Your NEXT command. Must NOT be a failed action from research analysis. If stuck, try directional movement.",
        "reason": "brief explanation based on research analysis and game state",
        "remember": "Record STRATEGIC ISSUES only: (1) UNSOLVED PUZZLES you discovered (e.g., 'locked grating blocks path east', 'need to cross the river somehow'), (2) OBVIOUS THINGS TO TRY that could unlock progress (e.g., 'get inside the white house', 'find a light source for dark areas'), (3) MAJOR OBSTACLES preventing advancement (e.g., 'troll demands payment to pass', 'cyclops is hostile and blocking path'). Do NOT record observations, items, or general notes. Memory is limited. Leave empty if no strategic issue discovered this turn.",
        "rememberImportance": "Score 1-1000 based on: How much will SOLVING/OVERCOMING this issue help us WIN the game (reach 350 points)? Major blocking puzzles/obstacles = 800-1000. Promising leads = 500-700. Minor puzzles = 100-400. This score determines priority when making decisions.",
        "item": "any new, interesting items you have found in this location, along with their locations, which are not already mentioned above. For example 'there is a box and a light bulb in the maintenance room'. Omit if there is nothing here.",
        "moved": "if you attempted to move in a certain direction, list the direction you tried to go. Otherwise, leave this empty."
    }}
    """

  @staticmethod
  def get_system_prompt():
    return """
    You are playing Zork One with the goal of winning the game by achieving a score of 350 points.
    Play as if for the first time, without relying on any prior knowledge of the Zork games.

    Objective: Reach a score of 350 points.
    Input Style: Use simple commands with one verb and one or two nouns, such as 'OPEN DOOR' or 'TURN SCREW WITH SCREWDRIVER.'
    Type "INVENTORY" to check items you're carrying and "SCORE" to view your current score (so you know if you're winning).
    Type "LOOK" to see where you are, and what is in the current location. Use this liberally.
    Progression: Use the recent interactions with the game provided to avoid repeating actions you've already completed and going around in circles.
    Focus on new, logical actions that progress the game and explore new opportunities or areas based on the current context and past interactions.
    """

  @staticmethod
  def get_research_agent_prompt():
    return """You are an assistant helping someone play Zork I.

    You have access to tools that let you query game history:

    HISTORY TOOLS:
    - get_recent_turns(n): Get the last N turns of detailed game history
    - get_full_summary(): Get a complete narrative summary of all game history

    Current game state:
    - Score: {score}
    - Location: {locationName}
    - Moves: {moves}
    - Game Response: {game_response}

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
