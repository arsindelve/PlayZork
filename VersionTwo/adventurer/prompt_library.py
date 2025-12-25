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
        "remember": "Use this field only for new, novel, or critical ideas for solving the game, new unsolved puzzles, or new obstacles essential to game progress. These are like Leonard's tattoos in Memento. Memory is limited, so avoid duplicates or minor details. Leave empty if unnecessary. Do not repeat yourself or duplicate reminders that already appear in the above prompt.",
        "rememberImportance": "the number, between 1 and 1000, of how important the above reminder is, 1 is not really, 1000 is critical to winning the game. Lower number items are likely to be forgotten when we run out of memory.",
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

    You have access to tools that let you query game history and memories:

    HISTORY TOOLS:
    - get_recent_turns(n): Get the last N turns of detailed game history
    - get_full_summary(): Get a complete narrative summary of all game history

    MEMORY TOOLS:
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
