class PromptLibrary:
  """
  A static class to store and manage prompts.
  """

  @staticmethod
  def get_history_processor_human_prompt():
    return """
    Here is the previous game interaction history summary up to this point:

    {summary}

    Here is the most recent game interaction. Append it to the new summary you create:

    Player Said: {player_response}
    Game: {game_response}
    
    """

  @staticmethod
  def get_history_processor_system_prompt():
    return "You are assisting someone playing ZORK I. You will summarize the game interaction history narratively, in a way that is most useful for helping them understand what has happened so far. Only summarize past interactions, do not provide advice or strategy. Do not provide a heading or title."

  @staticmethod
  def get_adventurer_prompt():
    return """
    You have played {moves} moves and have a score of {score}.

    You are currently in this location: {locationName}

    The game just responded: {game_response}

    Research context from your investigation:

    {research_context}

    Instructions: Based on the research you conducted and current game state, provide a JSON output without backticks. Use the following format:

    {{
        "command": "your command here. Choose commands that take logical next steps toward game progression and avoid previously attempted actions unless new clues or tools suggest a different outcome. Use your history to avoid going in circles. When stuck, explore new options. You want to go somewhere, you have to navigate manually using cardinal directions like NORTH, SOUTH, etc. Try all directions even if not listed as as possible exit", 
        "reason": "brief explanation of why this command was chosen based on game state and history",
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

    You have access to tools that let you query game history:
    - get_recent_turns(n): Get the last N turns of detailed game history
    - get_full_summary(): Get a complete narrative summary of all game history

    Current game state:
    - Score: {score}
    - Location: {locationName}
    - Moves: {moves}
    - Game Response: {game_response}

    Your task: Use the available tools to gather relevant context that will help make a good decision about what to do next.

    After you've gathered enough context, respond with:
    RESEARCH_COMPLETE: [Write a 2-3 sentence summary of the most relevant context you found]

    Example:
    RESEARCH_COMPLETE: The player has tried going NORTH from this location twice with no success. The summary indicates there's a rope in inventory that hasn't been used yet. The most recent turns show we're stuck in a loop in the forest area.
    """
