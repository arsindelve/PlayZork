class PromptLibrary:
  """
  A static class to store and manage prompts.
  """

  @staticmethod
  def get_adventurer_prompt():
    return """
    You have played {moves} moves and have a score of {score}. 

    You are currently in this location: {locationName}

    Here are your recent interactions with the game, from most recent to least recent. Study these carefully to avoid repeating yourself and going in circles: 

    {history}

    Instructions: Based on your recent game history, memories, and current context, provide a JSON output without backticks. Use the following format:

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
    Type "INVENTORY" to check items you’re carrying and "SCORE" to view your current score (so you know if you're winning).
    Type "LOOK" to see where you are, and what is in the current location. Use this liberally.
    Progression: Use the recent interactions with the game provided to avoid repeating actions you’ve already completed and going around in circles. 
    Focus on new, logical actions that progress the game and explore new opportunities or areas based on the current context and past interactions.
    """
