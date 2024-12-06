from VersionTwo.zork.zork_api_response import ZorkApiResponse
from .adventurer_response import AdventurerResponse
from .openai_client import OpenAIClient

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


class AdventurerService:
    def __init__(self):
        """
        Initializes the AdventurerService with a OpenAIClient.
        """
        self.client = OpenAIClient()
        self.system_prompt = (
            "You are playing Zork One with the goal of winning the game by achieving a score of 350 points. "
            "Play as if for the first time, without relying on any prior knowledge of the Zork games.\n\n"
            "Objective: Reach a score of 350 points.\n"
            "Input Style: Use simple commands with one verb and one or two nouns, such as 'OPEN DOOR' or 'TURN SCREW WITH SCREWDRIVER.'\n"
            'Type "INVENTORY" to check items you’re carrying and "SCORE" to view your current score (so you know if you\'re winning).\n'
            'Type "LOOK" to see where you are, and what is in the current location. Use this liberally.\n'
            "Progression: Use the recent interactions with the game provided to avoid repeating actions you’ve already completed and going around in circles. "
            "Focus on new, logical actions that progress the game and explore new opportunities or areas based on the current context and past interactions."
        )

        # Memory to track conversation history
        self.memory = ConversationBufferMemory(return_messages=True)

        # Define a template for user prompts
        self.user_prompt_template = PromptTemplate(
            input_variables=["moves", "locationName", "score", "history"],
            template=(
            """
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
            """)
        )
        
    def handle_user_input(self, last_game_response: ZorkApiResponse) -> AdventurerResponse:
        """
        Handles the user's input and generates an AI response.
        :param last_game_response: The user's input (e.g., a command or question).
        :return: The AI's response.
        """
        # Generate the prompt using the memory and last response
        user_prompt = self.__generate_prompt(last_game_response)
        
        # Call OpenAI API with the system prompt and user prompt
        raw_response = self.client.call_openai(self.system_prompt, user_prompt)

        # Parse and return the response
        adventurer_response = AdventurerResponse.model_validate_json(raw_response)
        
        # Store the interaction in memory
        self.memory.chat_memory.add_user_message(user_prompt)
        self.memory.chat_memory.add_ai_message(adventurer_response.command)
     
        return adventurer_response

    def __generate_prompt(self, last_game_response: ZorkApiResponse):
        """
        Generates a prompt for the game using data from the Zork API and game state.
        """
        history = self.memory.load_memory_variables({}).get("history", "No previous interactions.")
        
        return self.user_prompt_template.format(
            score=last_game_response.Score,
            locationName=last_game_response.LocationName,
            moves=last_game_response.Moves,
            history=history
        )