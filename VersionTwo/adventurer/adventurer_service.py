from VersionTwo.zork.zork_api_response import ZorkApiResponse
from .prompt_library import PromptLibrary
from .adventurer_response import AdventurerResponse

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage

import json

class AdventurerService:

    def __init__(self):
        """
        Initializes the AdventurerService with a OpenAIClient.
        """
        # Initialize the LLMs
        cheap_llm = ChatOpenAI(model="gpt-3.5-turbo",
                               temperature=0)  # Cheaper model
        expensive_llm = ChatOpenAI(
            model="gpt-4", temperature=0)  # More advanced model (if needed)

        # Retrieve the system and user prompts
        system_prompt = PromptLibrary.get_system_prompt()
        user_prompt = PromptLibrary.get_adventurer_prompt()

        # Create prompt templates for system and user messages
        system_message = SystemMessagePromptTemplate.from_template(
            system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(user_prompt)

        # Combine system and user messages into a chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message])
        self.chain = chat_prompt_template | cheap_llm

    def handle_user_input(
            self, last_game_response: ZorkApiResponse) -> AdventurerResponse:
        """
        Handles the user's input and generates an AI response.
        :param last_game_response: The user's input (e.g., a command or question).
        :return: The AI's response.
        """
        # Generate the input variables from the game state
        prompt_variables = self.__generate_prompt_variables(last_game_response)

        # Call the chain with the input variables to get the response
        response = self.chain.invoke(prompt_variables)

        # Ensure the response is an AIMessage and extract the content
       
        response_text = response.content  # Extract the content
        response_data = json.loads(response_text)

        # Validate the dictionary with AdventurerResponse
        try:
            adventurer_response = AdventurerResponse(**response_data)
        except Exception as e:
            raise ValueError(f"Validation failed for AdventurerResponse: {e}")

        return adventurer_response

    def __generate_prompt_variables(self, last_game_response: ZorkApiResponse):
        """
        Generates the variables required for the prompt.
        """
        return {
            "score": last_game_response.Score,
            "locationName": last_game_response.LocationName,
            "moves": last_game_response.Moves,
            "history": "No previous interactions."  # Placeholder for now
        }
