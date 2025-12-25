from zork.zork_api_response import ZorkApiResponse
from .prompt_library import PromptLibrary
from .adventurer_response import AdventurerResponse
from .history_processor import HistoryProcessor

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI


class AdventurerService:

    def __init__(self):
        """
        Initializes the AdventurerService with a OpenAIClient.
        This includes setting up the history processor, defining the LLM, and configuring prompt templates.
        """
        cheap_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        expensive_llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Initialize the history processor to track turns between the game and the adventurer
        self.history = HistoryProcessor(cheap_llm)

        # Retrieve the predefined system prompt
        system_prompt = PromptLibrary.get_system_prompt()

        # Retrieve the user-specific prompt for the adventurer's context
        user_prompt = PromptLibrary.get_adventurer_prompt()

        # Create a template for the system's initial message
        system_message = SystemMessagePromptTemplate.from_template(
            system_prompt)

        # Create a template for the human player's input
        human_message = HumanMessagePromptTemplate.from_template(user_prompt)

        # Combine system and human prompts into a chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message])

        # Chain the chat prompt template with the LLM to handle the adventurer's responses
        self.chain = chat_prompt_template | expensive_llm.with_structured_output(
            AdventurerResponse)

    def handle_user_input(
            self, last_game_response: ZorkApiResponse) -> AdventurerResponse:
        """
        Handles the user's input and generates an AI response based on the game state.
        :param last_game_response: The most recent response from the Zork game.
        :return: The AI-generated response for the adventurer.
        """
        # Generate variables required for the prompt using the game state
        prompt_variables = self.__generate_prompt_variables(last_game_response)

        # Invoke the LLM chain with the generated variables to produce an adventurer response
        adventurer_response = self.chain.invoke(prompt_variables)

        # Add the interaction to the history for tracking and context
        self.history.add_turn(last_game_response.Response,
                              adventurer_response.command)

        return adventurer_response

    def __generate_prompt_variables(self, last_game_response: ZorkApiResponse):
        """
        Generates the variables required to populate the adventurer's user prompt.
        :param last_game_response: The most recent Zork game response containing game state details.
        :return: A dictionary of variables to be used in the prompt.
        """

        # Extract relevant details from the game response and history
        return {
            "score": last_game_response.Score,  # Current game score
            "locationName":
            last_game_response.LocationName,  # Current location in the game
            "moves":
            last_game_response.Moves,  # Number of moves taken in the game
            "history":
            self.history.get_messages()  # Previous interactions for context
        }
