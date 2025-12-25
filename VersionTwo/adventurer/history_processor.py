from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from .prompt_library import PromptLibrary


class HistoryProcessor:

  def __init__(self, llm: ChatOpenAI) -> None:
    """
    Initialize the HistoryProcessor with an LLM and an empty history.
    :param llm: The language model to be used for summarization.
    """
    self.history = ""
    self.summarized_history = ""
    self.previous_command = "LOOK"

    # Create the system and user prompts
    system_message = SystemMessagePromptTemplate.from_template(
        PromptLibrary.get_history_processor_system_prompt())

    human_message = HumanMessagePromptTemplate.from_template(
        PromptLibrary.get_history_processor_human_prompt())

    # Combine prompts into a chat prompt template
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message, human_message])

    self.chain = chat_prompt_template | llm

  def add_turn(self, game_response: str, adventurer_response: str) -> None:
    """
    Add the latest game and adventurer responses to the history,
    summarize the history using the provided LLM, and append the summary.
    """
    # Append the current turn to the history
    self.history += f"Game: {game_response}\n"
    self.history += f"You: {adventurer_response}\n"

    # Invoke the LLM to get a summary
    self.summarized_history = self.chain.invoke(
        self.__generate_prompt_variables(game_response,
                                         self.previous_command)).content

    self.previous_command = adventurer_response

  def get_messages(self) -> str:
    """
    Retrieve the summarized conversation history or indicate if there are no interactions.
    """
    if self.history == "":
      return "No previous interactions."

    return self.summarized_history

  def __generate_prompt_variables(self, game_response: str,
                                  adventurer_response: str):

    # Extract relevant details from the game response and history
    return {
        "summary": self.summarized_history,
        "player_response": adventurer_response,
        "game_response": game_response
    }
