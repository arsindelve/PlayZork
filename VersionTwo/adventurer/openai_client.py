from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


class OpenAIClient:
    def __init__(self, temperature: float = 0.7, model_name: str = "gpt-4"):
        """
        Initializes the LangchainOpenAIClient with the specified model and temperature.
        :param temperature: Controls the randomness of the model's output.
        :param model_name: The OpenAI model to use (e.g., "gpt-4" or "gpt-3.5-turbo").
        """
        self.chat_model = ChatOpenAI(temperature=temperature, model=model_name)

    def call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """
        Calls OpenAI with a system prompt and a user prompt.
        :param system_prompt: The system-level instruction for context.
        :param user_prompt: The input from the user.
        :return: The response from the AI.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        print(user_prompt)

        response = self.chat_model.invoke(messages)
        
        print(response)
        
        return response.content
