from zork.zork_api_response import ZorkApiResponse
from .prompt_library import PromptLibrary
from .adventurer_response import AdventurerResponse

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.history import HistoryToolkit


class AdventurerService:

    def __init__(self, history_toolkit: HistoryToolkit):
        """
        Initializes the AdventurerService with two-phase agent architecture.
        Phase 1: Research agent that can call history tools
        Phase 2: Decision chain with structured output

        Args:
            history_toolkit: The HistoryToolkit instance for accessing game history
        """
        self.history_toolkit = history_toolkit

        # Create research agent (Phase 1) and decision chain (Phase 2)
        self.research_agent = self._create_research_agent()
        self.decision_chain = self._create_decision_chain()

    def _create_research_agent(self) -> Runnable:
        """
        Create the research agent that can call history tools (Phase 1)

        Returns:
            Runnable chain configured with history tools
        """
        llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0)
        tools = self.history_toolkit.get_tools()

        # Bind tools to the LLM
        llm_with_tools = llm.bind_tools(tools)

        # Create prompt for research agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_research_agent_prompt()),
            ("human", "{input}"),
        ])

        # Return chain that can call tools
        return prompt | llm_with_tools

    def _create_decision_chain(self) -> Runnable:
        """
        Create the decision chain with structured output (Phase 2)

        Returns:
            Runnable chain that returns AdventurerResponse
        """
        llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0)

        # Retrieve the predefined system prompt
        system_prompt = PromptLibrary.get_system_prompt()

        # Retrieve the user-specific prompt for the adventurer's context
        user_prompt = PromptLibrary.get_adventurer_prompt()

        # Create a template for the system's initial message
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)

        # Create a template for the human player's input
        human_message = HumanMessagePromptTemplate.from_template(user_prompt)

        # Combine system and human prompts into a chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message])

        # Chain with structured output
        return chat_prompt_template | llm.with_structured_output(AdventurerResponse)

    def _call_tools_and_get_summary(self, research_input: dict) -> str:
        """
        Call the research agent and execute any tool calls

        Args:
            research_input: Input dictionary for the research agent

        Returns:
            Research summary string
        """
        try:
            # Invoke the research agent chain
            response = self.research_agent.invoke(research_input)

            # Check if there are tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"[Research] Agent called {len(response.tool_calls)} tool(s)")

                tool_results = []
                tools_map = {tool.name: tool for tool in self.history_toolkit.get_tools()}

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call.get('args', {})

                    if tool_name in tools_map:
                        print(f"[Research] Calling {tool_name} with args: {tool_args}")
                        tool_result = tools_map[tool_name].invoke(tool_args)
                        tool_results.append(f"{tool_name} result: {tool_result}")
                    else:
                        print(f"[Research] Unknown tool: {tool_name}")

                # Combine tool results into summary
                return "\n\n".join(tool_results) if tool_results else "No tools executed successfully."

            # If no tool calls, use the content directly
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            print(f"[Research] Error during research phase: {e}")
            import traceback
            traceback.print_exc()
            return "Research failed due to error."

    def handle_user_input(self, last_game_response: ZorkApiResponse) -> AdventurerResponse:
        """
        Two-phase decision making with research and structured decision output

        Phase 1: Research - agent gathers context via history tools
        Phase 2: Decision - structured output with research context

        Args:
            last_game_response: The most recent response from the Zork game

        Returns:
            The AI-generated response for the adventurer
        """
        # Phase 1: Research - agent gathers context
        research_input = {
            "input": "Use the available tools to gather relevant history context.",
            "score": last_game_response.Score,
            "locationName": last_game_response.LocationName,
            "moves": last_game_response.Moves,
            "game_response": last_game_response.Response
        }

        research_context = self._call_tools_and_get_summary(research_input)

        # Phase 2: Decision - structured output
        decision_input = {
            "score": last_game_response.Score,
            "locationName": last_game_response.LocationName,
            "moves": last_game_response.Moves,
            "game_response": last_game_response.Response,
            "research_context": research_context
        }

        adventurer_response = self.decision_chain.invoke(decision_input)

        # NOTE: History update now happens in GameSession, not here

        return adventurer_response
