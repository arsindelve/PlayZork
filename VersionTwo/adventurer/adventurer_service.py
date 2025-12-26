from zork.zork_api_response import ZorkApiResponse
from .prompt_library import PromptLibrary
from .adventurer_response import AdventurerResponse

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.agent_graph import create_decision_graph
from game_logger import GameLogger


class AdventurerService:

    def __init__(self, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit):
        """
        Initializes the AdventurerService with LangGraph-managed flow.

        Flow: Research → Decide → Persist (managed by LangGraph)

        Args:
            history_toolkit: The HistoryToolkit instance for accessing game history
            memory_toolkit: The MemoryToolkit instance for storing strategic issues
        """
        self.history_toolkit = history_toolkit
        self.memory_toolkit = memory_toolkit
        self.logger = GameLogger.get_instance()

        # Turn number tracking (mutable reference for persist node)
        self.turn_number_ref = {"current": 0}

        # Create research agent and decision chain
        self.research_agent = self._create_research_agent()
        self.decision_chain = self._create_decision_chain()

        # Create LangGraph to manage the flow
        self.decision_graph = create_decision_graph(
            research_agent=self.research_agent,
            decision_chain=self.decision_chain,
            history_toolkit=self.history_toolkit,
            memory_toolkit=self.memory_toolkit,
            turn_number_ref=self.turn_number_ref
        )

    def _create_research_agent(self) -> Runnable:
        """
        Create the research agent that can call history tools (Phase 1)

        Returns:
            Runnable chain configured with history tools
        """
        # Use GPT-4o-mini for reasoning with tools
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Only history tools (memory is write-only, not queryable)
        tools = self.history_toolkit.get_tools()

        # Bind tools to the LLM and REQUIRE it to call at least one tool
        llm_with_tools = llm.bind_tools(tools, tool_choice="any")

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
        # Use GPT-4o-mini for decision making with structured output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

    def handle_user_input(self, last_game_response: ZorkApiResponse, turn_number: int) -> AdventurerResponse:
        """
        Execute the LangGraph decision flow: Research → Decide → Persist

        The graph manages the entire flow:
        1. Research node: Calls history tools to gather context
        2. Decision node: Generates AdventurerResponse with structured output
        3. Persist node: Stores strategic issues in memory (if flagged)

        Args:
            last_game_response: The most recent response from the Zork game
            turn_number: Current turn number for memory persistence

        Returns:
            The AI-generated response for the adventurer
        """
        # Update turn number reference for persist node
        self.turn_number_ref["current"] = turn_number

        # Initialize graph state
        initial_state = {
            "game_response": last_game_response,
            "research_context": "",
            "decision": None,
            "memory_persisted": False
        }

        # Execute the graph (Research → Decide → Persist)
        self.logger.log_research_start()
        final_state = self.decision_graph.invoke(initial_state)
        self.logger.log_research_complete(final_state["research_context"])

        # Extract decision from final state
        adventurer_response = final_state["decision"]

        # Log the final decision
        self.logger.log_decision(adventurer_response.command, adventurer_response.reason)

        if final_state["memory_persisted"]:
            self.logger.logger.info(
                f"MEMORY STORED: [{adventurer_response.rememberImportance}/1000] {adventurer_response.remember}"
            )

        return adventurer_response
