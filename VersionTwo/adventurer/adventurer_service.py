from zork.zork_api_response import ZorkApiResponse
from .prompt_library import PromptLibrary
from .adventurer_response import AdventurerResponse

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.mapping import MapperToolkit
from tools.agent_graph import create_decision_graph, ExplorerAgent
from game_logger import GameLogger
from typing import List, Tuple, Optional


class AdventurerService:

    def __init__(self, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit, mapper_toolkit: MapperToolkit):
        """
        Initializes the AdventurerService with LangGraph-managed flow.

        Flow: Research → Decide → Persist (managed by LangGraph)

        Args:
            history_toolkit: The HistoryToolkit instance for accessing game history
            memory_toolkit: The MemoryToolkit instance for storing strategic issues
            mapper_toolkit: The MapperToolkit instance for tracking the map
        """
        self.history_toolkit = history_toolkit
        self.memory_toolkit = memory_toolkit
        self.mapper_toolkit = mapper_toolkit
        self.logger = GameLogger.get_instance()

        # Turn number tracking (mutable reference for persist node)
        self.turn_number_ref = {"current": 0}

        # Create LLM for decisions and IssueAgent proposals
        self.decision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Create research agent and decision chain
        self.research_agent = self._create_research_agent()
        self.decision_chain = self._create_decision_chain()

        # Create LangGraph to manage the flow
        self.decision_graph = create_decision_graph(
            research_agent=self.research_agent,
            decision_chain=self.decision_chain,
            decision_llm=self.decision_llm,
            history_toolkit=self.history_toolkit,
            memory_toolkit=self.memory_toolkit,
            mapper_toolkit=self.mapper_toolkit,
            turn_number_ref=self.turn_number_ref
        )

    def _create_research_agent(self) -> Runnable:
        """
        Create the research agent that can call history and mapper tools (Phase 1)

        Returns:
            Runnable chain configured with tools
        """
        # Use GPT-4o-mini for reasoning with tools
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Combine history and mapper tools
        tools = self.history_toolkit.get_tools() + self.mapper_toolkit.get_tools()

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

        # Chain with structured output (using shared decision_llm)
        return chat_prompt_template | self.decision_llm.with_structured_output(AdventurerResponse)

    def handle_user_input(self, last_game_response: ZorkApiResponse, turn_number: int) -> Tuple[AdventurerResponse, List, Optional[ExplorerAgent]]:
        """
        Execute the LangGraph decision flow: SpawnAgents → Research → Decide → Persist

        The graph manages the entire flow:
        1. SpawnAgents node: Creates IssueAgents + ExplorerAgent
        2. Research node: Calls history tools to gather context
        3. Decision node: Generates AdventurerResponse with structured output
        4. Persist node: Stores strategic issues in memory (if flagged)

        Args:
            last_game_response: The most recent response from the Zork game
            turn_number: Current turn number for memory persistence

        Returns:
            Tuple of (AdventurerResponse, List[IssueAgent], Optional[ExplorerAgent]) - the decision, issue agents, and explorer agent
        """
        # Update turn number reference for persist node
        self.turn_number_ref["current"] = turn_number

        # Initialize graph state
        initial_state = {
            "game_response": last_game_response,
            "issue_agents": [],  # Will be populated by spawn_agents node
            "explorer_agent": None,  # Will be populated if unexplored directions exist
            "research_context": "",
            "decision": None,
            "memory_persisted": False
        }

        # Execute the graph (SpawnAgents → Research → Decide → Persist)
        self.logger.log_research_start()
        final_state = self.decision_graph.invoke(initial_state)
        self.logger.log_research_complete(final_state["research_context"])

        # Log spawned agents and their proposals
        num_agents = len(final_state["issue_agents"])
        if num_agents > 0:
            self.logger.logger.info(f"SPAWNED {num_agents} IssueAgents - Proposals:")
            for i, agent in enumerate(final_state["issue_agents"][:10], 1):  # Log first 10
                self.logger.logger.info(
                    f"  {i}. [{agent.importance}/1000] {agent.issue_content}"
                )
                self.logger.logger.info(
                    f"     > Proposed: '{agent.proposed_action}' (confidence: {agent.confidence}/100)"
                )
                if agent.reason:
                    self.logger.logger.info(
                        f"     > Reason: {agent.reason}"
                    )

        # Log explorer agent if present
        explorer_agent = final_state["explorer_agent"]
        if explorer_agent:
            self.logger.logger.info(f"SPAWNED 1 ExplorerAgent - Proposal:")
            self.logger.logger.info(
                f"  [EXPLORE] {explorer_agent.best_direction} from {explorer_agent.current_location}"
            )
            self.logger.logger.info(
                f"  > Proposed: '{explorer_agent.proposed_action}' (confidence: {explorer_agent.confidence}/100)"
            )
            if explorer_agent.reason:
                self.logger.logger.info(
                    f"  > Reason: {explorer_agent.reason}"
                )

        # Extract decision from final state
        adventurer_response = final_state["decision"]

        # Log the final decision
        self.logger.log_decision(adventurer_response.command, adventurer_response.reason)

        if final_state["memory_persisted"]:
            self.logger.logger.info(
                f"MEMORY STORED: [{adventurer_response.rememberImportance}/1000] {adventurer_response.remember}"
            )

        # Return the decision, issue agents, and explorer agent for display
        return adventurer_response, final_state["issue_agents"], final_state["explorer_agent"]
