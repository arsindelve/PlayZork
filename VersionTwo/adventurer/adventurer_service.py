from zork.zork_api_response import ZorkApiResponse
from .prompt_library import PromptLibrary
from .adventurer_response import AdventurerResponse

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.mapping import MapperToolkit
from tools.agent_graph import create_decision_graph, ExplorerAgent, LoopDetectionAgent, InteractionAgent, IssueClosedResponse, ObserverResponse
from game_logger import GameLogger
from config import get_cheap_llm, get_expensive_llm
from typing import List, Tuple, Optional


class AdventurerService:

    def __init__(self, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit, mapper_toolkit: MapperToolkit, inventory_toolkit):
        """
        Initializes the AdventurerService with LangGraph-managed flow.

        Flow: Research → Decide → Persist (managed by LangGraph)

        Args:
            history_toolkit: The HistoryToolkit instance for accessing game history
            memory_toolkit: The MemoryToolkit instance for storing strategic issues
            mapper_toolkit: The MapperToolkit instance for tracking the map
            inventory_toolkit: The InventoryToolkit instance for tracking inventory
        """
        self.history_toolkit = history_toolkit
        self.memory_toolkit = memory_toolkit
        self.mapper_toolkit = mapper_toolkit
        self.inventory_toolkit = inventory_toolkit
        self.logger = GameLogger.get_instance()

        # Turn number tracking (mutable reference for persist node)
        self.turn_number_ref = {"current": 0}

        # Create LLM for decisions and IssueAgent proposals (using expensive model)
        self.decision_llm = get_expensive_llm(temperature=0)

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
            inventory_toolkit=self.inventory_toolkit,
            turn_number_ref=self.turn_number_ref
        )

    def _create_research_agent(self) -> Runnable:
        """
        Create the research agent that can call history and mapper tools (Phase 1)

        Returns:
            Runnable chain configured with tools
        """
        # Use cheap model for reasoning with tools
        llm = get_cheap_llm(temperature=0)

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

        This chain evaluates proposals from IssueAgents and ExplorerAgent
        and chooses the best action.

        Returns:
            Runnable chain that returns AdventurerResponse
        """
        # Use new evaluation prompt that judges agent proposals
        system_prompt = PromptLibrary.get_decision_agent_evaluation_prompt()
        human_prompt = PromptLibrary.get_decision_agent_human_prompt()

        # Create a template for the system's initial message
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)

        # Create a template for the human player's input
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)

        # Combine system and human prompts into a chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message])

        # Chain with structured output (using shared decision_llm)
        return chat_prompt_template | self.decision_llm.with_structured_output(AdventurerResponse)

    def handle_user_input(self, last_game_response: ZorkApiResponse, turn_number: int) -> Tuple[AdventurerResponse, List, Optional[ExplorerAgent], Optional[LoopDetectionAgent], Optional[InteractionAgent], Optional[IssueClosedResponse], Optional[ObserverResponse], str]:
        """
        Execute the LangGraph decision flow: SpawnAgents → Research → Decide → CloseIssues → Observe → Persist

        The graph manages the entire flow:
        1. SpawnAgents node: Creates IssueAgents + ExplorerAgent + LoopDetectionAgent + InteractionAgent
        2. Research node: Calls history tools to gather context
        3. Decision node: Generates AdventurerResponse with structured output
        4. CloseIssues node: Identifies and removes resolved issues from memory
        5. Observe node: Identifies new strategic issues to track
        6. Persist node: Stores new strategic issues in memory (if flagged)

        Args:
            last_game_response: The most recent response from the Zork game
            turn_number: Current turn number for memory persistence

        Returns:
            Tuple of (AdventurerResponse, List[IssueAgent], Optional[ExplorerAgent], Optional[LoopDetectionAgent], Optional[InteractionAgent], Optional[IssueClosedResponse], Optional[ObserverResponse], str) - the decision, issue agents, explorer agent, loop detection agent, interaction agent, closed issues, observer response, and decision prompt
        """
        # Update turn number reference for persist node
        self.turn_number_ref["current"] = turn_number

        # Initialize graph state
        initial_state = {
            "game_response": last_game_response,
            "issue_agents": [],  # Will be populated by spawn_agents node
            "explorer_agent": None,  # Will be populated if unexplored directions exist
            "loop_detection_agent": None,  # Will be populated by spawn_agents node (always)
            "interaction_agent": None,  # Will be populated by spawn_agents node (always)
            "research_context": "",
            "decision": None,
            "decision_prompt": "",  # Will be populated by decision node
            "issue_closed_response": None,  # Will be populated by close_issues node
            "observer_response": None,  # Will be populated by observe node
            "memory_persisted": False
        }

        # Execute the graph (SpawnAgents → Research → Decide → CloseIssues → Observe → Persist)
        # Each agent logs its own activities and summaries
        self.logger.log_research_start()
        final_state = self.decision_graph.invoke(initial_state)
        self.logger.log_research_complete(final_state["research_context"])

        # Extract decision from final state
        # (All agents log their own summaries following SRP)
        adventurer_response = final_state["decision"]
        issue_closed_response = final_state["issue_closed_response"]
        observer_response = final_state["observer_response"]

        # Log the final decision
        self.logger.log_decision(adventurer_response.command, adventurer_response.reason)

        # Return the decision, issue agents, explorer agent, loop detection agent, interaction agent, closed issues, observer response, and decision prompt for display/reporting
        return adventurer_response, final_state["issue_agents"], final_state["explorer_agent"], final_state["loop_detection_agent"], final_state["interaction_agent"], issue_closed_response, observer_response, final_state.get("decision_prompt", "")
