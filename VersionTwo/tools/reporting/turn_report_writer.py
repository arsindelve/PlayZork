"""Turn Report Writer - Generates detailed markdown reports for each turn"""
from pathlib import Path
from typing import List, Optional
import logging


class TurnReportWriter:
    """Writes detailed turn reports to markdown files"""

    def __init__(self, logs_base_path: str = "logs/sessions"):
        """
        Initialize the report writer.

        Args:
            logs_base_path: Base directory for session logs
        """
        self.logs_base_path = logs_base_path
        self.logger = logging.getLogger(__name__)

    def write_turn_report(
        self,
        session_id: str,
        turn_number: int,
        location: str,
        score: int,
        moves: int,
        game_response: str,
        player_command: str,
        player_reasoning: str,
        issue_agents: List,
        explorer_agent,
        loop_detection_agent,
        decision_prompt: str,
        decision
    ):
        """
        Write a detailed markdown report for a single turn.

        Args:
            session_id: Game session ID
            turn_number: Turn number
            location: Current location
            score: Current score
            moves: Current move count
            game_response: Game's response text
            player_command: Command sent to game
            player_reasoning: Decision agent's reasoning
            issue_agents: List of IssueAgent objects
            explorer_agent: ExplorerAgent object or None
            loop_detection_agent: LoopDetectionAgent object or None
            decision_prompt: Formatted decision prompt
            decision: AdventurerResponse object
        """
        # Create session directory
        session_dir = Path(self.logs_base_path) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create markdown file
        report_path = session_dir / f"Turn-{turn_number}.md"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                # Header
                f.write(f"# Session {session_id} - Turn {turn_number}\n\n")

                # Game State
                f.write("## Game State\n")
                f.write(f"- **Location**: {location}\n")
                f.write(f"- **Score**: {score}\n")
                f.write(f"- **Moves**: {moves}\n")
                f.write(f"- **Game Output**:\n")
                f.write(f"  > {game_response}\n\n")
                f.write("---\n\n")

                # Agent Analysis
                f.write("## Agent Analysis\n\n")

                # IssueAgents
                if issue_agents:
                    for i, agent in enumerate(issue_agents, 1):
                        f.write(f"### IssueAgent #{i}: {agent.issue_content}\n")
                        f.write(f"**Importance**: {agent.importance}/1000\n")
                        f.write(f"**Location**: {agent.location}\n\n")

                        # Tool calls
                        if hasattr(agent, 'tool_calls_history') and agent.tool_calls_history:
                            f.write("**Research (Tool Calls)**:\n")
                            for tool_call in agent.tool_calls_history:
                                f.write(f"- `{tool_call['tool_name']}({tool_call['input']})`\n")
                                f.write(f"```\n{tool_call['output']}\n```\n\n")
                        else:
                            f.write("**Research (Tool Calls)**: None\n\n")

                        # Proposal
                        f.write("**Proposal**:\n")
                        f.write(f"- **Action**: {agent.proposed_action or 'None'}\n")
                        f.write(f"- **Confidence**: {agent.confidence or 0}/100\n")
                        f.write(f"- **Reason**: {agent.reason or 'N/A'}\n\n")
                        f.write("---\n\n")
                else:
                    f.write("*No IssueAgents active this turn*\n\n")
                    f.write("---\n\n")

                # ExplorerAgent
                if explorer_agent:
                    f.write("### ExplorerAgent\n")
                    f.write(f"**Best Direction**: {explorer_agent.best_direction}\n")
                    f.write(f"**Unexplored**: {len(explorer_agent.unexplored_directions)} directions\n")
                    f.write(f"**Current Location**: {explorer_agent.current_location}\n\n")

                    # Tool calls
                    if hasattr(explorer_agent, 'tool_calls_history') and explorer_agent.tool_calls_history:
                        f.write("**Research (Tool Calls)**:\n")
                        for tool_call in explorer_agent.tool_calls_history:
                            f.write(f"- `{tool_call['tool_name']}({tool_call['input']})`\n")
                            f.write(f"```\n{tool_call['output']}\n```\n\n")
                    else:
                        f.write("**Research (Tool Calls)**: None\n\n")

                    # Proposal
                    f.write("**Proposal**:\n")
                    f.write(f"- **Action**: {explorer_agent.proposed_action or 'None'}\n")
                    f.write(f"- **Confidence**: {explorer_agent.confidence or 0}/100\n")
                    f.write(f"- **Reason**: {explorer_agent.reason or 'N/A'}\n\n")
                    f.write("---\n\n")
                else:
                    f.write("### ExplorerAgent\n")
                    f.write("*No ExplorerAgent active this turn*\n\n")
                    f.write("---\n\n")

                # LoopDetectionAgent
                if loop_detection_agent:
                    f.write("### LoopDetectionAgent\n")
                    f.write(f"**Status**: {'Loop Detected' if loop_detection_agent.loop_detected else 'No Loop Detected'}\n")
                    if loop_detection_agent.loop_detected:
                        f.write(f"**Loop Type**: {loop_detection_agent.loop_type}\n")
                    f.write(f"**Current Location**: {loop_detection_agent.current_location}\n\n")

                    # Tool calls
                    if hasattr(loop_detection_agent, 'tool_calls_history') and loop_detection_agent.tool_calls_history:
                        f.write("**Research (Tool Calls)**:\n")
                        for tool_call in loop_detection_agent.tool_calls_history:
                            f.write(f"- `{tool_call['tool_name']}({tool_call['input']})`\n")
                            f.write(f"```\n{tool_call['output']}\n```\n\n")
                    else:
                        f.write("**Research (Tool Calls)**: None\n\n")

                    # Analysis
                    f.write("**Analysis**:\n")
                    f.write(f"- **Loop Detected**: {'Yes' if loop_detection_agent.loop_detected else 'No'}\n")
                    f.write(f"- **Confidence**: {loop_detection_agent.confidence or 0}/100\n")
                    if loop_detection_agent.loop_detected:
                        f.write(f"- **Proposed Action**: {loop_detection_agent.proposed_action}\n")
                        f.write(f"- **Reason**:\n```\n{loop_detection_agent.reason}\n```\n")
                    f.write("\n---\n\n")
                else:
                    f.write("### LoopDetectionAgent\n")
                    f.write("*No LoopDetectionAgent active this turn*\n\n")
                    f.write("---\n\n")

                # Decision Agent
                f.write("## Decision Agent\n\n")
                f.write("**Prompt**:\n")
                f.write(f"```\n{decision_prompt}\n```\n\n")

                f.write("**Decision**:\n")
                f.write(f"- **Command**: {player_command}\n")
                f.write(f"- **Reasoning**: {player_reasoning}\n\n")
                f.write("---\n\n")

                # Command sent
                f.write("## Command Sent to Game\n")
                f.write(f"```\n{player_command}\n```\n")

            self.logger.info(f"Turn report written to {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to write turn report: {e}")
            raise
