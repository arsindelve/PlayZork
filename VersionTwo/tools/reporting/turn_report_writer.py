"""Turn Report Writer - Generates detailed HTML reports for each turn"""
from pathlib import Path
from typing import List, Optional
import logging
import html


class TurnReportWriter:
    """Writes detailed turn reports to HTML files"""

    def __init__(self, logs_base_path: str = "logs/sessions"):
        """
        Initialize the report writer.

        Args:
            logs_base_path: Base directory for session logs
        """
        self.logs_base_path = logs_base_path
        self.logger = logging.getLogger(__name__)
        self.session_index_cache = {}  # Cache to track if we need to create header

    def _get_css(self) -> str:
        """Return CSS styles for the HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        header .meta {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .content {
            padding: 30px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section-title {
            font-size: 1.5em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        .subsection-title {
            font-size: 1.2em;
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .game-state {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 4px;
        }

        .game-state .stat {
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }

        .game-state .stat-label {
            font-weight: bold;
            color: #666;
        }

        .game-state .stat-value {
            color: #667eea;
            font-weight: bold;
        }

        .game-output {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            margin-top: 15px;
            border-left: 4px solid #667eea;
        }

        .context-box {
            background: #fffbeb;
            border: 1px solid #fbbf24;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .inventory-list {
            list-style: none;
            padding-left: 0;
        }

        .inventory-list li {
            padding: 8px;
            background: #f0fdf4;
            border-left: 3px solid #10b981;
            margin-bottom: 5px;
            border-radius: 3px;
        }

        .inventory-list li:before {
            content: "📦 ";
            margin-right: 8px;
        }

        .agent-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .agent-card.loop-detected {
            border-left: 5px solid #ef4444;
            background: #fef2f2;
        }

        .agent-card.interaction {
            border-left: 5px solid #8b5cf6;
        }

        .agent-card.explorer {
            border-left: 5px solid #3b82f6;
        }

        .agent-card.issue {
            border-left: 5px solid #f59e0b;
        }

        .agent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .agent-title {
            font-size: 1.1em;
            font-weight: bold;
            color: #1f2937;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }

        .badge-danger {
            background: #fee2e2;
            color: #991b1b;
        }

        .badge-success {
            background: #d1fae5;
            color: #065f46;
        }

        .badge-warning {
            background: #fef3c7;
            color: #92400e;
        }

        .badge-info {
            background: #dbeafe;
            color: #1e40af;
        }

        .badge-purple {
            background: #ede9fe;
            color: #5b21b6;
        }

        .metric {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 8px;
        }

        .metric-label {
            color: #6b7280;
            font-size: 0.9em;
        }

        .metric-value {
            font-weight: bold;
            color: #1f2937;
        }

        .tool-calls {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 15px;
            margin-top: 10px;
        }

        .tool-call {
            margin-bottom: 10px;
        }

        .tool-name {
            color: #8b5cf6;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }

        .tool-output {
            background: #1f2937;
            color: #e5e7eb;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin-top: 5px;
            white-space: pre-wrap;
            overflow-x: auto;
        }

        .proposal {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
        }

        .proposal-action {
            font-size: 1.1em;
            font-weight: bold;
            color: #1e40af;
            margin-bottom: 8px;
        }

        .decision-box {
            background: #ecfdf5;
            border: 2px solid #10b981;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }

        .decision-command {
            font-size: 1.3em;
            font-weight: bold;
            color: #065f46;
            font-family: 'Courier New', monospace;
            background: white;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #10b981;
        }

        .decision-reasoning {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 4px;
        }

        pre {
            background: #1f2937;
            color: #e5e7eb;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }

        code {
            background: #f3f4f6;
            color: #be123c;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .empty-state {
            text-align: center;
            padding: 40px;
            color: #9ca3af;
            font-style: italic;
        }

        .confidence-bar {
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #3b82f6 50%, #8b5cf6 100%);
            transition: width 0.3s ease;
        }

        header .back-link {
            color: white;
            text-decoration: none;
            opacity: 0.9;
            font-size: 0.9em;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: opacity 0.2s, transform 0.2s;
            padding: 8px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
        }

        header .back-link:hover {
            opacity: 1;
            transform: translateX(-4px);
            background: rgba(255, 255, 255, 0.15);
        }

        header .back-link .arrow {
            font-size: 1.2em;
        }
        """

    def _escape(self, text: str) -> str:
        """Escape HTML special characters"""
        if text is None:
            return ""
        return html.escape(str(text))

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
        interaction_agent,
        decision_prompt: str,
        decision,
        recent_history: Optional[str] = None,
        complete_history: Optional[str] = None,
        current_inventory: Optional[List[str]] = None
    ):
        """
        Write a detailed HTML report for a single turn.

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
            interaction_agent: InteractionAgent object or None
            decision_prompt: Formatted decision prompt
            decision: AdventurerResponse object
            recent_history: Recent history summary from tools
            complete_history: Complete history summary from tools
            current_inventory: List of items currently in inventory
        """
        # Create session directory
        session_dir = Path(self.logs_base_path) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create HTML file
        report_path = session_dir / f"Turn-{turn_number}.html"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                # HTML header
                f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session {self._escape(session_id)} - Turn {turn_number}</title>
    <style>
{self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div style="text-align: left; margin-bottom: 20px;">
                <a href="index.html" class="back-link">
                    <span class="arrow">←</span> Back to Session Index
                </a>
            </div>
            <h1>Session {self._escape(session_id)} - Turn {turn_number}</h1>
            <div class="meta">
                <span>Score: {score}</span> |
                <span>Moves: {moves}</span> |
                <span>Location: {self._escape(location)}</span>
            </div>
        </header>

        <div class="content">
""")

                # Game State Section
                f.write(f"""
            <section class="section">
                <h2 class="section-title">Game State</h2>
                <div class="game-state">
                    <div class="stat">
                        <span class="stat-label">Location:</span>
                        <span class="stat-value">{self._escape(location)}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Score:</span>
                        <span class="stat-value">{score}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Moves:</span>
                        <span class="stat-value">{moves}</span>
                    </div>
                </div>
                <div class="game-output">{self._escape(game_response)}</div>
            </section>
""")

                # Context Section
                f.write("""
            <section class="section">
                <h2 class="section-title">Context</h2>
""")

                # Recent History
                f.write("""
                <h3 class="subsection-title">Recent History</h3>
                <div class="context-box">
""")
                if recent_history:
                    f.write(f"                    <p>{self._escape(recent_history)}</p>\n")
                else:
                    f.write('                    <p class="empty-state">No recent history available</p>\n')
                f.write("                </div>\n")

                # Complete History
                f.write("""
                <h3 class="subsection-title">Complete History</h3>
                <div class="context-box">
""")
                if complete_history:
                    f.write(f"                    <p>{self._escape(complete_history)}</p>\n")
                else:
                    f.write('                    <p class="empty-state">No complete history available</p>\n')
                f.write("                </div>\n")

                # Inventory
                f.write("""
                <h3 class="subsection-title">Current Inventory</h3>
""")
                if current_inventory:
                    f.write('                <ul class="inventory-list">\n')
                    for item in current_inventory:
                        f.write(f'                    <li>{self._escape(item)}</li>\n')
                    f.write('                </ul>\n')
                else:
                    f.write('                <p class="empty-state">Inventory is empty</p>\n')

                f.write("            </section>\n")

                # Agent Analysis Section
                f.write("""
            <section class="section">
                <h2 class="section-title">Agent Analysis</h2>
""")

                # LoopDetectionAgent
                if loop_detection_agent:
                    loop_class = "agent-card loop-detected" if loop_detection_agent.loop_detected else "agent-card"
                    f.write(f'                <div class="{loop_class}">\n')
                    f.write('                    <div class="agent-header">\n')
                    f.write('                        <div class="agent-title">LoopDetectionAgent</div>\n')

                    if loop_detection_agent.loop_detected:
                        f.write(f'                        <span class="badge badge-danger">⚠️ LOOP DETECTED</span>\n')
                    else:
                        f.write('                        <span class="badge badge-success">✓ No Loop</span>\n')

                    f.write('                    </div>\n')

                    # Metrics
                    f.write('                    <div>\n')
                    f.write(f'                        <div class="metric"><span class="metric-label">Status:</span> <span class="metric-value">{"Loop Detected" if loop_detection_agent.loop_detected else "No Loop Detected"}</span></div>\n')
                    if loop_detection_agent.loop_detected:
                        f.write(f'                        <div class="metric"><span class="metric-label">Type:</span> <span class="metric-value">{self._escape(loop_detection_agent.loop_type)}</span></div>\n')
                    f.write(f'                        <div class="metric"><span class="metric-label">Confidence:</span> <span class="metric-value">{loop_detection_agent.confidence or 0}/100</span></div>\n')
                    f.write('                    </div>\n')

                    # Tool calls
                    if hasattr(loop_detection_agent, 'tool_calls_history') and loop_detection_agent.tool_calls_history:
                        f.write('                    <div class="tool-calls">\n')
                        f.write('                        <strong>Research (Tool Calls):</strong>\n')
                        for tool_call in loop_detection_agent.tool_calls_history:
                            f.write(f'                        <div class="tool-call">\n')
                            f.write(f'                            <div class="tool-name">{self._escape(tool_call["tool_name"])}({self._escape(tool_call["input"])})</div>\n')
                            f.write(f'                            <div class="tool-output">{self._escape(tool_call["output"])}</div>\n')
                            f.write('                        </div>\n')
                        f.write('                    </div>\n')

                    # Proposal
                    if loop_detection_agent.loop_detected:
                        f.write('                    <div class="proposal">\n')
                        f.write(f'                        <div class="proposal-action">Proposed Action: {self._escape(loop_detection_agent.proposed_action)}</div>\n')
                        if loop_detection_agent.reason:
                            f.write(f'                        <div><strong>Reason:</strong> {self._escape(loop_detection_agent.reason)}</div>\n')
                        f.write('                    </div>\n')

                    f.write('                </div>\n')

                # IssueAgents
                if issue_agents:
                    for i, agent in enumerate(issue_agents, 1):
                        f.write('                <div class="agent-card issue">\n')
                        f.write('                    <div class="agent-header">\n')
                        f.write(f'                        <div class="agent-title">IssueAgent #{i}: {self._escape(agent.issue_content)}</div>\n')
                        f.write(f'                        <span class="badge badge-warning">Importance: {agent.importance}/1000</span>\n')
                        f.write('                    </div>\n')

                        f.write('                    <div>\n')
                        f.write(f'                        <div class="metric"><span class="metric-label">Location:</span> <span class="metric-value">{self._escape(agent.location)}</span></div>\n')
                        f.write(f'                        <div class="metric"><span class="metric-label">Confidence:</span> <span class="metric-value">{agent.confidence or 0}/100</span></div>\n')
                        f.write('                    </div>\n')

                        # Tool calls
                        if hasattr(agent, 'tool_calls_history') and agent.tool_calls_history:
                            f.write('                    <div class="tool-calls">\n')
                            f.write('                        <strong>Research (Tool Calls):</strong>\n')
                            for tool_call in agent.tool_calls_history:
                                f.write(f'                        <div class="tool-call">\n')
                                f.write(f'                            <div class="tool-name">{self._escape(tool_call["tool_name"])}({self._escape(tool_call["input"])})</div>\n')
                                f.write(f'                            <div class="tool-output">{self._escape(tool_call["output"])}</div>\n')
                                f.write('                        </div>\n')
                            f.write('                    </div>\n')

                        # Proposal
                        f.write('                    <div class="proposal">\n')
                        f.write(f'                        <div class="proposal-action">Action: {self._escape(agent.proposed_action or "None")}</div>\n')
                        if agent.reason:
                            f.write(f'                        <div><strong>Reason:</strong> {self._escape(agent.reason)}</div>\n')
                        f.write('                    </div>\n')

                        f.write('                </div>\n')

                # InteractionAgent
                if interaction_agent:
                    f.write('                <div class="agent-card interaction">\n')
                    f.write('                    <div class="agent-header">\n')
                    f.write('                        <div class="agent-title">InteractionAgent</div>\n')

                    if interaction_agent.confidence > 0:
                        f.write('                        <span class="badge badge-purple">Interaction Detected</span>\n')
                    else:
                        f.write('                        <span class="badge badge-info">No Interactions</span>\n')

                    f.write('                    </div>\n')

                    f.write('                    <div>\n')
                    f.write(f'                        <div class="metric"><span class="metric-label">Confidence:</span> <span class="metric-value">{interaction_agent.confidence or 0}/100</span></div>\n')
                    f.write('                    </div>\n')

                    # Tool calls
                    if hasattr(interaction_agent, 'tool_calls_history') and interaction_agent.tool_calls_history:
                        f.write('                    <div class="tool-calls">\n')
                        f.write('                        <strong>Research (Tool Calls):</strong>\n')
                        for tool_call in interaction_agent.tool_calls_history:
                            f.write(f'                        <div class="tool-call">\n')
                            f.write(f'                            <div class="tool-name">{self._escape(tool_call["tool_name"])}({self._escape(tool_call["input"])})</div>\n')
                            f.write(f'                            <div class="tool-output">{self._escape(tool_call["output"])}</div>\n')
                            f.write('                        </div>\n')
                        f.write('                    </div>\n')

                    # Proposal
                    if interaction_agent.confidence > 0:
                        f.write('                    <div class="proposal">\n')
                        f.write(f'                        <div class="proposal-action">Action: {self._escape(interaction_agent.proposed_action)}</div>\n')
                        if interaction_agent.detected_objects:
                            f.write(f'                        <div><strong>Objects:</strong> {self._escape(", ".join(interaction_agent.detected_objects))}</div>\n')
                        if interaction_agent.inventory_items:
                            f.write(f'                        <div><strong>Using:</strong> {self._escape(", ".join(interaction_agent.inventory_items))}</div>\n')
                        if interaction_agent.reason:
                            f.write(f'                        <div><strong>Reason:</strong> {self._escape(interaction_agent.reason)}</div>\n')
                        f.write('                    </div>\n')

                    f.write('                </div>\n')

                # ExplorerAgent
                if explorer_agent:
                    f.write('                <div class="agent-card explorer">\n')
                    f.write('                    <div class="agent-header">\n')
                    f.write('                        <div class="agent-title">ExplorerAgent</div>\n')
                    f.write(f'                        <span class="badge badge-info">Exploring</span>\n')
                    f.write('                    </div>\n')

                    f.write('                    <div>\n')
                    f.write(f'                        <div class="metric"><span class="metric-label">Best Direction:</span> <span class="metric-value">{self._escape(explorer_agent.best_direction)}</span></div>\n')
                    f.write(f'                        <div class="metric"><span class="metric-label">Unexplored:</span> <span class="metric-value">{len(explorer_agent.unexplored_directions)} directions</span></div>\n')
                    f.write(f'                        <div class="metric"><span class="metric-label">Confidence:</span> <span class="metric-value">{explorer_agent.confidence or 0}/100</span></div>\n')
                    f.write('                    </div>\n')

                    # Tool calls
                    if hasattr(explorer_agent, 'tool_calls_history') and explorer_agent.tool_calls_history:
                        f.write('                    <div class="tool-calls">\n')
                        f.write('                        <strong>Research (Tool Calls):</strong>\n')
                        for tool_call in explorer_agent.tool_calls_history:
                            f.write(f'                        <div class="tool-call">\n')
                            f.write(f'                            <div class="tool-name">{self._escape(tool_call["tool_name"])}({self._escape(tool_call["input"])})</div>\n')
                            f.write(f'                            <div class="tool-output">{self._escape(tool_call["output"])}</div>\n')
                            f.write('                        </div>\n')
                        f.write('                    </div>\n')

                    # Proposal
                    f.write('                    <div class="proposal">\n')
                    f.write(f'                        <div class="proposal-action">Action: {self._escape(explorer_agent.proposed_action or "None")}</div>\n')
                    if explorer_agent.reason:
                        f.write(f'                        <div><strong>Reason:</strong> {self._escape(explorer_agent.reason)}</div>\n')
                    f.write('                    </div>\n')

                    f.write('                </div>\n')

                f.write("            </section>\n")

                # Decision Section
                f.write("""
            <section class="section">
                <h2 class="section-title">Decision Agent</h2>
                <div class="decision-box">
                    <h3>Final Decision</h3>
                    <div class="decision-command">""" + self._escape(player_command) + """</div>
                    <div class="decision-reasoning">
                        <strong>Reasoning:</strong><br>
                        """ + self._escape(player_reasoning) + """
                    </div>
                </div>

                <h3 class="subsection-title">Decision Prompt</h3>
                <pre>""" + self._escape(decision_prompt) + """</pre>
            </section>
""")

                # Footer
                f.write("""
        </div>
    </div>
</body>
</html>
""")

            self.logger.info(f"Turn report written to {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to write turn report: {e}")
            raise

    def update_session_index(
        self,
        session_id: str,
        turn_number: int,
        location: str,
        score: int,
        moves: int,
        player_command: str,
        game_response: str
    ):
        """
        Update the master session index file with the latest turn.
        Appends to existing file or creates new one if needed.

        Args:
            session_id: Game session ID
            turn_number: Turn number
            location: Current location
            score: Current score
            moves: Current move count
            player_command: Command sent to game
            game_response: Game's response text
        """
        session_dir = Path(self.logs_base_path) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        index_path = session_dir / "index.html"
        is_new_file = not index_path.exists()

        try:
            if is_new_file:
                # Create new index file with header
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write(self._get_session_index_header(session_id))
                    f.write(self._get_turn_entry(turn_number, location, score, moves, player_command, game_response))
                    f.write(self._get_session_index_footer())
            else:
                # Append to existing file
                # Read existing content
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find the closing tags and insert before them
                footer_pos = content.rfind('</div>\n    </div>\n</body>')
                if footer_pos == -1:
                    # Fallback: just append before </body>
                    footer_pos = content.rfind('</body>')

                if footer_pos != -1:
                    # Insert the new turn entry before the footer
                    new_content = (
                        content[:footer_pos] +
                        self._get_turn_entry(turn_number, location, score, moves, player_command, game_response) +
                        content[footer_pos:]
                    )

                    # Write back
                    with open(index_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

            self.logger.info(f"Session index updated: {index_path}")

        except Exception as e:
            self.logger.error(f"Failed to update session index: {e}")
            raise

    def _get_session_index_header(self, session_id: str) -> str:
        """Generate the header for the session index file"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session {self._escape(session_id)} - Game Progress</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .content {{
            padding: 30px;
        }}

        .turn-entry {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .turn-entry:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .turn-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f3f4f6;
        }}

        .turn-number {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
        }}

        .turn-stats {{
            display: flex;
            gap: 20px;
        }}

        .stat {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .stat-label {{
            font-size: 0.75em;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-value {{
            font-size: 1.1em;
            font-weight: bold;
            color: #1f2937;
        }}

        .turn-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 15px;
        }}

        .input-section,
        .output-section {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
        }}

        .input-section {{
            border-left: 4px solid #3b82f6;
        }}

        .output-section {{
            border-left: 4px solid #10b981;
        }}

        .section-title {{
            font-size: 0.85em;
            font-weight: bold;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }}

        .command {{
            background: #1f2937;
            color: #10b981;
            padding: 12px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            font-weight: bold;
        }}

        .response {{
            color: #1f2937;
            white-space: pre-wrap;
            font-size: 0.95em;
            line-height: 1.6;
        }}

        .turn-link {{
            text-align: right;
        }}

        .turn-link a {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .turn-link a:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}

        .location-badge {{
            display: inline-block;
            background: #eff6ff;
            color: #1e40af;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: 600;
            margin-left: 10px;
        }}

        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #9ca3af;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Session {self._escape(session_id)}</h1>
            <p>Game Progress & Turn History</p>
        </header>

        <div class="content">
"""

    def _get_turn_entry(
        self,
        turn_number: int,
        location: str,
        score: int,
        moves: int,
        player_command: str,
        game_response: str
    ) -> str:
        """Generate HTML for a single turn entry"""
        return f"""
            <div class="turn-entry">
                <div class="turn-header">
                    <div>
                        <span class="turn-number">Turn {turn_number}</span>
                        <span class="location-badge">📍 {self._escape(location)}</span>
                    </div>
                    <div class="turn-stats">
                        <div class="stat">
                            <span class="stat-label">Score</span>
                            <span class="stat-value">{score}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Moves</span>
                            <span class="stat-value">{moves}</span>
                        </div>
                    </div>
                </div>

                <div class="turn-content">
                    <div class="input-section">
                        <div class="section-title">Player Input</div>
                        <div class="command">&gt; {self._escape(player_command)}</div>
                    </div>
                    <div class="output-section">
                        <div class="section-title">Game Response</div>
                        <div class="response">{self._escape(game_response)}</div>
                    </div>
                </div>

                <div class="turn-link">
                    <a href="Turn-{turn_number}.html">View Detailed Analysis →</a>
                </div>
            </div>
"""

    def _get_session_index_footer(self) -> str:
        """Generate the footer for the session index file"""
        return """
        </div>
    </div>
</body>
</html>
"""
