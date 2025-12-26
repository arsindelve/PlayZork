from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from typing import List, Tuple


class DisplayManager:
    """
    Manages the Rich console display with four main regions:
    1. Game I/O - Shows the conversation between game and agent
    2. Summary - Shows the current game summary
    3. Memories - Shows flagged important memories
    4. Map - Shows discovered location transitions
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Split screen into two columns: I/O (left) and Info (right)
        self.layout.split_row(
            Layout(name="game_io", ratio=1),
            Layout(name="info_panel", ratio=1)
        )

        # Split the right column into three rows: Summary (top), Memories (middle), Map (bottom)
        self.layout["info_panel"].split_column(
            Layout(name="summary", ratio=1),
            Layout(name="memories", ratio=1),
            Layout(name="map", ratio=1)
        )

        # Initialize content
        self.game_turns: List[Tuple[str, str, str, str, list, str]] = []  # (location, game_text, command, reasoning, closed_issues, new_issue)
        self.current_summary = "Game has not started yet."
        self.long_running_summary = "Game has not started yet."
        self.current_memories = "No memories recorded yet."
        self.current_map = "No map data yet."
        self.current_location = "Unknown"
        self.current_score = 0
        self.current_moves = 0

        # Initialize display
        self._update_display()

        # Start live display
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
        self.live.start()

    def add_turn(self, location: str, game_text: str, command: str, score: int, moves: int, reasoning: str = "", closed_issues: list = None, new_issue: str = None):
        """
        Add a new turn to the game I/O display

        Args:
            location: Current location name
            game_text: Game's response text
            command: Command issued by agent
            score: Current score
            moves: Current move count
            reasoning: Decision Agent's reasoning for this command
            closed_issues: List of issues that were resolved this turn
            new_issue: New issue identified by Observer this turn
        """
        self.game_turns.append((location, game_text, command, reasoning, closed_issues or [], new_issue))
        self.current_location = location
        self.current_score = score
        self.current_moves = moves

        # Keep only last 10 turns to prevent overflow
        if len(self.game_turns) > 10:
            self.game_turns = self.game_turns[-10:]

        self._update_display()

    def update_summary(self, summary: str, long_running_summary: str = None):
        """
        Update the game summary display

        Args:
            summary: New recent summary text (last 15 turns)
            long_running_summary: New comprehensive summary text (optional)
        """
        self.current_summary = summary
        if long_running_summary is not None:
            self.long_running_summary = long_running_summary
        self._update_display()

    def update_memories(self, memories_text: str):
        """
        Update the memories display

        Args:
            memories_text: Formatted memories text
        """
        self.current_memories = memories_text
        self._update_display()

    def update_map(self, map_text: str):
        """
        Update the map display

        Args:
            map_text: Formatted map transitions text
        """
        self.current_map = map_text
        self._update_display()

    def update_agents(self, issue_agents: list, explorer_agent):
        """
        Update the agents display (formats internally)

        Args:
            issue_agents: List of IssueAgent objects
            explorer_agent: ExplorerAgent object or None
        """
        from tools.agent_graph import ExplorerAgent

        all_agents = issue_agents + ([explorer_agent] if explorer_agent else [])

        if all_agents:
            # Sort by confidence descending
            all_agents.sort(key=lambda a: a.confidence if a.confidence else 0, reverse=True)

            agents_text = ""
            for i, agent in enumerate(all_agents[:10], 1):  # Show top 10
                # Determine type and display accordingly
                if isinstance(agent, ExplorerAgent):
                    agents_text += f"{i}. [EXPLORE] {agent.best_direction} from {agent.current_location}\n"
                    agents_text += f"   Unexplored: {len(agent.unexplored_directions)} total"
                    if agent.mentioned_directions:
                        agents_text += f" (Mentioned: {', '.join(agent.mentioned_directions)})"
                    agents_text += "\n"
                else:  # IssueAgent
                    agents_text += f"{i}. [{agent.importance}/1000] {agent.issue_content}\n"
                    agents_text += f"   Turn {agent.turn_number} @ {agent.location}\n"

                # Show proposal (same format for both)
                if agent.proposed_action and agent.confidence is not None:
                    agents_text += f"   > Proposal: {agent.proposed_action}\n"
                    if agent.reason:
                        agents_text += f"   > Reason: {agent.reason}\n"
                    agents_text += f"   > Confidence: {agent.confidence}/100\n"
                else:
                    agents_text += f"   > Proposal: (pending)\n"

                agents_text += "\n"

            self.current_memories = agents_text.strip()
        else:
            self.current_memories = "No agents active."

        self._update_display()

    def update_map_from_transitions(self, transitions: list):
        """
        Update the map display from transition objects (formats internally)

        Args:
            transitions: List of LocationTransition objects
        """
        if transitions:
            map_text = ""
            for trans in transitions:
                map_text += f"{trans.from_location} --[{trans.direction}]--> {trans.to_location} (T{trans.turn_discovered})\n"
            self.current_map = map_text.strip()
        else:
            self.current_map = "No map data yet."

        self._update_display()

    def _update_display(self):
        """Update all four regions of the display"""
        # Update Game I/O region
        io_content = self._build_io_content()
        # Keep only the last 5 turns to prevent overflow (scrolling doesn't work well in Rich panels)
        # Older turns are still available in summaries
        self.layout["game_io"].update(Panel(
            io_content,
            title=f"[bold cyan]Game I/O (Last 5 Turns)[/bold cyan] - Score: [yellow]{self.current_score}[/yellow] | Moves: [green]{self.current_moves}[/green]",
            subtitle="[dim]Press Ctrl+C to quit[/dim]",
            border_style="cyan"
        ))

        # Update Summary region
        summary_content = self._build_summary_content()
        self.layout["summary"].update(Panel(
            summary_content,
            title="[bold magenta]History Summary[/bold magenta]",
            border_style="magenta"
        ))

        # Update Memories region
        memories_content = self._build_memories_content()
        self.layout["memories"].update(Panel(
            memories_content,
            title="[bold yellow]Issues / Puzzles / Obstacles[/bold yellow]",
            border_style="yellow"
        ))

        # Update Map region
        map_content = self._build_map_content()
        self.layout["map"].update(Panel(
            map_content,
            title="[bold green]Map - Location Transitions[/bold green]",
            border_style="green"
        ))

    def _build_io_content(self) -> Text:
        """Build the game I/O content from turns"""
        content = Text()

        if not self.game_turns:
            content.append("Waiting for game to start...", style="dim")
            return content

        # Show only last 5 turns to prevent overflow (older turns available in summary)
        recent_turns = self.game_turns[-5:]

        for i, (location, game_text, command, reasoning, closed_issues, new_issue) in enumerate(recent_turns):
            # Location header
            content.append(f"\n[{location}]\n", style="bold cyan")

            # Decision Agent reasoning (shown BEFORE command)
            if reasoning:
                content.append("🤖 Decision: ", style="bold yellow")
                content.append(f"{reasoning}\n\n", style="yellow")

            # Agent command (what was sent to the game)
            content.append("> ", style="bold green")
            content.append(f"{command}\n\n", style="green")

            # Game response (to the command above)
            content.append(game_text, style="white")
            content.append("\n")

            # Closed issues (shown AFTER game response, in red/orange)
            if closed_issues:
                content.append("\n🔒 Issues Resolved:\n", style="bold red")
                for issue in closed_issues:
                    content.append(f"  ✓ {issue}\n", style="red")

            # New issue identified (shown AFTER game response, in bright green)
            if new_issue:
                content.append("\n🔍 New Issue Identified:\n", style="bold bright_green")
                content.append(f"  → {new_issue}\n", style="bright_green")

            # Separator between turns (except last)
            if i < len(recent_turns) - 1:
                content.append("─" * 50 + "\n", style="dim")

        return content

    def _build_summary_content(self) -> Text:
        """Build the summary content"""
        content = Text()

        # Current location highlight
        content.append("Location: ", style="bold white")
        content.append(f"{self.current_location}\n\n", style="cyan")

        # Recent Summary (last 15 turns)
        content.append("Recent:\n", style="bold white")
        content.append(self.current_summary, style="white")
        content.append("\n", style="white")

        # Separator
        content.append("─" * 40 + "\n", style="dim")

        # Long-Running Summary (everything)
        content.append("Complete:\n", style="bold white")
        content.append(self.long_running_summary, style="white")

        return content

    def _build_memories_content(self) -> Text:
        """Build the memories content"""
        content = Text()

        content.append(self.current_memories, style="yellow")

        return content

    def _build_map_content(self) -> Text:
        """Build the map content"""
        content = Text()

        content.append(self.current_map, style="green")

        return content

    def display_research(self, message: str):
        """
        Temporarily display research phase information

        Args:
            message: Research message to display
        """
        # Could add a temporary overlay or status message
        # For now, we'll just let it flow through normal output
        pass

    def stop(self):
        """Stop the live display"""
        if self.live:
            self.live.stop()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
