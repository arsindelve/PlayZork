from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from typing import List, Tuple


class DisplayManager:
    """
    Manages the Rich console display with two main regions:
    1. Game I/O - Shows the conversation between game and agent
    2. Summary - Shows the current game summary
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Split screen into two columns: I/O (left) and Summary (right)
        self.layout.split_row(
            Layout(name="game_io", ratio=2),
            Layout(name="summary", ratio=1)
        )

        # Initialize content
        self.game_turns: List[Tuple[str, str, str]] = []  # (location, game_text, command)
        self.current_summary = "Game has not started yet."
        self.current_location = "Unknown"
        self.current_score = 0
        self.current_moves = 0

        # Initialize display
        self._update_display()

        # Start live display
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
        self.live.start()

    def add_turn(self, location: str, game_text: str, command: str, score: int, moves: int):
        """
        Add a new turn to the game I/O display

        Args:
            location: Current location name
            game_text: Game's response text
            command: Command issued by agent
            score: Current score
            moves: Current move count
        """
        self.game_turns.append((location, game_text, command))
        self.current_location = location
        self.current_score = score
        self.current_moves = moves

        # Keep only last 10 turns to prevent overflow
        if len(self.game_turns) > 10:
            self.game_turns = self.game_turns[-10:]

        self._update_display()

    def update_summary(self, summary: str):
        """
        Update the game summary display

        Args:
            summary: New summary text
        """
        self.current_summary = summary
        self._update_display()

    def _update_display(self):
        """Update both regions of the display"""
        # Update Game I/O region
        io_content = self._build_io_content()
        self.layout["game_io"].update(Panel(
            io_content,
            title=f"[bold cyan]Game I/O[/bold cyan] - Score: [yellow]{self.current_score}[/yellow] | Moves: [green]{self.current_moves}[/green]",
            subtitle="[dim]Press Ctrl+C to quit[/dim]",
            border_style="cyan"
        ))

        # Update Summary region
        summary_content = self._build_summary_content()
        self.layout["summary"].update(Panel(
            summary_content,
            title="[bold magenta]Game Summary[/bold magenta]",
            border_style="magenta"
        ))

    def _build_io_content(self) -> Text:
        """Build the game I/O content from turns"""
        content = Text()

        if not self.game_turns:
            content.append("Waiting for game to start...", style="dim")
            return content

        for i, (location, game_text, command) in enumerate(self.game_turns):
            # Location header
            content.append(f"\n[{location}]\n", style="bold cyan")

            # Agent command (what was sent to the game)
            content.append("> ", style="bold green")
            content.append(f"{command}\n\n", style="green")

            # Game response (to the command above)
            content.append(game_text, style="white")
            content.append("\n")

            # Separator between turns (except last)
            if i < len(self.game_turns) - 1:
                content.append("─" * 50 + "\n", style="dim")

        return content

    def _build_summary_content(self) -> Text:
        """Build the summary content"""
        content = Text()

        # Current location highlight
        content.append("Current Location:\n", style="bold white")
        content.append(f"{self.current_location}\n\n", style="cyan")

        # Summary
        content.append("History Summary:\n", style="bold white")
        content.append(self.current_summary, style="white")

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
