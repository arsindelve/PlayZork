import logging
import os
from datetime import datetime


class GameLogger:
    """
    Centralized logging for the game session.
    Creates a new log file for each session and captures all important events.
    """

    _instance = None

    def __init__(self, session_id: str):
        """
        Initialize the game logger for a specific session.

        Args:
            session_id: The session identifier
        """
        self.session_id = session_id
        self.log_dir = "logs"

        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log filename based on session_id
        self.log_file = os.path.join(self.log_dir, f"game_{session_id}.log")

        # Clear/reset the log file for this session
        with open(self.log_file, 'w') as f:
            f.write(f"=== Game Session Log ===\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # Set up logger
        self.logger = logging.getLogger(f"GameSession_{session_id}")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)

        # Format: timestamp - level - message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.logger.info("Logger initialized")

    def log_turn_start(self, turn_number: int, command: str):
        """Log the start of a turn"""
        self.logger.info("=" * 80)
        self.logger.info(f"TURN {turn_number} START - Command: {command}")
        self.logger.info("=" * 80)

    def log_game_response(self, response: str):
        """Log the game's response"""
        self.logger.info(f"GAME RESPONSE:\n{response}")

    def log_research_start(self):
        """Log the start of research phase"""
        self.logger.info("\n--- RESEARCH PHASE START ---")

    def log_tool_call(self, tool_name: str, args: dict):
        """Log a tool call"""
        self.logger.info(f"TOOL CALL: {tool_name} with args: {args}")

    def log_tool_result(self, tool_name: str, result: str):
        """Log a tool result"""
        self.logger.debug(f"TOOL RESULT ({tool_name}):\n{result[:500]}...")  # Truncate long results

    def log_research_complete(self, context: str):
        """Log research phase completion"""
        self.logger.info(f"RESEARCH COMPLETE:\n{context}")
        self.logger.info("--- RESEARCH PHASE END ---\n")

    def log_decision_start(self, score: int, location: str, moves: int):
        """Log the start of decision phase"""
        self.logger.info("\n--- DECISION PHASE START ---")
        self.logger.info(f"Score: {score}, Location: {location}, Moves: {moves}")

    def log_decision(self, command: str, reason: str):
        """Log the agent's decision"""
        self.logger.info(f"DECISION: {command}")
        self.logger.info(f"REASON: {reason}")

    def log_summary_update(self, summary: str):
        """Log summary update"""
        self.logger.debug(f"SUMMARY UPDATED:\n{summary[:300]}...")  # Truncate

    def log_error(self, error: str):
        """Log an error"""
        self.logger.error(f"ERROR: {error}")

    def log_session_end(self, final_score: int, final_moves: int):
        """Log session end"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"SESSION END - Final Score: {final_score}, Final Moves: {final_moves}")
        self.logger.info("=" * 80)

    @classmethod
    def get_instance(cls, session_id: str = None):
        """Get or create the singleton logger instance"""
        if cls._instance is None or (session_id and cls._instance.session_id != session_id):
            if session_id is None:
                raise ValueError("session_id required for first initialization")
            cls._instance = cls(session_id)
        return cls._instance
