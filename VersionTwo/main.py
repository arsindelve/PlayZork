import asyncio
import sys
from dotenv import load_dotenv
from game_session import GameSession

# Load environment variables from .env file
load_dotenv()

async def main():
    # Initialize the GameSession with a session ID
    session = GameSession(session_id="v17")

    # Example turn: player provides input
    await session.play()


# Run the async main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Clean exit without traceback
        print("\n\n🎮 Game interrupted. Goodbye!")
        print("Thanks for playing!")
        sys.exit(0)
