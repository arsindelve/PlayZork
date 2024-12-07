import asyncio
from game_session import GameSession


async def main():
    # Initialize the GameSession with a session ID
    session = GameSession(session_id="1234567")

    # Example turn: player provides input
    await session.play()


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
