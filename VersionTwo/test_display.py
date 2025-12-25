"""
Quick test of the Rich display manager
"""
import asyncio
from dotenv import load_dotenv
from game_session import GameSession

# Load environment variables
load_dotenv()


async def test_display():
    """Run a short game session to test the display"""
    session = GameSession(session_id="test-123")

    # Patch the play method to only run 3 turns
    original_play = session.play

    async def short_play():
        from display_manager import DisplayManager

        with DisplayManager() as display:
            try:
                # Initialize the game state
                await session.zork_service.play_turn("verbose")
                adventurer_response = await session._GameSession__play_turn("look", display)

                # Run only 3 turns
                for count in range(1, 4):
                    adventurer_response = await session._GameSession__play_turn(adventurer_response, display)
                    await asyncio.sleep(2)  # Pause to see the display update

                # Wait before closing
                await asyncio.sleep(5)

            except Exception as e:
                display.stop()
                print(f"An error occurred during gameplay: {e}")
                import traceback
                traceback.print_exc()

    await short_play()


if __name__ == "__main__":
    asyncio.run(test_display())
