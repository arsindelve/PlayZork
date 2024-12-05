import asyncio
from zork_service import ZorkService


async def main():
    # Initialize the Zork service with a session ID
    service = ZorkService(session_id="123456")

    try:
        # Example turn: make an API call
        response = await service.play_turn(input_text="open door")
        # Display the API response
        service.display_response(response)
    except Exception as e:
        print(str(e))


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
