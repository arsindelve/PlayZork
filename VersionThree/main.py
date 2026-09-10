"""Minimal Opus-only baseline for text adventure games.

No agents, no memory layer, no langchain. Just a transcript, prompt
engineering, and the Anthropic SDK. Designed as the comparison point
for VersionTwo's elaborate scaffolding.
"""
import os
import uuid
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GAME_URL = "https://6kvs9n5pj4.execute-api.us-east-1.amazonaws.com/Prod/Planetfall"
MODEL = "claude-opus-4-7"
MAX_TURNS = 400
TARGET_SCORE = 80

SYSTEM_PROMPT = """You are playing a classic text adventure game.

Goal: reach a score of {target}.

Each turn you receive the game's state. Reply with EXACTLY ONE command —
no quotes, no markdown, no explanation, just the command on a single line.

IMPORTANT — NO TRAINING-DATA CHEATING:
This game may resemble a classic interactive fiction title you have seen in
your training data. You MUST NOT use that prior knowledge. Do not recall
walkthroughs, hint files, FAQs, solution maps, treasure locations, or puzzle
answers from prior exposure to any game.

Solve from in-game observation ONLY. React to what the game's text tells
you THIS run — locations as described, items as discovered, puzzles as
encountered. If a command suggests itself because "I remember the solution
is X," treat that as leakage and instead pick a command grounded in what
you have actually observed this session.

Examples of valid commands: NORTH, OPEN DOOR, TAKE LAMP, EXAMINE SIGN,
READ LEAFLET, UNLOCK CHEST WITH KEY, INVENTORY, LOOK.

Strategy:
- Examine everything. Small details matter.
- Read every sign, note, and book you find.
- Take items unless they are obviously immovable.
- Try combinations: USE X ON Y, PUT X IN Y, ATTACK X WITH Y.
- Track which directions and items lead nowhere — don't repeat dead ends.
- Heed warnings. If the game says somewhere is dangerous, don't go there
  unless you have a specific reason."""


def play_turn(client: httpx.Client, session_id: str, command: str) -> dict:
    r = client.post(GAME_URL, json={
        "Input": command, "SessionId": session_id, "NoGeneratedResponses": True,
    })
    r.raise_for_status()
    return r.json()


def format_state(state: dict) -> str:
    inv = state.get("inventory") or []
    return (
        f"Location: {state.get('locationName', 'Unknown')}\n"
        f"Score: {state.get('score', 0)}/{TARGET_SCORE}, Moves: {state.get('moves', 0)}\n"
        f"Inventory: {', '.join(inv) if inv else 'empty'}\n\n"
        f"{(state.get('response') or '').strip()}"
    )


def main() -> None:
    session_id = f"opus-baseline-{uuid.uuid4().hex[:8]}"
    anthropic = Anthropic()
    transcript: list[dict] = []

    print(f"Session: {session_id} | Model: {MODEL} | Target: {TARGET_SCORE}")

    with httpx.Client(timeout=30) as game:
        state = play_turn(game, session_id, "look")

        for turn in range(1, MAX_TURNS + 1):
            print(f"\n=== Turn {turn} | Score {state.get('score', 0)}/{TARGET_SCORE} | {state.get('locationName')} ===")
            print((state.get("response") or "").strip())

            if state.get("score", 0) >= TARGET_SCORE:
                print(f"\n*** WON in {turn - 1} turns. Final score: {state.get('score')} ***")
                return

            transcript.append({"role": "user", "content": format_state(state)})
            reply = anthropic.messages.create(
                model=MODEL,
                max_tokens=200,
                system=SYSTEM_PROMPT.format(target=TARGET_SCORE),
                messages=transcript,
            )
            command = reply.content[0].text.strip().splitlines()[0].strip()
            transcript.append({"role": "assistant", "content": command})

            print(f">> {command}")
            state = play_turn(game, session_id, command)

    print(f"\nReached {MAX_TURNS}-turn limit. Final score: {state.get('score', 0)}/{TARGET_SCORE}")


if __name__ == "__main__":
    main()
