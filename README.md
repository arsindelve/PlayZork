# Teaching LLMs to Play Zork: Why ChatGPT is Like Mr. Meeseeks

If you're unfamiliar with Mr. Meeseeks from Rick and Morty, here's a quick rundown: when you summon a Meeseeks, they show up to help you accomplish a specific task. Once they complete it, they disappear. They're smart and capable—just don't give them anything too complex! However, they arrive with no context, no memory, and no understanding of your assignment beyond what you tell them. So, if you ask them to help you win a game they've never played before, be prepared to explain the rules and all the moves you've already made. And if you summon another Meeseeks, you'll have to repeat yourself entirely.

## ChatGPT and Other LLMs Are Like That

While large language models (LLMs) like ChatGPT can remember limited amounts of information during a single conversation, for the most part they don't have long-term memory across sessions. Imagine asking ChatGPT to help you solve an escape room: the first thing you'd need to do is describe the room, then you explain all the clues you've already found and how you've applied them. Once that's done, it might offer a sensible suggestion for what to try next. But "summon" ChatGPT again, and you'll need to repeat the process from scratch.

This year, I've been working on an AI-enhanced version of the 1980s text adventure game Zork. It's almost done and it's pretty cool. (Really!) Recently, I started wondering: what would it take to get ChatGPT to play and win the game on its own? Like the escape room, I'm confident it could come up with intelligent actions to try, like "look under the bed" or "press the red button." But, much like a Meeseeks, once the API call ends, it vanishes and forgets everything. Later, if it finds a key in a new room, it won't remember that it found a locked door 25 turns ago. It won't remember that we encountered a hungry cyclops and are searching for food. Without continuity of memory, it's unlikely to solve the game.

## The Experiment: Giving ChatGPT Memory

My solution? I'm going to programmatically give ChatGPT the memory it lacks. Upon each "summoning" (each API call), I'll remind the "Chat-Meeseeks" of all the cool stuff it found, the puzzles it hasn't solved yet, everything in its inventory, and more. The goal is to see if this added context can provide enough awareness for it to beat the game. Each new "summoning" will include a complete recap of everything that's happened so far. But I won't cheat - ChatGPT will be responsible for adding to it's own recap by indicating (using a predefined JSON data structure): "I pressed a yellow button and it did nothing" or "be careful in the maze." It will be a lot like Leonard's tattoos in Memento.

Will it work? I might hit the token limit, and the cost in credits could add up, but I'm excited to try!

## Getting Started

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Create a .env file with your OpenAI API key
cp .env.example .env
# Edit .env and add your actual API key

# Run the game (25 turns)
uv run python VersionTwo/main.py
```

## Project Structure

- **VersionTwo/** - Current Python implementation using LangChain
- **VersionOne/** - Earlier C# implementation (archived)

See [CLAUDE.md](./CLAUDE.md) for detailed architecture documentation.
