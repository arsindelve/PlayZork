# Status: v0.1-arxiv (2025-12-25)

## What Works

• Two-phase agent architecture (research → decision)
• History toolkit with tool-based access
• Memory toolkit with importance scoring
• Llama 3.3 local model for all reasoning
• Tool calling functional (get_recent_turns, get_full_summary, memory tools)
• Structured output enforced (AdventurerResponse schema)
• State persists across turns (history, memory)
• Dual summary system (recent + long-running)
• Rich terminal display with live updates
• 100% local except Zork API (Ollama via IPv6 to Mac host)

## What Does NOT Work

• Does not solve Zork (gets stuck early)
• Memory tools never called by agent
• Research agent shortcuts (only calls get_recent_turns)
• Gets stuck in command loops
• No multi-step reasoning (single tool call per research phase)
• Agent ignores get_full_summary and memory tools
• Exploration shallow and repetitive
• No progress beyond starting area in most runs

## What This Tag Represents

• First working two-phase architecture
• Tool-based memory infrastructure in place but underutilized
• Pre-LangGraph baseline
• Agent has tools but doesn't use them effectively
• Architecture stable, reasoning quality insufficient
