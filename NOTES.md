# Notes (2025-12-25)

## What surprised me

• Llama 3.3 supports tool calling perfectly (assumed we needed Qwen 2.5:72b)
• Agent only calls get_recent_turns despite having 5 tools available
• Memory tools never called even once across multiple game runs
• IPv6 required for Parallels networking (not IPv4 as expected)
• Agent loops happen faster than history can prevent them

## What feels fragile

• Research agent prompt doesn't compel comprehensive tool usage
• Tool descriptions clearly not compelling enough for LLM
• Memory importance scoring (had type coercion bugs, LLM returns wrong types)
• Single-shot research phase (one tool call then done, no iteration)
• Structured output parsing (AdventurerResponse validation breaks easily)
• The assumption that having tools means the agent will use them

## One question I don't know how to answer yet

• Does forcing multi-step reasoning (LangGraph) actually improve game progress or just add latency?
