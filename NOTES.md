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

---

# Notes (2026-08-21)

## What surprised me

• A full audit of a "healthy-looking" system found 22 serious correctness bugs — agent scaffolding fails *silently*; the runs never crashed on any of them
• One wrong tool name (`get_current_inventory` vs `get_inventory`) invisibly killed the entire IssueAgent inventory capability — "empty" every turn, forever, no error
• ChatOllama silently ignores `tool_choice="any"` — the "research is required to call tools" guarantee never existed under the default provider
• The prompt's own worked example (`[5, 12]`) is a live trigger: an example-echoing model closes real issues 5 and 12
• "Parallel" agent fan-out measured at only ~1.9× against local Ollama; contention roughly doubled individual call latency
• 25% of measured turn time happens *after* the command is already chosen
• One `MOVE RUG` permanently deletes a map exit (substring direction matching + UNIQUE constraint that silently rejects corrections)
• Turn 1 with only two subagents took 7m25s
• The 2025-12-25 LangGraph question is now partly answered: as wired (a pure linear chain), it adds structure but zero parallelism — the latency is real and the concurrency is illusory

## What feels fragile

• No exception handling anywhere in the turn path — any single malformed LLM response ends the run
• The map can only degrade, never repair (BLOCKED is forever; death teleports become walkable edges)
• Inventory cache and DB drift apart permanently; phantom items feed every prompt
• Deterministic regex parsers outrank the LLM with garbage (`TAKE NOTHING` at confidence 90)
• The timeout/retry design adds duplicate load to the very queue that caused the timeout

## One question I don't know how to answer yet

• Once the scaffolding is actually correct, how much of the 14B model's past failure was reasoning capacity vs. corrupted inputs? If the fixed *single-shot baseline* already plays much better, the architecture's measurable headroom shrinks — the thesis experiment has to be designed so that answer is a finding, not a disappointment.
