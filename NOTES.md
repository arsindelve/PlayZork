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


---

# Notes (2026-08-24)

## What surprised me

• The backend was answering questions we were paying a 14B model to guess at. `inventory`, `lastMovementDirection`, `exits` and a per-object list of accepted commands were all in the response payload, declared-but-unread or not declared at all. Three separate issues (#16, #21, #14) were really one: *stop inferring what the server already reports.*
• **Concurrency on this machine is entirely fictional.** Flat throughput at 1, 2, 4 and 8 simultaneous requests. The original audit measured "~1.9×, not 4×" and read it as degraded parallelism; it is *absent*, and the 1.9× was queue overlap in log timestamps.
• Consequently the graph fan-out — the change that finally made LangGraph earn its keep — produced no reliable speedup. Turns 1–2 improved, turns 3–4 got worse, net 7% and inside the noise.
• A prompt rule that says only what **not** to do is an invitation. Told "never add an item already held", the model emitted a *removal* and emptied the inventory. The fix was to state the no-op explicitly *and* enforce it in code.
• Five live-run catches in a row against a green suite. Four were silent rather than loud — which is the price of error containment, and worth the price, but it means the suite is not the last line of defence.
• Zork's Forest room description contains "impassible mountains". A refusal-detector pattern I wrote would have matched it and fabricated a wall *on arrival*. It survived only because Infocom spells it with an `i` and I had typed an `a`.

## What feels fragile

• The measurement blind spot repeats easily: `BigPictureAnalyzer` and `DeathAnalyzer` called `.invoke()` directly and were invisible for weeks, so three of sixteen calls per turn were unmeasurable and the first latency diagnosis was confidently wrong.
• Room identity by exits signature handles the Forest cluster but not maze rooms with identical topology. The merge bias means the failure is the status quo rather than corruption — but the maze is exactly where the game demands good mapping.
• The control arm is *generous* — it sees the map and tracked issues, which are themselves scaffolding under test. That is the conservative choice, but it needs deciding explicitly before runs, not by default.
• Every timing in the repo is Mac-specific, including the "concurrency buys nothing" finding. All of it must be re-baselined after the hardware change.

## One question I don't know how to answer yet

• Now that the scaffolding is correct, the 2026-08-21 question can finally be asked — but the answer may be unflattering in a way worth preparing for. If the fixed **single-shot** arm plays comparably, the honest finding is that this architecture's value is not decision quality. That would still be a result: much of what was rebuilt this week — deterministic world state, the API signals, repetition suppression — helps *both* arms and is where the measurable gain actually came from. The thesis may be about scaffolding correctness rather than deliberation, and the experiment should be designed so that outcome is a finding rather than a disappointment.


---

# Notes (2026-09-10)

## What surprised me

• **The agent opened the escape route, then proposed closing it and walked away.** At Behind House it chose `OPEN WINDOW`; the game replied *"you open the window far enough to allow entry."* The next turn the window's own IssueAgent proposed `CLOSE WINDOW`, the IssueClosedAgent retired the issue — quoting "allow entry" as proof it was done — and the arbiter picked `NORTH`. Across 25 turns nobody ever proposed entering. The single move that scores was never on any ballot.
• **The invitation to enter was read as permission to stop.** "You open the window far enough to allow entry" was taken as "issue complete," because the issue was authored as "examine or open it" — the action, not the aim. Each agent behaved correctly against a target that was simply the wrong shape.
• **The whole multi-agent design is only as runnable as the model's default verbosity.** What made local play possible was not any audit fix — it was one line disabling qwen3's thinking. A trivial prompt: 118 output tokens with thinking, 2 without. Five of those firing concurrently at a serial Ollama server buried every turn under the 180s timeout; one 25-turn attempt managed a single decision in ~13 minutes. The architecture's cost is set by the model's defaults, not just its own call count.
• **My own fix reflex was the architecture's bug in miniature.** "Open something = progress → propose entering it" solved the example in front of me, not the class. The reactive framing is contagious: it is exactly how the proposers already work, and exactly why they miss.

## What feels fragile

• **The roster of drives is a start, not a complete set.** The agents *are* the competing goals; the arbiter can only choose among what they advocate — explore, resolve-issues, interact-here. A move no drive is aimed at ("get inside the house") is never offered, however good the arbiter is. Whether the fix is a new drive or an existing one reaching further is undecided.
• **Issues are authored as leaf actions, not aims.** The Observer writes the thing to *do* ("open the window"), not the aim ("get inside"), so the one drive with momentum toward the house terminates — or reverses to `CLOSE WINDOW` — the instant its action completes, exactly when the payoff is one step away. The closer was right; the target was the wrong shape.
• **The experiment tooling is itself a source of error.** Backgrounded game processes on Windows did not die when the watcher "killed" them (MSYS `kill` against a detached python), orphaning several that held VRAM and quietly degraded concurrency between runs. Timing measurements are only as trustworthy as the process hygiene around them.

## One question I don't know how to answer yet

• A drive aimed at the objective has to persist past the completion of any single action — the window issue died the moment it was opened — yet if it persists *and* is strong it risks winning every vote and collapsing the deliberation into a single voice, which is the very single-shot baseline the architecture is meant to beat. How does a drive stay aimed at an aim across many turns while remaining just one competing voice the arbiter weighs? That balance is the open question, and it is where the proposed solution goes next.

• **The design that came out of chewing on the above is written up in `docs/GOAL_AGENTS_PROPOSAL.md`, including a decision log of how we reached it (2026-09-10) — the wrong turns included, because that is where the choices were actually made.** Short version: goals become ephemeral self-terminating agents ("Meeseeks") in one tree rooted at the game objective; importance is relative to the parent (never absolute); a hard ~25 budget makes creation an admission decision and eviction the give-up death; dedup keys on the target state/referent. Unresolved and blocking: whether qwen3:14b can create and spawn goals at the right altitude (probe says state-shaping OK, altitude + grounding shaky).
