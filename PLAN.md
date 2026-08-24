# Plan of Attack: From Audit to Thesis Experiment

**Date:** 2026-08-21
**Input:** 27 GitHub issues ([#1–#22](https://github.com/arsindelve/PlayZork/issues) correctness, [#23–#27](https://github.com/arsindelve/PlayZork/issues/23) orchestration) from a full multi-agent code audit of every agent, the mapper, and the turn engine. Findings were verified by executing the relevant code paths; orchestration timings come from a real run log (`logs/game_codex-smoke-20260821.log`).

## Research framing

The project's original motivation (external memory for small-context models) is obsolete; the standing research question — candidate Master's in AI thesis — is:

> **Can multi-agent deliberation architecture (advocacy agents + arbiter + deterministic toolkits) let much less powerful models solve long-horizon tasks like Zork?**

That framing dictates the ordering below. A weak model cannot compensate for corrupted scaffolding, so correctness of the deterministic state (map, inventory, memory) is a *precondition for the experiment being interpretable* — and turn latency is experimental throughput.

**Ordering principle:** runs must survive → state must be trustworthy → turns must be fast → proposals must be honest → then run the experiment. Each milestone gates the next.

## Measured baseline (why M4 exists)

Turn 1 of the 2026-08-21 smoke run took **7m25s** with only 2 subagents and zero IssueAgents:

| Phase | Time | Share |
|---|---|---|
| History summaries (serial, blocking) | 86s | 19% |
| Spawn agents (4 LLM calls, ~1.9× parallelism) | 146s | 33% |
| Research node (redundant) | 27s | 6% |
| Decision | 74s | 17% |
| Post-decision bookkeeping (observe + persist) | 112s | 25% |

---

## Milestone 1 — Runs survive *(~half day, first)*

| # | Issue | Task | Effort | Status |
|---|---|---|---|---|
| 1 | [#1](https://github.com/arsindelve/PlayZork/issues/1) | Tool-invoke guards, `gather(return_exceptions=True)`, game-loop fallback command | S | ✅ done |
| 2 | [#2](https://github.com/arsindelve/PlayZork/issues/2) | Wrap dedup LLM call in retry; fail open | XS | ✅ done |
| 3 | [#3](https://github.com/arsindelve/PlayZork/issues/3) | Budget coherence; catch `TimeoutError` per turn; stage memory closures until persist | S | ✅ done |
| 4 | [#24](https://github.com/arsindelve/PlayZork/issues/24) *(option 1 only)* | Run the two summaries concurrently (−40s+/turn); full fix in M4 | XS | ✅ done |

**Milestone 1 complete.** Runs now survive single-turn failures, the turn budget is coherent, and turn start is ~40–60s faster. Issues [#1](https://github.com/arsindelve/PlayZork/issues/1), [#2](https://github.com/arsindelve/PlayZork/issues/2), [#3](https://github.com/arsindelve/PlayZork/issues/3) closed; [#24](https://github.com/arsindelve/PlayZork/issues/24) stays open for options 2–3 in M4.

*Rationale: every later fix is validated by long runs; today any single malformed LLM response ends the run.*

**Correction to the #3 plan (2026-08-21):** this milestone originally called for a 60–90s per-attempt timeout. The measured smoke run contradicts that — individual qwen2.5:14b calls took 43–113s under agent contention, so a 90s cap would abort *healthy* calls and (per [#27](https://github.com/arsindelve/PlayZork/issues/27)) pile retries onto an already-slow server. Taken instead: per-attempt 180s × 3 attempts (envelope 546s) inside a turn budget raised to 1200s, with the invariant `TURN_BUDGET_SECONDS ≥ 2 × retry envelope` enforced in `config.py` and covered by a test. M4 should revisit the budget downward once turn latency drops.

## Milestone 2 — Five-minute fixes that restore designed behavior *(~half day)* — ✅ COMPLETE

| # | Issue | Task | Effort | Status |
|---|---|---|---|---|
| 5 | [#6](https://github.com/arsindelve/PlayZork/issues/6) | Rename phantom `get_current_inventory` → `get_inventory` (IssueAgents see inventory again) | XS | ✅ done |
| 6 | [#20](https://github.com/arsindelve/PlayZork/issues/20) | Pass `current_turn` in the issue closer (closer and spawner agree) | XS | ✅ done |
| 7 | [#9](https://github.com/arsindelve/PlayZork/issues/9) | Normalize direction abbreviations (kills N-vs-NORTH explorer loop) | XS | ✅ done |
| 8 | [#19](https://github.com/arsindelve/PlayZork/issues/19) | Validate closed IDs ⊆ shown list; neutralize `[5, 12]` prompt example | S | ✅ done |
| 9 | [#7](https://github.com/arsindelve/PlayZork/issues/7) | Treat `"Unknown"` location as no-location | XS | ✅ done |

**What M2 actually turned up.** Three of the five were worse than filed, and one was a regression M1 introduced:

- **#6** was *made worse* by M1. Once `invoke_tool_safely` stopped silently dropping unknown tools, the `"Error: unknown tool ..."` string was split on commas into phantom carried items — injecting the name of every available tool into the proposal prompt as inventory. Prompt poisoning, not just a missing capability. The lesson generalizes: every consumer of `tool_calls_history` must treat an `Error:` result as a non-result.
- **#7** was broader than filed. `"Unknown"` was fabricated in two roles, and the second (`current_location` for the whole turn) is worse: it zeroes *every* IssueAgent (pathfinding from a nonexistent node → NO PATH → mandatory confidence 0) while handing the ExplorerAgent its maximum possible EV of 47.5 — the arbiter gets exactly one non-zero proposal, an arbitrary direction from an empty exit set, precisely when the agent knows least.
- **#20**'s arithmetic in the issue was wrong: the decay crossover is **11 turns** of relative age, not 44 — the inversion arrives 4× sooner than reported. It also needs >30 open memories to bite, which no live session has reached yet.
- **#9**'s read-side collapse has a trap worth remembering: a legacy DB can hold both `('A','BLOCKED','N')` and `('A','B','NORTH')`, so the collapse must prefer a real passage or it would hide a known exit from the pathfinder.

## Milestone 3 — Trustworthy world state *(~2–3 days)* — ✅ COMPLETE

Applied in dependency order, which the investigation revised from the original plan: **#11 → #12 → #10 → #14**, with #8, #13, #21 independent.

| # | Issue | Task | Status |
|---|---|---|---|
| 10 | [#11](https://github.com/arsindelve/PlayZork/issues/11) | Upsert map transitions: success overwrites BLOCKED (the keystone) | ✅ done |
| 11 | [#10](https://github.com/arsindelve/PlayZork/issues/10) | Tokenized direction extraction; record BLOCKED only on an explicit refusal | ✅ done |
| 12 | [#13](https://github.com/arsindelve/PlayZork/issues/13) | Casefold location lookups (before M4 — the TurnContext calls these tools directly) | ✅ done |
| 13 | [#14](https://github.com/arsindelve/PlayZork/issues/14) | Record raw-command edges (`CLIMB TREE`) for non-cardinal movement | ✅ done |
| 14 | [#12](https://github.com/arsindelve/PlayZork/issues/12) | Gate mapper update on death detection | ✅ done |
| 15 | [#8](https://github.com/arsindelve/PlayZork/issues/8) | Word-boundary alias matching in explorer spawn | ✅ done |
| 16 | [#21](https://github.com/arsindelve/PlayZork/issues/21) | Inventory: DB as source of truth, dedupe adds, conservative fuzzy removals | ✅ done |
| — | [#15](https://github.com/arsindelve/PlayZork/issues/15) | Half-fixed by #10 — stays open for room identity | open |

**Ordering was wrong in the original plan.** #11 had to land with #12, not before it: the upsert treats a respawn room as observed evidence, so after #11 an ungated death turn stopped merely *adding* a bad edge and started *destroying* a correct one. And #10 had to precede #14, because the substring extractor stole commands (`GO IN` → `N`) before #14's branch could ever see them.

**What M3 turned up beyond the issue text:**

- **The backend already answers two of these questions.** `ZorkApiResponse` declares `PreviousLocationName` and `LastMovementDirection`, populated on both hosted games and read *nowhere* in the codebase. `LastMovementDirection` resolves #14's entire vocabulary natively (`climb tree` → `"Up"`, `in` → `"In"`). It is sticky on non-movement turns so it is not a drop-in replacement, but it should be consumed before M4. There is also an undeclared `exits` array (N=0, S=1, E=2, W=3, Up=10, Down=11) — though it lists non-traversable exits, so it is not a walkable-exit oracle.
- **#15 was upgraded from PLAUSIBLE to CONFIRMED on live data**: a *successful* EAST between two rooms both named "Forest", which the old code recorded as a permanent wall. #10 turns that from *actively wrong* into *silently incomplete*. Room identity still needs fingerprinting — and the `exits` array is a concrete signal for it, since the two Forests differ (`[3,2,1]` vs `[3,0,1]`).
- **A docstring in `locations.py` was empirically false.** It claimed death sequences return an empty LocationName; live probing shows a death reports the *respawn* room, so #7's guard never covered #12. Corrected.
- **`INVENTORY` does not cost a move** on either hosted backend (verified by probe) — so a periodic resync would not pollute the score@moves metric the thesis experiment depends on. That removes the main objection to deferring it.
- **The two response-text predicates need opposite biases** and are deliberately kept as separate functions in `response_signals.py`: death must *over*-detect (a false negative writes an edge that now also destroys the true one), a movement refusal must *under*-detect (a false positive fabricates a wall the explorer treats as explored and never retries).

**Checkpoint:** ✅ run 2026-08-22 (session `m3-checkpoint-20260822`, 15 turns) — see STATUS.md. Validated #1, #3, #8, #10, #12, #13, #14, #24-opt1 on live play; found and fixed an inventory regression no unit test could reach (`17a4354`); filed [#33](https://github.com/arsindelve/PlayZork/issues/33).

**Two checkpoint findings change what comes next:**

1. **Turn latency grows superlinearly.** *(Attribution corrected the same day — see STATUS.md. The summaries are not the cause; the call count is constant at 16/turn and each call slows as history-shaped prompt content grows, dominated by BigPictureAnalyzer's 50-turn window on the expensive model. #24 option 3 was therefore NOT implemented. #25 — fewer calls — is the dominant lever.)* Turn time more than doubled in nine turns (79s → 194s), roughly half of it the summary phase (13.9s → 65.9s, ~34% of a turn) — on a fixed model and machine, with option 1 already applied. Option 1 halved a constant and did nothing to the growth *rate*. **[#24](https://github.com/arsindelve/PlayZork/issues/24) option 2 was implemented immediately** (summaries off the critical path), but [#25](https://github.com/arsindelve/PlayZork/issues/25) keeps its priority in M4: taking summarization off the critical path removes it from turn latency entirely, a larger and cheaper win than shaving research round-trips. This is the binding constraint on whether the thesis protocol (N seeded runs × several conditions) is runnable at all.
2. **The agent deadlocked**, alternating two known-refused actions for five turns. Three gaps compound: the map could not learn the wall ([#33](https://github.com/arsindelve/PlayZork/issues/33), [#31](https://github.com/arsindelve/PlayZork/issues/31)), nothing suppressed repetition ([#18](https://github.com/arsindelve/PlayZork/issues/18), M5), and loop detection is off ([#22](https://github.com/arsindelve/PlayZork/issues/22)). **#22's "keep disabled?" question now has data: the capability is needed** — but #18 is the better vehicle, being deterministic, LLM-free, and aimed at the cause rather than the symptom.

## Milestone 4 — Turn engine restructure *(~3–5 days)* — in progress

### Measured starting point (2026-08-22 checkpoint, session `m3-checkpoint-20260822`)

**16 LLM calls per turn, constant.** ~360 call-seconds compressed into ~228s wall clock at ~1.6× effective parallelism against a single Ollama server (`OLLAMA_NUM_PARALLEL` confirmed unset, per [#27](https://github.com/arsindelve/PlayZork/issues/27)). Turn time grew 79s → 228s and then plateaued; the growth is per-call, not per-count, driven by history-shaped prompt content rather than the summaries.

**Reducing the call count is therefore the dominant lever, not reordering the calls.**

| # | Issue | Task | Status |
|---|---|---|---|
| — | [#24](https://github.com/arsindelve/PlayZork/issues/24) opt 2 | Summaries off the critical path; `record_turn()` stays inline, `refresh_summaries()` dispatched and coalesced | ✅ done `4c9b18f` |
| — | [#24](https://github.com/arsindelve/PlayZork/issues/24) opt 3 | Bound summary growth | ❌ **not done — targets a non-cause** (summaries never exceeded 832 chars) |
| — | — | Instrument `BigPictureAnalyzer` + `DeathAnalyzer` (3 of 16 calls/turn were invisible); bound its window via `BIG_PICTURE_HISTORY_TURNS` | ✅ done `4c9b18f` |
| — | [#30](https://github.com/arsindelve/PlayZork/issues/30) | Consume the backend's own signals | ✅ done — see below |

### #30 turned out to be much larger than filed

The backend sends **eleven** fields; the response model declared six and *read* four. Probing both hosted backends found four undeclared fields, and one of them changes the architecture:

- **`inventory`** — the game's own item list, verified authoritative (a failed `TAKE` leaves it unchanged, a successful one updates it). **This replaces the LLM `InventoryAnalyzer` entirely**: one fewer cheap-model call every turn, and it retires the whole class of drift [#21](https://github.com/arsindelve/PlayZork/issues/21) was fighting — no name matching, no add/remove inference, no phantom items. The analyzer is kept only as a fallback for backends that omit the field.
- **`previousLocationName`** — `prev != loc` is an authoritative "did we move?" test, replacing inference from room-name comparison.
- **`lastMovementDirection`** — the direction the game actually moved us (`climb tree` → `Up`, `enter window` → `In`). Preferred over both the command tokenizer and [#14](https://github.com/arsindelve/PlayZork/issues/14)'s raw-command labels, because it is canonical *and* executable, so the explorer correctly sees `UP` as explored. **Sticky**, so it is gated behind the non-movement deny-list.
- **`exits`** — an int-enum exit list. Not a walkable-exit oracle (North of House reports `7` while NW and SW are refused), but a usable room **fingerprint**: the two rooms both named "Forest" differ (`[3,2,1]` vs `[3,0,1]`). That is the signal [#15](https://github.com/arsindelve/PlayZork/issues/15) needs.
- **`actionsAvailableFromLocation`** — object → accepted commands, e.g. `{"window": ["open window", "close window", "examine window"]}`. A deterministic source for the InteractionAgent; folded into #25 below.

### #25 as built

Per-turn LLM calls, with N tracked issues: **10 + 2N → 5 + N**.

`TurnContext` is assembled once per turn in code from local SQLite reads and the turn response, then sliced per agent. Removed: the research node, the per-IssueAgent research call, the ExplorerAgent research call, the InteractionAgent "inventory check" call (106s on the audited turn, to fetch a list the code already had), and the Observer's research call. Each agent now makes exactly one LLM call — the one that requires actual judgement.

A live run caught what the unit tests missed: deleting the research phases also deleted the function-local `from llm_utils import ...` the surviving proposal calls relied on. #1's containment logged `OBSERVE failed, skipping this turn` and kept playing — so the Observer was *silently disabled* rather than crashing. The test that should have caught it only inspected source text; it now executes all four agents.

### Also landed

| # | Issue | Task | Status |
|---|---|---|---|
| — | [#28](https://github.com/arsindelve/PlayZork/issues/28) | Prompt JSON examples reach the model as valid JSON | ✅ done — **narrower than filed**: the template-rendered prompts were already correct; only the plain-string path was over-escaped |

### #23 + #26 as built — the graph finally fans out

`close_issues` and `observe` were chained *after* `decide`, so 18–24% of every turn (measured 2026-08-24) was spent on bookkeeping once the command was already chosen. Neither has any data dependency on the decision — both read the game response, available at the top of the turn.

```
build_context ─┬─ spawn_agents → decide ─┐
               ├─ close_issues ──────────┤
               └─ observe ───────────────┴─ persist → END
```

**Measured: turn 1 went 49s → 29s with an identical 6 LLM calls** — pure overlap, no work removed. The three branches now start at the same timestamp in the logs.

Two things had to be true first, and both were changes in their own right:

- **Nodes return only their own keys.** They previously mutated and returned the whole state, which LangGraph reads as every node writing every key — an immediate concurrent-update conflict. Their write sets turned out to be perfectly disjoint, which is what made the fan-out possible at all.
- **`build_context` is its own node.** Hoisted out of spawn so all three branches share one consistent snapshot of memory and the turn's facts.

**A live run caught what 502 unit tests did not:** three separate `add_edge(x, "persist")` calls are *not* a join. The branches have different depths (spawn→decide is two hops, close and observe are one), so persist was scheduled in the super-step where close/observe finished **and again** when decide finished — it ran **twice per turn**, double-applying the turn's bookkeeping. LangGraph's list start_key (`add_edge([...], "persist")`) is the real join. Pinned by two tests.

This is also the first genuine use of LangGraph in the project. Per [#26](https://github.com/arsindelve/PlayZork/issues/26) the graph was previously a straight line that would have behaved identically as sequential `await`s.

### Remaining

| # | Issue | Task |
|---|---|---|
| 17 | [#25](https://github.com/arsindelve/PlayZork/issues/25) | Deterministic TurnContext | ✅ **done** — measured live: turn 1 went **79s → 40s** and **12 → 6 LLM calls**; turn 2 **114s → 98s**, **16 → 10 calls**. Closes [#4](https://github.com/arsindelve/PlayZork/issues/4), [#5](https://github.com/arsindelve/PlayZork/issues/5), [#17](https://github.com/arsindelve/PlayZork/issues/17) — verified: no `bind_tools`/`tool_choice` remains anywhere in the tree, so those failure modes are structurally impossible rather than fixed. |
| 18 | [#23](https://github.com/arsindelve/PlayZork/issues/23) + [#26](https://github.com/arsindelve/PlayZork/issues/26) | One refactor: graph ends at `decide`; bookkeeping (`close/observe/persist`) off the critical path; async-ify Observer/IssueClosedAgent; `turn_number` into graph state |
| 19 | [#24](https://github.com/arsindelve/PlayZork/issues/24) *(full)* | Summaries off the critical path **— promoted to first in M4, see checkpoint finding 1** |
| 20 | [#27](https://github.com/arsindelve/PlayZork/issues/27) | **Re-scoped by measurement (2026-08-24).** Benchmarked: this Ollama serves **zero** useful parallelism — throughput flat at 0.26 req/s across 1/2/4 concurrency with realistic prompts, so per-call latency scales linearly with concurrency. The audit's "1.9×" was queue overlap, not concurrent service. A semaphore therefore cannot improve throughput; it would only cap latency variance and make timeouts meaningful. **The only levers are fewer calls, shorter prompts, fewer output tokens, or different serving.** See STATUS.md 2026-08-24. |

*Expected outcome: measured 7m25s turn → ~2–3 min, with the architecture more faithful to the design, not less.*

## Milestone 5 — Honest proposals *(~1–2 days)* — in progress

**Promoted ahead of the remaining latency work (2026-08-24).** The M3 checkpoint showed the agent deadlocking from turn 11, alternating two commands the game had already refused. A faster GPU deadlocks faster; until this is fixed, a thesis run cannot produce meaningful `score@turns` data because every run flatlines.

| # | Issue | Task | Status |
|---|---|---|---|
| — | [#18](https://github.com/arsindelve/PlayZork/issues/18) | Repetition suppression | ✅ done |
| — | [#33](https://github.com/arsindelve/PlayZork/issues/33) | Terrain refusals added to the allow-list, from observed play | ✅ done |

**#18 as built.** `TurnContext` computes the set of commands already shown to do nothing *in this room* — a turn counts as unproductive when it neither scored nor changed location. Two consumers:

1. The agents' prompt block lists them **with the response each produced**, so the model sees *why* rather than just being forbidden.
2. `_format_agent_proposals` **zeroes the expected value** of any repeat and annotates it for the arbiter.

The second exists because of the #21 lesson: a 14B model given a bare prohibition invented its own way around it. Prompt text alone is not a mechanism.

Deliberately conservative, on the same reasoning as #11's BLOCKED rule — a false suppression silently removes a legitimate action and nothing in the game text would ever correct it:
- scoped to the current room (`OPEN DOOR` failing in the Kitchen says nothing about the Cellar)
- scoped to the recent window, so suppression **ages out** as the world changes
- matching is case/whitespace only, never semantic
- the first turn in the window is skipped, since the command that walked us into the room has no predecessor to prove it did something

**#33 as built.** Probed the live backend for real refusal phrasings rather than guessing. `"The forest becomes impenetrable to the north."` is topology and is now matched; `"The windows are all boarded."` is an object refusal and correctly is not. A drafted `impassable + mountains` pattern was **removed**: Zork's Forest room *description* reads *"The forest thins out, revealing impassible mountains"* — scenery on a **successful** move, which would have fabricated a wall on arrival.

| — | [#16](https://github.com/arsindelve/PlayZork/issues/16) | InteractionAgent: parser demoted to a hint; the game's own accepted-command list is authoritative | ✅ done |
| — | [#15](https://github.com/arsindelve/PlayZork/issues/15) | Room identity — same-named rooms told apart by their exits signature | ✅ done |

**#16.** The primary fix was not better regexes: [#30](https://github.com/arsindelve/PlayZork/issues/30) surfaced `ActionsAvailableFromLocation`, an authoritative per-object list of the commands the game will accept, so the agent no longer guesses what is interactable from prose. The parser is demoted to a hint that can never short-circuit the LLM — which was the real defect, since it skipped the LLM on precisely the turns it misfired. Its three bugs are fixed too, for the fallback path.

**#15.** The short-term half was already done by #10 (no BLOCKED without an explicit refusal). The long-term half is now in: rooms are identified by *name + exits signature*, so Zork's several "Forest" rooms and the maze stop collapsing into one node. Verified live — `Forest` `[2,0,1]` and `Forest #2` `[3,2,1]` are now separate nodes with correct edges.

Fingerprinting deliberately uses **exits, not the description**: a description changes when an object is taken, which would spawn a phantom room every time something moved. And the match is **containment, not equality**, biased to merge rather than split — opening the trap door legitimately adds an exit, and a wrong split fragments the map into halves that can never be reconnected, where a wrong merge is merely today's behaviour.

## Milestone 5 — remaining

| # | Issue | Task |
|---|---|---|
| 21 | [#16](https://github.com/arsindelve/PlayZork/issues/16) | Demote InteractionAgent's deterministic parser to a hint; always run the LLM phase |
| 22 | [#18](https://github.com/arsindelve/PlayZork/issues/18) | Repetition suppression (check recent turns for the same failed proposal) |
| — | [#17](https://github.com/arsindelve/PlayZork/issues/17) | Should already be closed by #25 — verify |

## Milestone 5b — Goal-directedness in the arbiter *(planned, not built)*

### The gap, measured

Session `analysis-run-20260824`, 20 turns, **score 0**, and the agent never approached the house — the entire Zork opening. Turns 14–20 read:

```
15  EAST    -> Forest
16  NORTH   -> "The forest becomes impenetrable to the north."
17  SOUTH   -> Clearing
18  EAST    -> Canyon View
19  NORTH   -> "You cannot go that way."
20  SOUTH   -> "You cannot go that way."
```

It walked east into the forest to **Canyon View** — a scenic dead-end branch — while the house, its unopened window, the trophy case and essentially every early point sat three rooms west, unvisited since turn 4.

**Nothing in the architecture is accountable for the objective.** IssueAgents advocate for what has been *discovered*; the ExplorerAgent advocates for what is *unvisited* and treats all unexplored directions as equivalent; the InteractionAgent advocates for what is *in the room*. All three are reactive. `GAME_OBJECTIVE` appears in prompts as a header but drives no proposal.

The grating episode shows the cost precisely: a locked grating with no key produced a correctly-formed issue, a correctly-decaying importance (648 → 382), and six `"nothing"` proposals — while the actual opening was never considered, because nothing had *discovered* it and discovery is a random walk.

### Why B rather than A, C or D

| option | verdict |
|---|---|
| **A. A GoalAgent** — a new proposer reasoning from objective + score | Most likely to make it play Zork, but risks becoming a single-shot player in an agent costume. If it wins most turns the architecture has collapsed into the baseline and the comparison is confounded. |
| **B. Strengthen the arbiter** ← chosen | Small, adds no LLM call, introduces no new proposer. |
| C. Goal-aware ExplorerAgent | A heuristic patch; helps only exploration. |
| D. Change nothing, report it | Scientifically clean, but leaves the system unable to play. |

**B is chosen because it does not change what is being compared.** The arbiter already exists and already ranks; giving it the objective changes *how it ranks*, not *what the architecture is*.

**Known limitation, to be stated in the write-up:** the arbiter can only rank what it is given. It cannot generate "go back to the house" if no agent proposed it. B therefore improves selection among existing proposals; it does **not** close the generation gap. If the measurements below show it is insufficient, that is itself the result that motivates arm A.

### What to change

1. **`get_decision_agent_evaluation_prompt`** — supply the objective, the current score, and the score trajectory (turns since it last moved). Instruct: prefer proposals that plausibly advance the objective; when nothing does, prefer exploration that moves *toward* known-but-unentered structures over exploration that moves away.
2. **`get_decision_agent_human_prompt`** / `decision_node` — add `objective`, `score`, `turns_since_score_change`, and a short list of *known but unvisited* map frontier rooms (already derivable from `TurnContext.exits` + the map).
3. **No new call, no new agent, no schema change.**

### How to tell whether it worked

Re-run 25 turns and compare against `analysis-run-20260824` as the control:

- **score@turns** — does the score move at all inside 25 turns?
- **distinct rooms** and whether the house is entered
- **arbiter override rate** — B should *raise* it, since the objective gives the arbiter a reason to disagree with raw EV
- **wasted turns** — should fall if aimless frontier-walking drops

### Built and measured (2026-08-24) — two of three components work

Verified against `analysis-run-20260824` as control:

| component | status |
|---|---|
| objective in the arbiter prompt | ✅ the arbiter now reasons in objective terms — *"it advances the objective by acquiring a new item"*, where the control cited only confidence and EV |
| score trajectory | ✅ *"Score: 0 — UNCHANGED FOR 6 TURNS — the current approach is not scoring, prefer a change of direction"* reaches the arbiter as intended |
| **frontier** | ❌ **structurally empty** |

**The frontier definition was wrong, and the reason generalises.** It was defined as *map nodes reached but not departed from*. On a linear exploration path — A→B→C, which is what the agent actually does — every room reached has also been left, so the set is **always empty**. The one signal intended to redirect the agent toward the objective is unavailable precisely when it is needed.

The useful frontier is **not in the map**. *"You are facing the north side of a white house"* names the most valuable object in the opening, and a house is not a map node — it is a noun in a room description that nothing converts into a destination. A map-derived frontier cannot represent it in principle.

**Trajectory was unchanged for the first 8 turns**, identical to the control. At the decisive turn (5, at North of House), the explorer chose NORTH on genuinely correct evidence — the description says *"a narrow path winds through the trees to the north"* and the game confirms NORTH is a real exit, so NORTH scores 6 against EAST's 4. The evidence points north; EAST to Behind House is a real exit that nothing recommends. **This is not a scoring bug — it is the generation gap.**

### The frontier is fixed (2026-08-24) — it now reads room text

`tools/mapping/landmarks.py`. Deterministic extraction, no LLM call, no added
latency. A closed vocabulary of *enterable structures* rather than general noun
extraction: a parser or a model would surface furniture and abstractions, and a
wrong landmark sends the agent chasing something that does not exist, whereas a
missed one costs only what today already costs. False negative beats false
positive, as everywhere else in the world model.

Deciding "visited" was the subtle part, and every name-matching rule fails:

| rule | why it fails |
|---|---|
| substring | "West of House" contains "house" — standing **outside** suppresses the best lead in the game |
| strip positional prefix | "West of House" → "House" → same failure |
| exact equality | nothing is ever named "House", so entering via the window into the Kitchen never clears it and it nags forever |

What works is the room text itself. A landmark retires when a visited room is
named for it ("Cellar"), or when a visited room is named **in the same
sentence** — Zork's Kitchen reads *"you are in the kitchen of the white
house"*, so being inside still names it. Sentence scope keeps that safe:
*"west of a white house"* never contains the location name "West of House", so
being outside retires nothing. Recency does the rest — descriptions that stop
mentioning a landmark let it age out, and **the world model never has to know
what "inside" means.**

The Kitchen case shows the upside: with the house retired, the same pass
surfaces *"dark chimney"* — the actual route down to the Cellar.

Verified against the live session DB with real toolkits, not just unit tests:
the map frontier was empty on the recorded linear path and the house now
surfaces. Suite 635 → 654.

### Measurement correction: 5b costs ~4% more, not 27% less

The first read of the run analysis showed 10,313 tokens/turn against the
control's 14,132 and looked like a large saving from the #16/#18 memoization.
It was **a turn-count artifact** — the control had run 26 turns and the verify
run 11, and later turns cost more because summaries and history grow. On
matched turns 1–10 the honest figures are:

| | tokens/turn |
|---|---|
| control (turns 1–10) | 10,898 |
| 5b (turns 1–10) | 11,344 (**+4.1%**) |
| control (all 26 turns) | 14,697 ← the misleading denominator |

+4.1% is what a richer arbiter prompt *should* cost. **Any token comparison
between runs of different lengths must be matched-turn.** This is the third
attribution error of this kind in the project, and the general rule is now:
never compare a per-turn mean across runs of unequal length.

Neither arm has scored. 26 turns and 11 turns, both 0 points — the agent is
not entering the house, which is what the frontier fix targets.

### The generation gap, demonstrated cleanly (2026-08-24, `frontier3-20260824`)

With the frontier working, the decisive turn reads:

```
=== PLACES SEEN BUT NOT ENTERED ===
  - white house (named nearby, never entered)
  - boarded front door (named nearby, never entered)

=== AGENT PROPOSALS ===
ExplorerAgent: [Confidence: 95/100, EV: 47.5]
  Proposed Action: NORTH
```

**One proposal.** The arbiter chose NORTH because NORTH was the only thing on
the ballot. The house is in its context, correctly identified as unentered, and
no agent turned it into a candidate.

This is a materially stronger result than the earlier run, where the same
NORTH could be blamed on the empty frontier. That confound is now removed: the
signal is present, correct, and prominent, and the outcome is unchanged. The
failure is **generation, not ranking** — and no amount of arbiter tuning
addresses it, because arbitration operates on a set of size one.

It is worth noting how nearly this was missed. Two consecutive runs reported
"No unfollowed leads" from two *different* causes (see the commit log), and
both times the honest-looking conclusion available was "5b doesn't help".
Believing that would have attributed a wiring bug to the architecture.

### Consequence: arm A is now motivated by evidence, not speculation

B's pre-registered limitation has been demonstrated rather than predicted:

> *"the arbiter can only rank what it is given … B improves selection among existing proposals; it does not close the generation gap. If the measurements show it is insufficient, that is itself the result that motivates arm A."*

That result is in. A GoalAgent's remit should be specifically **to turn nouns in room text into destinations** — the thing no current agent does. Its ablation value is unchanged and its shape is now clearer.

If a cheaper intermediate is wanted first: redefine the frontier as *places named in recent room descriptions that do not appear in the map*, which is deterministic, needs no new LLM call, and would have surfaced "white house" on turn 4.

### IDEA (not yet fleshed out): frontier leads → the existing issue store

Recorded 2026-08-24 after establishing the generation gap. **Not designed yet**
— being grounded against Planetfall first, see below.

The discussion started at "add a GoalAgent" and collapsed to something smaller.
There is **no type distinction between a goal and an issue**. Compare:

```
stored today:  "Grating at Clearing — open or unlock it"
would be:      "White house at West Of House — named but never entered; find a way in"
```

Same shape. A GoalAgent to hold the second would have been a distinction
without a difference, and "kill the troll" proves it — the troll is in the room
with you, affords a verb, and the observer flags it exactly like the grating.
It already works.

**What actually differs is provenance.** The ObserverAgent records
*affordances, not objectives*. Every issue it stored across two full runs was a
manipulable object in the room being stood in:

```
t1 [700] Small mailbox at West Of House — open it and examine contents
t6 [500] Pile of leaves at Clearing — take leaves and investigate
t7 [800] Grating at Clearing — open or unlock it
```

The white house is named in the same description as the mailbox and is never
flagged, because from outside it affords nothing — the door is boarded. So the
gap is in the *producer*, not in the representation.

The sketch, therefore: `persist_node` (the graph's single writer) stores
frontier leads as ordinary issues, deduped exactly like the observer's. The
memory record's `location` field already means the right thing —

```python
location: str              # Where we were when we learned this
```

— the **sighting room**, not where the target is. For the grating those
coincide; for the house the sighting room is `West Of House`, a real map node,
so existing BFS pathfinding routes to it unchanged and the IssueAgent gains one
line: `DIRECTION TO 'West Of House': SOUTH`.

Predicted effect on the decisive turn — the ballot goes from one entry to two,
making it a contested turn where arbitration can act at all:

```
before:  ExplorerAgent [EV 47.5] NORTH
after:   ExplorerAgent [EV 47.5] NORTH
         IssueAgent #1 [EV ????] SOUTH   ← toward the house
```

Secondary, smaller: `direction_to` returns `NO PATH` both for a mapped room
with no known route and for a target that was never a room at all. An agent
told `NO PATH` reasonably drops the lead. Worth distinguishing, but fix 1
avoids it by construction, so it is a guardrail rather than a prerequisite.

**Open question this must answer before being built** — does it help escape the
Planetfall explosion? That is a *timed* objective, unlike entering the house,
and a lead that arrives too late is worth nothing. See the Planetfall findings
below.

### GROUNDING: would it help escape the Planetfall explosion? **No.** (2026-08-24)

Run `pf-20260824`. The answer is unambiguous and it invalidates the idea above
as a priority. Every stage the idea would add **already worked**, and the run
still failed.

The game hands over the escape route in the STARTING ROOM, turn 1:

```json
"actionsAvailableFromLocation": {"escape pod bulkhead": ["open bulkhead", "close bulkhead"]}
```

And the pipeline handled it correctly, at every stage:

1. the parser read it — `actionsAvailableFromLocation` is a declared alias ✓
2. it reached the prompt — *"THE GAME ACCEPTS THESE COMMANDS HERE: escape pod bulkhead: open bulkhead, close bulkhead"* ✓
3. the **ObserverAgent stored it**, at importance 900, with the correct sighting room: `"Locked pod bulkhead at Deck Nine — open bulkhead and examine escape pod"` ✓
4. the **InteractionAgent proposed it**: `OPEN escape pod bulkhead`, confidence 70 ✓

Then the arbiter chose `GO UP` and walked away from the pod.

```
InteractionAgent: [Confidence: 70/100]             OPEN escape pod bulkhead
ExplorerAgent:    [Confidence: 95/100, EV: 47.5]   GO UP                     ← chosen
```

**The InteractionAgent is not given an expected value at all.** From
`_format_agent_proposals`:

| agent | EV formula |
|---|---|
| IssueAgent | `(importance/1000) × (conf/100) × 100 × mult` |
| ExplorerAgent | `(unexplored/10) × (conf/100) × 50 × mult` |
| **InteractionAgent** | **none — confidence only** |

The arbiter is instructed to rank by expected value, and the only agent that
proposes object interactions is structurally unrankable. Worse, the explorer's
EV scales with `unexplored/10`, which is at its **maximum at the start of every
game** — exactly when the pod mattered. Turn 2: `(10/10)×(95/100)×50 = 47.5`.

This also explains Zork, where ExplorerAgent won 16 of 26 contested turns. The
same bias, in a game where it merely wasted turns instead of being fatal.

**So the frontier→issue idea addresses the wrong failure.** Zork's problem was
generation — the house was never proposed. Planetfall's problem is the exact
opposite: the correct action *was* proposed, by the right agent, at the right
turn, and lost the ranking. Adding more proposals to a ballot that
systematically under-ranks interaction would not have helped, and might have
hurt by adding more EV-bearing exploration-flavoured entries.

Two further findings from the same run confirm the idea does not transfer:

- **The landmark vocabulary is Zork-shaped.** On *"To starboard is the Ion
  Reactor ... and to port is an escape pod"* it returns `['corridor']` — the
  most useless noun on a spaceship, while missing the objective. `pod`,
  `reactor`, `bay`, `airlock`, `lift`, `deck` are all absent from
  `LANDMARK_NOUNS`. A closed vocabulary is defensible for precision but does
  not survive a change of game, and this project runs three backends.
- **The issue was already stored with a valid map location** ("Deck Nine"), so
  the sighting-room mechanism the idea depends on was never the blocker.
  Instead, once the agent had wandered to Deck Eight, `IssueAgent #1` proposed
  **`nothing` at confidence 0** for a 900-importance issue two rooms away —
  it does not use its own precomputed `DIRECTION TO` to navigate back.

**Revised priority.** What would actually help escape the explosion, in order:

1. give the InteractionAgent an EV, so the pod proposal is rankable at all
2. apply the repeat/undo multiplier to it — `note, _ = repeat_note(...)`
   **discards the multiplier**, so #18's demotion is a cosmetic warning line
   for this agent rather than a mechanism, which is precisely the
   "prompt text is not a mechanism" failure the project already learned once
3. surface the game clock — `Time` is parsed (`alias="time"`, observed
   advancing 4654 → 4708) and **never read anywhere**. On a timed objective
   the agents cannot see the deadline they are being judged against
4. make the objective specific — Planetfall's is configured as
   `"Complete the mission"`, which is interpolated into every prompt and tells
   the arbiter nothing about escaping
5. teach `IssueAgent` to navigate toward its issue's location instead of
   returning `nothing`
6. fix movement undo detection and the direction vocabulary (below)

The frontier→issue idea drops to the bottom of that list. It remains a
reasonable answer to Zork's generation gap; it is not the bottleneck.

### Bugs found by playing Planetfall (2026-08-24)

None of these were visible in Zork.

- **`Time` parsed, never used** — see above. The only `Time` matches in the
  codebase are `timeout` and `setTimeout`.
- **Direction vocabulary is compass-only.** Planetfall's ship uses
  port/starboard/fore/aft. Not fatal for movement — the backend translates
  (`starboard` → `lastMovementDirection: "E"`) and compass commands are
  accepted — but `find_mentioned_directions` never matches them, so the
  explorer's `+2 mentioned` bonus is **systematically unavailable for lateral
  moves**:

  | direction | game exit | mentioned | cardinal | total |
  |---|---|---|---|---|
  | UP | +3 | +2 (*"a gangway leads up"*) | +0 | **5** |
  | EAST (= starboard) | +3 | **0** (unrecognised) | +1 | **4** |

  The agent climbed the ship — `GO UP, GO UP` — because up was the only axis
  its vocabulary could see. Teaching it `starboard` makes EAST score 6 and win.
- **`_INVERSES` contains no compass directions**, so `undoes_recent_progress`
  cannot tell that EAST undoes WEST, or DOWN undoes UP.
- **`inverse_of` requires an object after the verb** (`len(tokens) > verb_len`),
  so a bare `WEST` could never resolve an inverse even if one were listed.
- **`normalize_command` does not canonicalise movement** — `GO WEST`, `WEST`
  and `W` are three distinct keys, so repetition suppression and the
  `succeeded` map miss synonyms. `extract_direction` already maps all three to
  `WEST` and is strict enough to use here (it returns `None` for `TAKE LAMP`,
  `OPEN WINDOW`, `PUSH NORTH WALL`).

  Combined effect, observed in Zork `frontier3-20260824`: the agent reached
  **Behind House** — the room containing the window into the house — and
  oscillated `GO WEST → EAST → GO WEST` straight back off it, with **zero**
  suppressions fired across 16 turns. Planetfall reproduced it vertically:
  `GO UP → GO UP → GO DOWN`.

### Pre-register arm A as an ablation

Add **`+goal_agent`** as a fourth experimental arm alongside lean single-shot / full single-shot / multi-agent. That way goal-directed *proposal generation* is measured rather than quietly patched in, and the gap between multi-agent and multi-agent+goal_agent is exactly the size of the generation deficit described above.

## Milestone 6 — Decision point, not a fix

[#22](https://github.com/arsindelve/PlayZork/issues/22) LoopDetectionAgent: keep disabled through baseline experiments; treat "fixed + re-enabled" as an optional ablation arm. Don't spend the five fixes unless the experiment design wants that arm.

---

## Serving

**vLLM backend added** (2026-08-24) so moving to GPU is a config change, not a port:

```
PLAYZORK_LLM_PROVIDER=vllm
PLAYZORK_VLLM_BASE_URL=http://localhost:8000/v1
PLAYZORK_VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct
```

vLLM speaks the OpenAI API, so the existing client works unchanged. What differs is **continuous batching**: this Mac's Ollama serves concurrent requests at flat throughput (measured 0.26 req/s at 1, 2 and 4 concurrency), i.e. no parallelism at all, while the system issues 5–10 concurrent calls per turn. That single difference is worth more than every orchestration change in M4 combined.

Measured baseline on the current machine (Apple M5, 24GB), qwen2.5:14b:

| | rate |
|---|---|
| generation | 14 tok/s |
| prefill | 237 tok/s |
| one 3.5k-token call | 14.4s, of which **10.4s is prefill** |

Prefill dominating is why a GPU should help disproportionately here — it is compute-bound and parallel, where generation is bandwidth-bound and sequential.

**Re-baseline everything after the move.** Every timing in this document and in STATUS.md is Mac-specific, including the "concurrency buys nothing" finding, which is a property of *this serving stack* and not of the architecture.

## Result aggregation — planned, not built

Runs will happen on at least two machines (Mac now, an RTX 5070 Ti PC from 2026-08-30) and possibly in the cloud. **Operational state stays in local SQLite; only results are shared.**

### Why not move persistence to DynamoDB

Considered and rejected for the hot path, on three grounds:

1. **The ranking logic is SQL.** Lazy importance decay is computed *in the query* — `CAST(importance * pow(?, MAX(0, ? - turn_number)) AS INTEGER)` with `ORDER BY effective_importance DESC`. That is the mechanism behind [#20](https://github.com/arsindelve/PlayZork/issues/20). DynamoDB has no computed ordering, so it becomes fetch-all-and-sort in Python — workable at this scale, but a rewrite of something a whole milestone was spent making correct.
2. **The map write rule is a conditional update.** [#11](https://github.com/arsindelve/PlayZork/issues/11)'s asymmetry (a real destination overwrites BLOCKED, never the reverse) is `ON CONFLICT … DO UPDATE … WHERE`, with 14 tests written against real SQLite semantics. Expressible as a Dynamo `ConditionExpression`, but re-deriving subtle logic against a different consistency model is risk with no payoff.
3. **Latency lands on the hot path and does not shrink.** `TurnContext` alone does ~6 reads per turn, plus mapper and persist traffic. Local SQLite is microseconds; Dynamo is ~10–20 ms per round trip. ~1–2 s/turn is negligible against today's 40–110 s turns but becomes a *growing* fraction once the GPU cuts turn time — exactly when many seeded runs are being executed.

**Scale does not motivate it either:** the entire dataset across every session ever run is **244 KB**. A 50-turn session is ~60 turn rows, ~12 memories, ~18 map edges.

**And a local file is better for the thesis.** *"Here is the database for run 7"* is one small file that can be attached to an appendix, diffed, or re-analysed years later. A shared mutable table has no natural per-run boundary.

### What is actually needed

Runs are independent — nothing reads another run's map mid-game. The real requirement is **comparing results across machines**, which is one write at the end of a run, not a persistence-layer change.

**Planned:** a `runs` record written once per completed run, to DynamoDB or simply S3/JSON:

| field | why |
|---|---|
| `session_id`, `condition`, `seed` | identifies the arm |
| `turns`, `final_score`, `scoring_turns` | `score@turns` |
| `wall_seconds`, `total_tokens`, `tokens_by_operation` | `score@wall-clock`, `score@tokens` |
| `git_sha`, `model`, `provider`, `serving_config` | provenance — required by the pre-registration below, and the reason a GPU run and a Mac run can be compared at all |
| `wasted_turns`, `override_rate`, `agent_win_counts` | the behavioural measures `run_analysis` already computes |

Everything in that table is already produced by `tools/reporting/run_analysis.py`; this is an upload step, not new analysis. No hot-path cost, and it makes Saturday's GPU runs land alongside today's Mac runs automatically.

**Not built yet — deliberately.** It is only worth building once the run harness exists, so the two land together.

## Instrumentation for the experiment

**Per-turn token accounting** (`VersionTwo/token_meter.py`, `turn_tokens` table) — landed 2026-08-24.

Wall-clock cannot compare architectures across machines, and on any one machine it is a proxy for token volume anyway: this box's Ollama benchmarks at *flat* throughput across 1/2/4/8 concurrent requests, so concurrency changes nothing and only token count moves the number. Tokens are hardware-independent, so a laptop run and a GPU run are directly comparable, and the multi-agent arm gets charged for what it actually costs rather than for how fast the box happens to be.

Counts come from the provider's own metadata, never an estimate. Structured-output chains return a Pydantic model carrying no usage metadata, so **totals are a floor, not an estimate** — an invented number would be worse than a missing one here. Metering lives in `llm_utils`, the single choke point both retry helpers pass through, and is measured *between turn starts* so background work (summaries, big-picture, death analysis) is attributed to the turn that spawned it.

**Report token counts alongside `score@wall-clock` in every result.** Otherwise the multi-agent arm is penalised for token volume in a way that says nothing about the architecture.

## The experiment — control arm built

**`PLAYZORK_CONDITION=single_shot | multi_agent`** (default `multi_agent`). Both implement the same interface, so `GameSession`, reports, persistence and token accounting are unaffected by which arm runs.

`SingleShotService` makes **one inference per turn** with everything in context: current room, inventory, known exits, the map so far, tracked issues, what has already been tried here without effect, recent turns, and both summaries. Same model tier as the arbiter, so the comparison isolates architecture rather than model quality.

**Deliberately generous to the control.** A weak baseline would make the whole comparison meaningless, and a generous one is the conservative choice — it is harder for the treatment to win. Worth a decision before the real runs: this is arguably *too* generous, since `tracked_issues` and `known_map` are themselves scaffolding under test. A leaner variant (history + game state only) is a prompt-field change away and would make a cleaner ablation ladder:

| arm | context |
|---|---|
| single-shot, lean | game state + history only |
| single-shot, full | + map, memory, exits *(built)* |
| multi-agent | + advocacy and arbitration *(built)* |

First live run (`baseline-smoke-20260824`) played sensibly — `EXAMINE MAILBOX` → `OPEN MAILBOX` → `TAKE LEAFLET` — at **~20s/turn against the treatment's ~70s**.

**Measured cost per turn, both arms** (from `turn_tokens`; the control's 3 calls are 1 decision + 2 shared background summaries, which both arms pay):

| arm | LLM calls | tokens/turn |
|---|---|---|
| single-shot | 3 | ~1,900 |
| multi-agent | 6–10 | *(to be measured over a matched run)* |

## Then: the experiment

M1–M4 is the platform. Thesis protocol: same 14B model, N seeded runs each of

- **(a)** single-shot with full history in context (the "big context makes scaffolding redundant" baseline) — **built**, `PLAYZORK_CONDITION=single_shot`,
- **(b)** full multi-agent architecture — **built**, `PLAYZORK_CONDITION=multi_agent`,
- plus component ablations (no map, no issue memory, batched vs per-agent proposals, loop detection on/off),

measured on **score@turns**, **score@wall-clock** *and* **score@tokens**. The third is not optional: on fixed serving, wall-clock is a proxy for token volume, and without tokens no cross-machine comparison means anything.

---

## Framing the result before running it

**The experiment can produce three outcomes, and two of them are commonly mistaken for failure. Decide now how each is written up, so the answer is a finding rather than a disappointment.**

The instinct to guard against is treating (b) beating (a) as "success" and anything else as a null result. That framing would waste the most interesting outcomes and creates an incentive to keep tuning arm (b) until it wins — which is how a thesis stops being an experiment.

### Outcome 1 — deliberation wins

Arm (b) reaches a higher score at equal turns. This is the hypothesis as stated, and the write-up is straightforward. **The obligation is to show it is deliberation doing the work**, not extra tokens: report score@tokens, and use the ablations to identify which component carries the effect. A win that disappears once token budget is equalised is a finding about verbosity, not architecture.

### Outcome 2 — the arms are comparable

Entirely plausible, and **not a null result.** The honest reading is that this architecture's value is not decision quality — and that conclusion has direct evidence behind it, because much of what was rebuilt in M1–M5 helps *both* arms:

- deterministic world state (map, inventory, memory correctness)
- the backend signals the agents were previously guessing at (#30)
- repetition suppression (#18)
- room identity (#15)

Both arms consume the same `TurnContext`. If they perform alike, the measurable gain came from **scaffolding correctness**, not from advocacy and arbitration. That is a publishable claim with an unusually clear evidence trail: the audit found 27 defects in the deterministic layer, each with a documented failure mode, several of which silently corrupted agent inputs for whole sessions.

It also directly answers the question that motivated the reframing: *not* "does the framework help a weak model", but **"how much of a weak model's apparent incapacity is actually corrupted input?"** — the open question logged in NOTES.md on 2026-08-21.

### Outcome 3 — the baseline wins

Also a real result, and the most useful one for anybody building these systems. It would say the deliberation overhead costs more than it returns at this model scale, which is a finding about *when* multi-agent architectures are justified — a question the field mostly assumes rather than measures.

If this happens, resist the pull to keep tuning (b). Report it, then investigate *why*: the ablation ladder is designed to localise the cost.

### What this means for design decisions still open

- **The control arm is deliberately generous** — it currently receives the map and tracked issues, which are themselves scaffolding under test. That is the conservative choice (it is harder for the treatment to win), but it also makes Outcome 2 harder to interpret, because both arms then share the scaffolding. **Run the lean variant too** (history + game state only) and the three rungs separate the two variables cleanly:

  | arm | context | isolates |
  |---|---|---|
  | single-shot, lean | game state + history | does scaffolding help at all? |
  | single-shot, full | + map, memory, exits | does scaffolding alone suffice? |
  | multi-agent | + advocacy and arbitration | does deliberation add anything? |

  The gap between rungs 1 and 2 is the scaffolding effect; between 2 and 3 is the deliberation effect. Without rung 2, those two are confounded and Outcome 2 cannot be attributed.

- **Pre-register the metrics and the number of seeds** before running, and do not change arm (b) after seeing results without re-running everything. This project has an unusually good paper trail — every fix is a commit with a documented failure mode and a regression test — and that credibility is worth protecting.

- **Report the serving configuration.** The measured "concurrency buys nothing" result is a property of this stack, not of the architecture, and it materially affects wall-clock. A reader on batching serving would see different numbers from identical code.

---

**Free closures along the way:** #4, #5, #17 (via #25); #15 (via #10 + the exits fingerprint).
**Rough total:** 8–12 focused days to a runnable experiment — largely spent, with the platform now built and the remaining work being the runs themselves.
