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

## Milestone 5 — Honest proposals *(~1–2 days)*

| # | Issue | Task |
|---|---|---|
| 21 | [#16](https://github.com/arsindelve/PlayZork/issues/16) | Demote InteractionAgent's deterministic parser to a hint; always run the LLM phase |
| 22 | [#18](https://github.com/arsindelve/PlayZork/issues/18) | Repetition suppression (check recent turns for the same failed proposal) |
| — | [#17](https://github.com/arsindelve/PlayZork/issues/17) | Should already be closed by #25 — verify |

## Milestone 6 — Decision point, not a fix

[#22](https://github.com/arsindelve/PlayZork/issues/22) LoopDetectionAgent: keep disabled through baseline experiments; treat "fixed + re-enabled" as an optional ablation arm. Don't spend the five fixes unless the experiment design wants that arm.

---

## Then: the experiment

M1–M4 is the platform. Thesis protocol: same 14B model, N seeded runs each of

- **(a)** single-shot with full history in context (the "big context makes scaffolding redundant" baseline),
- **(b)** full multi-agent architecture,
- plus component ablations (no map, no issue memory, batched vs per-agent proposals, loop detection on/off),

measured on **score@turns** and **score@wall-clock**. M5 sharpens arm (b) but doesn't block starting (a).

**Free closures along the way:** #4, #5, #17 (via #25); #15 partially (via #10).
**Rough total:** 8–12 focused days to a runnable experiment.
