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

## Milestone 3 — Trustworthy world state *(~2–3 days)*

[#11](https://github.com/arsindelve/PlayZork/issues/11) goes first: while corrections are impossible, every other mapper bug is permanent.

| # | Issue | Task |
|---|---|---|
| 10 | [#11](https://github.com/arsindelve/PlayZork/issues/11) | Upsert map transitions: success overwrites BLOCKED (the keystone) |
| 11 | [#10](https://github.com/arsindelve/PlayZork/issues/10) | Tokenized direction extraction (drop `MOVE`, require standalone direction word); record BLOCKED only when the response text indicates a failed move (also mitigates [#15](https://github.com/arsindelve/PlayZork/issues/15)) |
| 12 | [#13](https://github.com/arsindelve/PlayZork/issues/13) | Casefold location lookups (do before M4 — the TurnContext calls these tools directly) |
| 13 | [#14](https://github.com/arsindelve/PlayZork/issues/14) | Record raw-command edges (`CLIMB TREE`) for non-cardinal movement |
| 14 | [#12](https://github.com/arsindelve/PlayZork/issues/12) | Gate mapper update on death detection |
| 15 | [#8](https://github.com/arsindelve/PlayZork/issues/8) | Word-boundary alias matching in explorer spawn |
| 16 | [#21](https://github.com/arsindelve/PlayZork/issues/21) | Inventory: DB as single source of truth, dedupe adds, fuzzy removals, periodic `INVENTORY` resync |
| — | [#15](https://github.com/arsindelve/PlayZork/issues/15) | Accept #10's mitigation; park full room-identity (description fingerprinting) as future work |

**Checkpoint:** one supervised 20–30 turn session; read the HTML reports before starting M4.

## Milestone 4 — Turn engine restructure *(~3–5 days)*

| # | Issue | Task |
|---|---|---|
| 17 | [#25](https://github.com/arsindelve/PlayZork/issues/25) | Deterministic TurnContext (deletes research node + per-agent research calls; **closes [#4](https://github.com/arsindelve/PlayZork/issues/4), [#5](https://github.com/arsindelve/PlayZork/issues/5), [#17](https://github.com/arsindelve/PlayZork/issues/17) as side effects** — verify, then close) |
| 18 | [#23](https://github.com/arsindelve/PlayZork/issues/23) + [#26](https://github.com/arsindelve/PlayZork/issues/26) | One refactor: graph ends at `decide`; bookkeeping (`close/observe/persist`) off the critical path; async-ify Observer/IssueClosedAgent; `turn_number` into graph state |
| 19 | [#24](https://github.com/arsindelve/PlayZork/issues/24) *(full)* | Summaries off the critical path |
| 20 | [#27](https://github.com/arsindelve/PlayZork/issues/27) | Client-side semaphore sized to `OLLAMA_NUM_PARALLEL`; fix/retire the sync retry path. Batched proposals = thesis ablation arm, not default |

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
