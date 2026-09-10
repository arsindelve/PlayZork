# Goal-Agents ("Meeseeks") — a recursive, self-terminating goal layer

**Status:** Proposal. Not built. Design direction as of 2026-09-10. The feasibility
gate in the last section is unresolved and blocks implementation.

## Why

The 25-turn run `analysis25b-20260910` (see STATUS.md 2026-09-10) exposed the gap
concretely. At Behind House the arbiter chose `OPEN WINDOW`; the game said *"you
open the window far enough to allow entry."* The next turn the window's IssueAgent
proposed **`CLOSE WINDOW`**, the IssueClosedAgent retired the issue quoting "allow
entry" as proof it was done, and the arbiter picked `NORTH`. No agent ever proposed
entering. Score stayed 0 — not because the opening was never found, but because no
drive in the roster was aimed *through* it.

Two findings, in the architecture's own terms (the agents ARE the competing drives):

1. **The roster of drives is a start, not a complete set.** Nothing advocates "get
   inside the house."
2. **Issues are authored as leaf actions, not aims.** The Observer writes the thing
   to *do* ("open the window"), so the one drive with momentum toward the house
   terminates — or reverses (`CLOSE WINDOW`) — the instant its action completes,
   exactly when the payoff is one step away.

This proposal is the candidate fix.

## Core distinction: a fact is not a goal

The troll is not an issue *or* a goal. It is a **fact**: a hostile thing that blocks
the bridge and can kill you. From one fact, several goals can be born — *get past
it*, *kill it*, *avoid it*, *bribe it* — and the same fact feeds two different drives
at once (survive; make progress). What the system calls "issues" today are **facts**
(the Observer noticed a window, a troll, a grating); they were never goals. The
fuzziness between "issue" and "goal" dissolves once facts and goals are separate
things: facts are observations, goals are intended target states derived from facts +
the objective.

## A goal is an agent with a lifecycle — a Meeseeks

A goal-agent is spawned to achieve **one target state**, advocates toward it, and
**terminates itself when the state is reached** — like a Rick-and-Morty Meeseeks,
which exists only to complete its task and then *poofs*.

Lifecycle, with three exits:

- **Born** — created with a target *state*, not an action ("be inside the house", not
  "open the window").
- **Advocates** — each turn it is unblocked, it proposes the next concrete action
  toward its state. This is the one operation that already works: it is what the
  current IssueAgent does, and qwen produced `OPEN WINDOW` correctly.
- **Completes → self-terminates.** The goal owns its death test, and the test is a
  **state** check that is usually deterministic and needs no LLM: score increased,
  location changed, troll gone. This is what fixes the window — "be inside" dies on
  `location == inside`, never on "I opened something", so it can never propose
  `CLOSE WINDOW`.
- **Gives up → starves.** A goal that *cannot* complete is the dangerous case — the
  Meeseeks that can't finish and goes insane; in our run it was the turn-11 deadlock
  alternating two refused commands. The goal does **not** decide to quit (it never
  will). The world stops feeding it: **decay + repetition-suppression are the pain
  that grows the longer it exists**, and eventually starve it out. Completion is
  self-death; give-up is death from below. We already have the second mechanism.
- **Spawns → goes dormant.** See recursion.

## Recursion: spawning is decomposition

A Meeseeks that hits a precondition it cannot meet **spawns a child goal and waits**.
"Kill the troll" with no weapon spawns "find a weapon"; "cross the bridge" spawns
"kill the troll" as a method. This is the metaphor's own logic (Meeseeks spawn
Meeseeks), and it does real work:

- **It is the answer to "who creates sub-goals":** parents do, in context, which is
  the tractable version of goal creation — decomposing a known parent beats inventing
  an aim from nothing.
- **It gives "give up" somewhere to travel.** A child that exhausts its options
  *poofs reporting failure upward*; the parent then tries a **sibling** ("bribe the
  troll") before giving up itself. Failure propagates up the tree instead of a single
  goal spinning forever.
- **Altitude falls out of the tree.** "Cross the bridge" (aim) is the parent; "kill
  the troll" (method) is a child. The parent dies only when you're across; the child
  dies when the troll is dead. Choosing the target state's altitude = choosing where
  in the tree a node sits.

**The arbiter is unchanged, and this is what preserves the thesis.** Only **actionable
leaves advocate** each turn; a parent waiting on a child is **dormant** and proposes
nothing. So the ballot is the current frontier of the goal tree — the unblocked
Meeseeks that want to act now. The tree does **generation** (what goals exist, what
gets spawned next); the arbiter still does **selection** (picks among competing
actionable leaves). The competing drives are now the *leaves of a living tree*, not a
flat list. Advocacy + arbitration is untouched — it is fed better.

## Two populations of drives

- **Innate / standing:** *explore* and *survive (don't die)*. Always on, never
  created, never terminate. (ExplorerAgent is the first of these.)
- **Dynamic / spawned:** everything else — born from facts + objective (roots) or from
  parents (children), each with the lifecycle above.

## Who creates goals — split to make it tractable

- **Roots** are seeded from the world: the Observer's facts + the game objective ("get
  inside", "cross the bridge"). The Observer's job shifts from "flag a fact to act on"
  to "name an *aim* worth rooting."
- **Children** are spawned by parents on an unmet precondition (decomposition in
  context).

This splits the crushing "invent a goal from nothing" burden: the Observer only has to
recognize root-worthy aims; everything below is contextual decomposition.

## The feasibility gate (UNRESOLVED — blocks build)

**Serious open concern: owning and spawning goals may be beyond qwen3:14b.** Goal
*creation* has historically been the hardest thing to get the local LLM to do — it
reflexively emits the next *action* ("open the window") instead of an *aim* ("be
inside"). If the weak model can't do the two hard operations, the architecture
relocates the intelligence into a step the target model can't perform, and the thesis
("can *much weaker* models solve long-horizon tasks via this architecture") eats
itself.

The qwen-dependent surface is only two operations (the rest of the lifecycle is
deterministic and off the LLM):

1. **Create a root goal as a target state**, not an action.
2. **Decide to spawn + name the sub-goal** when blocked.

**Resolve this by measurement before building.** Probe qwen3:14b (and the slower
qwen3.8:27b for a capability-vs-size comparison) on those two operations against the
real Behind-House context, temperature 0, N samples, and count how often it produces a
target-state goal vs. an action, and a coherent sub-goal vs. mush. A code guard that
rejects verb-shaped goals (the project's own "prompt is not a mechanism" rule) may drag
a borderline model over the line.

**Fallback if qwen cannot:** split roles by capability — a stronger model does the
*rare* hard operations (root creation, spawn decisions), qwen does the *frequent*
cheap advocacy. Infra already supports per-instance providers (openai/vllm;
VersionThree). This shifts the claim from "weak models can" to "weak *workers* under
occasional strong *planning* can" — which is a legitimate, informative result about
*where* the capability floor is, not a failure.

## Deliberately undecided

- Whether the missing behaviour is a **new drive** or an **existing drive reaching too
  literally** (the InteractionAgent's drive "engage the window" is arguably right, but
  it only emits the backend's object-verbs — open/close/examine — never "enter").
- The altitude convention for roots (how abstract an aim the Observer should name).
- The exact starvation threshold that declares a leaf hopeless.
- Whether qwen3:14b clears the feasibility gate at all — pending the probe.
