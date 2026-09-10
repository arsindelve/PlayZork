# Goal-Agents ("Meeseeks") — a recursive, self-terminating goal layer

**Status:** Proposal. Not built. Design + decision log as of 2026-09-10. The core loop
(create goal + closing condition, persist, advocate the follow-through) is **validated
on a traversal puzzle** with qwen3:14b — it took the exact run failure from score 0 to
score 10 end-to-end (see the feasibility section). What remains unproven is the
**binding** part: admission control under the ~25-goal budget (when to create a goal at
all), and altitude/grounding across non-traversal puzzles. The design was reached
through the reasoning recorded in the Decision Log at the bottom — read that to
understand *why* each choice was made, not just what it is.

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
   to *do* ("open the window"), so the one drive with momentum terminates — or
   reverses (`CLOSE WINDOW`) — the instant its action completes, exactly when the
   payoff is one step away.

This proposal is the candidate fix.

## Core distinction: a fact is not a goal

The troll is not an issue *or* a goal. It is a **fact**: a hostile thing that blocks
the bridge and can kill you. From one fact, several goals can be born — *get past
it*, *kill it*, *avoid it*, *bribe it* — and the same fact feeds two different drives
(survive; make progress). What the system calls "issues" today are **facts** (the
Observer noticed a window, a troll, a grating); they were never goals. Facts are
observations; goals are intended **target states** derived from facts + the objective.

## A goal is an agent with a lifecycle — a Meeseeks

A goal-agent is spawned to achieve **one target state**, advocates toward it, and
**terminates when the state is reached** — like a Rick-and-Morty Meeseeks, which
exists only to complete its task and then *poofs*.

- **Born** — created with a target *state*, not an action ("be inside the house",
  not "open the window").
- **Advocates** — each turn it is unblocked, it proposes the next concrete action
  toward its state. This is the operation that already works: it is what the current
  IssueAgent does, and qwen produced `OPEN WINDOW` correctly.
- **Completes → self-terminates** on a **state test** that is usually deterministic
  and needs no LLM: score increased, location changed, object gone. This is what
  fixes the window — "be inside" dies on `location == inside`, never on "I opened
  something", so it can never propose `CLOSE WINDOW`.
- **Gives up → starves** (see budget/eviction below). A goal that cannot complete is
  the dangerous case — the Meeseeks that can't finish and goes insane; in our run it
  was the turn-11 deadlock. The goal does **not** decide to quit. Decay lowers its
  rank so it stops winning votes; eviction-at-cap reclaims its slot. Give-up is death
  from outside, not a self-diagnosis.
- **Spawns → goes dormant** (see recursion).

## Recursion: spawning is decomposition

A Meeseeks that hits a precondition it cannot meet **spawns a child goal and waits**.
"Kill the troll" with no weapon spawns "find a weapon"; "cross the bridge" spawns
"kill the troll" as a method. This does real work:

- **It is the answer to "who creates sub-goals":** parents do, in context —
  decomposing a known parent beats inventing an aim from nothing.
- **It gives "give up" somewhere to travel.** A child that exhausts its options
  *poofs reporting failure upward*; the parent then tries a **sibling** ("bribe the
  troll") before giving up. Failure propagates up the tree.
- **Altitude falls out of the tree.** "Cross the bridge" (aim) is the parent; "kill
  the troll" (method) is a child. The parent dies only when you're across; the child
  when the troll is dead.

**The arbiter is unchanged, and this preserves the thesis.** Only **actionable leaves
advocate** each turn; a parent waiting on a child is **dormant** and proposes nothing.
The ballot is the current frontier of the goal tree. The tree does **generation**
(what goals exist, what gets spawned next); the arbiter still does **selection**. The
competing drives are the *leaves of a living tree*, not a flat list. Advocacy +
arbitration is fed better, not changed.

## The single tree, importance, and the budget

**There is one tree, rooted at the game objective.** Every goal is a descendant of
the objective (directly, or via a parent that spawned it). This is what makes
importance meaningful.

**Importance is relative to the parent, never absolute.** Asking the model for an
absolute 1–1000 is meta — it can't rank a goal against a landscape it cannot see. In
a tree it never has to:

- **Roots** are weighed against the **objective** (the fixed, given reference): "does
  this state advance the objective?"
- **Children** *derive* importance from their parent (inherit, possibly discounted) —
  "find a weapon" is worth what "kill the troll" is worth. The model's judgment is
  always **local** (root vs. objective; child vs. its one parent), never global. That
  also shrinks the qwen-dependent surface, which matters for the feasibility gate.
- **Effective importance = the chain from the objective down to the node.** That is
  what ranking and eviction use.

**Budget: ~25 goals maximum.** Creation is therefore *relative admission*, not
absolute judgment: admit a new goal only if its effective importance beats the
**weakest current slot**. Decay lowers the floor over time so stale goals evict
themselves; a full budget forces every newcomer to beat an incumbent. "When NOT to
make a goal" ≈ "it would not survive the slot economy."

**Death propagates through the tree, for free.** Parent completes or is evicted → its
children are orphaned → their derived importance collapses → they starve on the next
cap pressure. The Meeseeks poofs and the sub-Meeseeks it spawned poof with it. No
separate cleanup bookkeeping.

**Bias inversion — state this loudly.** Everywhere in the world model the invariant is
*false-negative beats false-positive; record nothing when in doubt.* Under a goal cap
this **flips**: a spurious goal is now directly expensive because it **evicts a real
one**. So goal admission must be *conservative* — the opposite bias from the rest of
the system. Someone will try to "fix" it back; don't let them.

Known risks / wrinkles:

- **Decay can evict a goal that was important but merely early** — "kill the troll"
  with no weapon yet, decaying out before the sword is found. Recursion cushions this
  (an active child props the parent's importance), but a blocked goal with no live
  child is exactly the one that starves — right if truly stuck, wrong if the
  precondition was around the corner. Decay rate is the tuning knob.
- **Sibling ranking** still needs a local judgment (two children of one parent), but
  it is small — comparing two things in one context, not the universe.
- **Shared sub-goals** ("find a weapon" serves troll *and* cyclops): in a strict tree
  it belongs to one parent and dies with it; if it matters elsewhere it is re-rooted
  when re-observed. Accept occasional re-creation rather than build a DAG.

## Goal identity and duplication

The budget does **not** solve duplication — decay and eviction treat two copies as two
goals (they decay in parallel, and eviction might bump the *original* and keep the
*dup*). Under a cap a duplicate is doubly costly: a second slot for one aim, and it
splits importance/advocacy into two half-strength goals that can thrash. Dedup is a
separate, mandatory mechanism, and it matters more here than in the old flat list.

The goal-as-state reframe gives a **better dedup key than text**:

- **Key on the target state / referent, not the sentence.** "I am inside the house"
  and "enter the white house" are the same goal because they resolve to the same
  target (location ∈ inside-rooms). A goal's identity ≈ **(referent, aim)**, where the
  referent is a location/object. This is exact and needs no LLM, replacing today's
  fuzzy LLM semantic dedup.
- **It kills the biggest practical source: re-visit re-flagging.** The agent loops
  through rooms; each re-entry the Observer re-sees the window and wants to re-create
  the goal. Referent-keyed dedup — checking **open *and* recently-closed** goals for
  that referent before creating — stops it cold. (Keep the current comparison against
  closed issues; it is what prevents re-flagging "take the leaves" after you took
  them.)
- **Subsumption** is the dup type recursion introduces. "open the window" is not
  textually a duplicate of "get inside the house," but it is a *sub-goal* of it — if
  the Observer roots it independently while "get inside" exists (and may already have
  spawned it), you get a redundant slot. So the check is not just "same referent?" but
  "is this referent/aim already **covered** by an existing goal *or one of its
  children*?" The parent/child links make this answerable.

**The admission gate, assembled** (LLM does only steps 1 and 3):

1. **Game-signaled actionability**, not scenery — a locked/blocked thing, a takeable
   item, a named-but-unentered exit, a hazard, an explicit refusal. "There's a rug" →
   no goal; "there's a *locked* grating" → candidate. Grounded in an observed signal,
   not predicted significance (which is unreliable and makes the LLM over-flag). *[LLM
   or heuristic]*
2. **Not already covered** by an existing goal or its children (referent/state-keyed,
   incl. recently-closed). *[deterministic]*
3. **Model weighs importance** — relative to parent/objective. *[LLM, local]*
4. **Admit if it beats the weakest slot;** decay + eviction do the rest.
   *[deterministic]*

## Creation responsibility, split

- **Roots** seeded from the world: the Observer's facts + the game objective. The
  Observer's job shifts from "flag a fact to act on" to "name an *aim* worth rooting"
  — and, under the budget, to *decline* most facts.
- **Children** spawned by parents on an unmet precondition (decomposition in context).

## The feasibility gate — the loop is validated on a traversal puzzle (2026-09-10)

**Original concern: owning and spawning goals may be beyond qwen3:14b** (it has
historically emitted actions, not aims). Probed against the **real Behind House
backend metadata** (temp 0 + temp 0.7 samples, think:false). Result: the core loop
works reliably on this puzzle; the open risks are elsewhere.

1. **Bare-scenario goal (terse prompt, "target state" only).** 5/5 state-shaped, zero
   imperatives — but altitude wobbled: on the window it said *"the window is fully
   open"* (the leaf), not "I am inside." Misleading in isolation — see (2).

2. **Grounded goal + closing condition (real room payload).** Asked for a GOAL *and* a
   CLOSING condition, given the actual description + exits + accepted-commands:
   **11/12 produced the get-inside goal with an inside-STATE closing condition**
   ("…you are inside the house"), across both the thin real objective ("reach 350")
   and a rich one. 1/12 hallucinated a "key" in the closing (grounding failure).
   **Key lesson: ask for the goal *and its closing condition*, and let altitude live in
   the closing condition — qwen writes that reliably even when the goal label reads
   action-y ("open the window to gain access…").** A closing of "you are inside" does
   NOT fire on window-open — the exact fix for the run's bug.

3. **Follow-through advocacy.** With the goal "I am inside" still open and the window
   now open, **6/6 proposed `ENTER WINDOW`** — the follow-through no drive produced in
   the live run.

4. **End-to-end against the live backend.** `open window` → `enter window` →
   **Kitchen, score 10** — Zork's first points. The precise `analysis25b` failure
   (open → `CLOSE WINDOW` → walk away → score 0), reproduced as a *fix*, with
   qwen3:14b driving every LLM step reliably.

**Grounding note (matters for the roster):** after opening, the "entry is possible"
signal appears in **`exits`** (it gained W=3 and 8="In"), NOT in
`actionsAvailableFromLocation` (still only open/close/examine window). So the
InteractionAgent (object-verbs) structurally cannot propose entering; a movement/goal
drive reading `exits`/location can. Closing conditions and advocacy should read
`exits`/location, not the object-command list.

**What this settles and what it does not.** It validates the *loop mechanics* — create
goal + closing, persist while closing unmet, advocate the follow-through — and shows
qwen3:14b can drive each LLM step reliably **on this (traversal) puzzle**. It does NOT
settle: (a) **admission control** — when to create a goal vs not, under the ~25 budget
(the actual binding constraint; untested here); (b) altitude/closing quality on
**non-traversal** puzzles; (c) **grounding robustness** (1/12 hallucinated a closing);
(d) the **qwen3.8:27b** comparison. The relative-importance + mostly-deterministic
admission design keeps the per-step LLM surface small, which is why the 14B has a
chance. Probe scripts: `nav_zork.py`, `qwen_goal_probe3.py`, `qwen_advocacy_probe.py`
(session scratchpad).

**Fallback if qwen cannot:** a stronger model does the *rare* hard operations (root
creation, spawn decisions), qwen does the *frequent* cheap advocacy. Infra already
supports per-instance providers. This shifts the claim from "weak models can" to "weak
*workers* under occasional strong *planning* can" — a legitimate, informative result
about *where* the capability floor is, not a failure.

## Deliberately undecided

- Whether the missing behaviour is a **new drive** or an **existing drive reaching too
  literally** (InteractionAgent emits only backend object-verbs — never "enter").
- The **altitude convention** for roots (how proximate an aim the Observer should name)
  — the one thing the probe shows qwen wobbles on.
- The **decay rate** that declares a blocked leaf hopeless without evicting the merely
  early.
- **Admission control** under the ~25 budget — when to create a goal vs decline. This
  is the binding constraint and is untested; the run over-created trivia (mailbox,
  leaves).
- Altitude/closing quality on **non-traversal** puzzles, and **grounding robustness**
  (closing conditions must be checkable against real state; 1/12 hallucinated one).
- The **qwen3.8:27b** comparison, when the author points at a host. (The loop itself is
  no longer in doubt on the 14B for a traversal puzzle — see feasibility section.)

---

## Decision log — how this design was reached (2026-09-10)

Recorded because the wrong turns are where the decisions actually happened.

1. **The finding.** A 25-turn local run scored 0. Forensics: the agent opened the
   Behind-House window (the game said "…allow entry"), then the window's own IssueAgent
   proposed `CLOSE WINDOW`, the IssueClosedAgent retired the issue citing "allow entry",
   and the arbiter walked `NORTH`. Nobody proposed entering.

2. **Wrong framing #1 (assistant), corrected (author).** The assistant first called
   this a "generation gap" and drifted to "maybe the thesis is really about generation,
   not deliberation." The author corrected two errors: external-memory was a *motivation*,
   never the thesis; the thesis was and is *"can multi-agent deliberation let much weaker
   models solve long-horizon tasks."* And the finding is not a challenge to it — adequate
   candidate generation is a *precondition* for the comparison to be interpretable, the
   same category as the earlier state-correctness work. "Generation" was dropped as
   jargon; the frame is the author's: **the agents ARE the competing drives; the roster
   is incomplete.**

3. **Wrong turn #2 (assistant), rejected (author).** The assistant proposed a surgical
   rule: "when an interaction opens a traversal, emit the movement." The author rejected
   it as pattern-matching one example ("open = progress" is not universal). Correct: it
   solved the window, not the class.

4. **Fact vs goal (author).** The troll is a *fact*; facts spawn multiple goals and feed
   multiple drives. Today's "issues" are facts, never goals. This dissolved the
   issue/goal fuzziness.

5. **Meeseeks (author).** Goals should be their own agents with a lifecycle that
   *self-terminates on completion* — a Meeseeks.

6. **Recursion (assistant).** Spawning = decomposition; failure propagates up; parents
   go dormant and only leaves advocate — so generation lives in the tree and the arbiter
   still only selects. The author flagged this as resolving several concerns.

7. **Feasibility worry (author) + probe (assistant).** The author worried owning/spawning
   goals is beyond qwen. Probing qwen3:14b: state-shaping is fine; **altitude and
   grounding are the shaky parts** (window → leaf; hallucinated door key).

8. **Over-indexing on the window (author).** The window is the *traversal* puzzle type,
   a minority. Acquisition / deposit / hazard puzzles — most of the score — have no
   altitude fork; goal-as-state is clean there. So the window's altitude wrinkle is not
   emblematic.

9. **The real challenge (author): when to make a goal, not which.** With a hard cap
   (~25 goals), admission control is the binding constraint — and the run already
   over-creates trivia (mailbox, leaves as goals). Consequence: the false-positive bias
   *inverts* here (a spurious goal evicts a real one), and admission should gate on
   *game-signaled actionability*, not predicted significance.

10. **Importance "against what?" (author) → single tree (assistant+author).** An absolute
    importance score is meta — the model can't rank against an invisible landscape.
    Resolution: importance is relative to the *parent*; roots anchor to the *objective*;
    one tree rooted at the objective makes "against what" permanently answerable, and
    death/eviction propagate down it for free.

11. **Duplication (author).** The budget does not dedup. The goal-as-state reframe gives
    a state/referent dedup key (better than LLM text-matching); recursion adds the
    subsumption case (a candidate root already covered by an existing goal or its
    children). Re-visit re-flagging is the main practical source.

Open at end of session: the feasibility gate (structured retest of qwen; qwen3.8
comparison), altitude convention, decay rate, and the new-drive-vs-widened-drive
question.
