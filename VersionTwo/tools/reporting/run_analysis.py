"""Summarise a completed run: progress, cost, and where it went wrong.

Reading a 50-turn session by scrolling the log does not scale, and the things
worth noticing — a score that stops moving, a command issued five times, an
agent whose proposals are never chosen — are exactly the things a human skims
past. This extracts them.

Doubles as the experiment's reporting layer: `score@turns`, `score@wall-clock`
and `score@tokens` for a run, which is what any comparison between conditions
needs (see PLAN.md).
"""
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)")


def _ts(line: str) -> Optional[datetime]:
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


@dataclass
class Turn:
    number: int
    command: str = ""
    response: str = ""
    location: str = ""
    score: int = 0
    started: Optional[datetime] = None
    seconds: float = 0.0
    llm_calls: int = 0


@dataclass
class Decision:
    """One turn's proposals and what the arbiter did with them."""

    turn: int
    proposals: List[Tuple[str, str, float]] = field(default_factory=list)  # agent, action, EV
    chosen: str = ""
    reason: str = ""

    @property
    def top_ev(self) -> Optional[Tuple[str, str, float]]:
        return max(self.proposals, key=lambda p: p[2]) if self.proposals else None

    @property
    def overrode_top_ev(self) -> bool:
        """Did the arbiter pass over the highest-scoring proposal?

        This is the load-bearing question for the architecture. If the answer
        is always no, the LLM arbiter is an expensive `max()` and the
        deliberation is decorative — which is exactly what the arbitration
        ablation is meant to test.
        """
        top = self.top_ev
        if not top or not self.chosen:
            return False
        return _norm(self.chosen) != _norm(top[1])


def _norm(command: str) -> str:
    return " ".join((command or "").upper().split())


@dataclass
class RunAnalysis:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    memories_stored: List[str] = field(default_factory=list)
    issues_closed: List[str] = field(default_factory=list)
    deaths: int = 0
    failures: int = 0
    suppressions: int = 0
    map_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    tokens: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    decisions: List[Decision] = field(default_factory=list)

    # ---- arbitration ----------------------------------------------------

    @property
    def contested_turns(self) -> List[Decision]:
        """Turns where more than one agent proposed — the only turns on which
        arbitration could possibly have mattered."""
        return [d for d in self.decisions if len(d.proposals) > 1]

    @property
    def overrides(self) -> List[Decision]:
        return [d for d in self.contested_turns if d.overrode_top_ev]

    @property
    def override_rate(self) -> float:
        contested = self.contested_turns
        return len(self.overrides) / len(contested) if contested else 0.0

    def agent_win_counts(self) -> Counter:
        wins = Counter()
        for d in self.decisions:
            for agent, action, _ in d.proposals:
                if _norm(action) == _norm(d.chosen):
                    wins[agent] += 1
                    break
        return wins

    # ---- progress -------------------------------------------------------

    @property
    def final_score(self) -> int:
        return self.turns[-1].score if self.turns else 0

    @property
    def scoring_turns(self) -> List[Turn]:
        """Turns where the score actually moved — the only unambiguous
        evidence of progress the game gives us."""
        out, previous = [], 0
        for t in self.turns:
            if t.score > previous:
                out.append(t)
            previous = max(previous, t.score)
        return out

    @property
    def wasted_turns(self) -> int:
        """Turns whose command was issued earlier in the same room and whose
        response was identical — provably no new information."""
        seen, wasted = set(), 0
        for t in self.turns:
            key = (t.location.casefold(), " ".join(t.command.upper().split()),
                   t.response.strip()[:120])
            if key in seen:
                wasted += 1
            seen.add(key)
        return wasted

    def repeated_commands(self, minimum: int = 3) -> List[Tuple[str, str, int]]:
        counts = Counter((t.location, " ".join(t.command.upper().split()))
                         for t in self.turns)
        return [(loc, cmd, n) for (loc, cmd), n in counts.most_common() if n >= minimum]

    @property
    def distinct_locations(self) -> int:
        return len({t.location.casefold() for t in self.turns if t.location})

    # ---- cost -----------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return sum(i + o for i, o, _ in self.tokens.values())

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.turns)

    def summary(self) -> str:
        lines = [
            f"session            {self.session_id}",
            f"turns              {len(self.turns)}",
            f"final score        {self.final_score}",
            f"scoring turns      {len(self.scoring_turns)}"
            + (f"  (turns {[t.number for t in self.scoring_turns]})" if self.scoring_turns else ""),
            f"distinct rooms     {self.distinct_locations}",
            f"map edges          {len(self.map_edges)}",
            f"wasted turns       {self.wasted_turns}"
            + (f"  ({self.wasted_turns / len(self.turns) * 100:.0f}% of the run)" if self.turns else ""),
            f"issues stored      {len(self.memories_stored)}",
            f"issues closed      {len(self.issues_closed)}",
            f"deaths             {self.deaths}",
            f"turn failures      {self.failures}",
            f"repeats suppressed {self.suppressions}",
        ]
        if self.total_seconds:
            lines.append(f"wall clock         {self.total_seconds/60:.1f} min "
                         f"({self.total_seconds/max(1,len(self.turns)):.0f}s/turn)")
        if self.contested_turns:
            lines.append(
                f"contested turns    {len(self.contested_turns)} "
                f"(>1 proposal; arbitration could matter)")
            lines.append(
                f"arbiter overrides  {len(self.overrides)} "
                f"({self.override_rate*100:.0f}% of contested) "
                f"— if 0%, the arbiter is an expensive max()")
            wins = self.agent_win_counts()
            if wins:
                lines.append("agent wins         "
                             + ", ".join(f"{a}={n}" for a, n in wins.most_common()))
        if self.total_tokens:
            lines.append(f"tokens             {self.total_tokens} "
                         f"({self.total_tokens/max(1,len(self.turns)):.0f}/turn)")
            if self.final_score:
                lines.append(f"tokens per point   {self.total_tokens/self.final_score:.0f}")
        return "\n".join(lines)


def analyse(log_path: str, db_path: Optional[str] = None) -> RunAnalysis:
    session = log_path.split("game_")[-1].rsplit(".log", 1)[0]
    run = RunAnalysis(session_id=session)

    current: Optional[Turn] = None
    for line in open(log_path, errors="replace"):
        if m := re.search(r"###  TURN (\d+) START - Command: (.*)", line):
            if current:
                run.turns.append(current)
            current = Turn(number=int(m.group(1)), command=m.group(2).strip(),
                           started=_ts(line))
            continue
        if current is None:
            continue
        if m := re.search(r"Game Response \(first 100\): (.*)", line):
            current.response = m.group(1).strip()
        elif m := re.search(r"Location: (.*)", line):
            if not current.location:
                current.location = m.group(1).strip()
        elif m := re.search(r"Score: (\d+)", line):
            current.score = int(m.group(1))
        elif "POST http://localhost:11434" in line or "POST http" in line and "chat" in line:
            current.llm_calls += 1
        elif "MEMORY STORED" in line:
            if m := re.search(r"MEMORY STORED: \[[^\]]*\] (.*)", line):
                run.memories_stored.append(m.group(1).strip())
        elif "CLOSED ID" in line:
            run.issues_closed.append(line.strip()[-90:])
        elif "Death detected" in line:
            run.deaths += 1
        elif "ALREADY TRIED HERE" in line:
            run.suppressions += 1
        elif re.search(r"Turn \d+ failed", line):
            run.failures += 1
        elif "Agent Proposals:" in line:
            decision = Decision(turn=current.number)
            run.decisions.append(decision)
        elif run.decisions and (m := re.match(
                r"(IssueAgent #\d+|ExplorerAgent|InteractionAgent|LoopDetectionAgent):"
                r".*?(?:EV: ([\d.]+))?\]", line)):
            run.decisions[-1].proposals.append([m.group(1), "", float(m.group(2) or 0)])
        elif run.decisions and run.decisions[-1].proposals and (
                m := re.search(r"Proposed Action: (.*)", line)):
            last = run.decisions[-1].proposals[-1]
            if not last[1]:
                last[1] = m.group(1).strip()
        elif m := re.search(r"DECISION MADE: (.*)", line):
            if run.decisions:
                run.decisions[-1].chosen = m.group(1).strip()
        elif m := re.search(r"REASON: (.*)", line):
            if run.decisions and not run.decisions[-1].reason:
                run.decisions[-1].reason = m.group(1).strip()
    if current:
        run.turns.append(current)
    for d in run.decisions:
        d.proposals = [tuple(p) for p in d.proposals]

    for a, b in zip(run.turns, run.turns[1:]):
        if a.started and b.started:
            a.seconds = (b.started - a.started).total_seconds()

    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            run.map_edges = list(conn.execute(
                "SELECT from_location, direction, to_location FROM map_transitions "
                "WHERE session_id = ?", (session,)))
            run.tokens = {
                t: (i, o, c) for t, i, o, c in conn.execute(
                    "SELECT turn_number, input_tokens, output_tokens, llm_calls "
                    "FROM turn_tokens WHERE session_id = ?", (session,))}
        except sqlite3.Error:
            pass
    return run


if __name__ == "__main__":  # pragma: no cover
    import sys
    log = sys.argv[1]
    db = sys.argv[2] if len(sys.argv) > 2 else "data/zork_sessions.db"
    run = analyse(log, db)
    print(run.summary())
    if repeats := run.repeated_commands():
        print("\nmost repeated (location, command, times):")
        for loc, cmd, n in repeats[:10]:
            print(f"   {n:>3}x  {cmd:28} @ {loc}")
