"""Telling apart two rooms that report the same name (GitHub issue #15).

The mapper identifies a room by its display name, so Zork's several rooms
called "Forest" — and the whole maze, where every room reports the same name —
collapse into one map node with all their exits merged into a fictional
super-room. #10 stopped the worst symptom (successful moves between same-named
rooms are no longer recorded as walls), but the map still cannot represent
those rooms separately.

The backend supplies a usable discriminator: an `exits` array (#30). Two rooms
both called "Forest" report different exit sets — [3,2,1] and [3,0,1] in
observed play — so topology distinguishes them where the name cannot.

WHY EXITS AND NOT THE DESCRIPTION. A description changes as the world changes:
take the lamp and the room stops mentioning it. Fingerprinting on that would
create a phantom new room every time an object moved, fragmenting the map —
far worse than the merging it set out to fix. Exits are topology and change
only when the world's structure does.

BIAS: MERGE, DON'T SPLIT. Exits can still change legitimately — opening the
trap door adds a DOWN exit to the Living Room. So a room matches an existing
one when either exit set contains the other, and the union is kept. A wrong
merge reproduces today's behaviour, which is survivable; a wrong split
fragments the map into rooms that can never be connected, and nothing in the
game would ever reveal the error. Same asymmetry as #11's BLOCKED rule.
"""
from typing import Dict, Iterable, List, Optional, Tuple


def exits_signature(exits: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """Order-independent, hashable form of the backend's exits array."""
    if not exits:
        return ()
    try:
        return tuple(sorted({int(e) for e in exits}))
    except (TypeError, ValueError):
        return ()


def is_compatible(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """True when two exit signatures could describe the same room.

    Containment either way, not equality: a room gains exits when the world
    opens up (the trap door), and demanding equality would split it in two.
    An empty signature is compatible with anything — an unknown discriminator
    must never cause a split.
    """
    if not a or not b:
        return True
    sa, sb = set(a), set(b)
    return sa <= sb or sb <= sa


def label_for(name: str, index: int) -> str:
    """Human-readable label for the index-th room sharing `name`.

    The first keeps the bare name so existing maps, reports and prompts are
    unchanged for the overwhelmingly common case of a uniquely-named room.
    """
    return name if index == 0 else f"{name} #{index + 1}"


class RoomRegistry:
    """Resolves (name, exits) to a stable label, learning as it goes."""

    def __init__(self):
        # name -> list of exit signatures, one per distinct room seen
        self._rooms: Dict[str, List[set]] = {}

    def resolve(self, name: Optional[str], exits: Optional[Iterable[int]] = None) -> str:
        """Return the canonical label for the room we are standing in.

        Without a name there is nothing to resolve; without exits we cannot
        discriminate, so we fall back to the bare name — i.e. exactly the
        pre-#15 behaviour, never worse.
        """
        if not name:
            return ""
        key = " ".join(name.split()).casefold()
        signature = exits_signature(exits)

        known = self._rooms.setdefault(key, [])
        if not signature:
            # No discriminator: assume the first room of this name.
            if not known:
                known.append(set())
            return label_for(name, 0)

        for index, seen in enumerate(known):
            if is_compatible(tuple(sorted(seen)), signature):
                seen.update(signature)  # the world may have opened up
                return label_for(name, index)

        known.append(set(signature))
        return label_for(name, len(known) - 1)

    def distinct_count(self, name: str) -> int:
        """How many separate rooms are known to share this display name."""
        return len(self._rooms.get(" ".join((name or "").split()).casefold(), []))
