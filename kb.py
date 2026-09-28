"""Agent knowledge representation: facts and a fact store.

The point of this module is to stop representing an
agent's situation with ad-hoc Python variables and start representing it
*explicitly*, as ground facts that rules can inspect.

Two concepts live here:

* ``Fact``          -- an immutable ground assertion: ``predicate(args...)``
* ``KnowledgeBase`` -- a set of facts with add / remove / query operations

Nothing in this module knows about the world, agents, or planning.  It is
purely the substrate that future Step's forward-chaining reasoner will operate on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple, Union


@dataclass(frozen=True)
class Fact:
    """A ground piece of knowledge.

    Examples::

        Fact("at", ("Rescue-1", "E"))
        Fact("road_open", ("E", "F"))
        Fact("needs_medical", ("civ1",))
        Fact("task_pending", ("deliver_medicine_D",))

    Facts are frozen (hashable) so a KnowledgeBase can be a plain set, which
    makes duplicate detection and removal trivial.
    """

    predicate: str
    args: Tuple = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(str(a) for a in self.args)})"


# A fact may be written either as a Fact or as a plain (predicate, args) tuple.
FactLike = Union[Fact, Tuple[str, Sequence]]


def _coerce(fact: FactLike) -> Fact:
    if isinstance(fact, Fact):
        return fact
    predicate, args = fact
    return Fact(predicate, tuple(args))


class KnowledgeBase:
    """A set of ground facts an agent currently believes.

    Deliberately minimal: add, remove, clear, contains, query, iterate.  All
    reasoning (deriving *new* facts) belongs to ``reasoner.py``.
    """

    def __init__(self, facts: Iterable[FactLike] = ()) -> None:
        self._facts: set[Fact] = set()
        for fact in facts:
            self.add(fact)

    # ------------------------------------------------------------------ #
    #  Mutation
    # ------------------------------------------------------------------ #

    def add(self, fact: FactLike) -> bool:
        """Add a fact.  Returns True if it was not already known."""
        f = _coerce(fact)
        if f in self._facts:
            return False
        self._facts.add(f)
        return True

    def remove(self, fact: FactLike) -> bool:
        """Remove a fact.  Returns True if it was present."""
        f = _coerce(fact)
        if f not in self._facts:
            return False
        self._facts.discard(f)
        return True

    def clear(self) -> None:
        self._facts.clear()

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def contains(self, fact: FactLike) -> bool:
        return _coerce(fact) in self._facts

    def query(self, predicate: str, *args) -> List[Fact]:
        """Return every fact with this predicate whose args match.

        ``None`` in ``args`` acts as a wildcard::

            kb.query("at", "Rescue-1", None)   # where is Rescue-1?
            kb.query("road_open", None, "F")   # what leads to F?
            kb.query("needs_medical", None)    # who needs medical help?
        """
        matches = []
        for f in self._facts:
            if f.predicate != predicate or len(f.args) != len(args):
                continue
            if all(want is None or want == got for want, got in zip(args, f.args)):
                matches.append(f)
        return sorted(matches, key=str)

    def facts(self) -> List[Fact]:
        """All known facts, sorted for stable display."""
        return sorted(self._facts, key=str)

    # ------------------------------------------------------------------ #
    #  Dunder sugar
    # ------------------------------------------------------------------ #

    def __contains__(self, fact: FactLike) -> bool:
        return self.contains(fact)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.facts())

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"KnowledgeBase({len(self._facts)} facts)"