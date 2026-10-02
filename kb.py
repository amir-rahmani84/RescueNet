"""Agent knowledge representation: facts and a fact store.

A KnowledgeBase is now indexed by predicate and (predicate, first argument),
so `query` no longer has to scan every fact. This is the single biggest
constant-factor win across rules, actions, and planning, because every
unification step goes through `query`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple, Union


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: Tuple = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(str(a) for a in self.args)})"


FactLike = Union[Fact, Tuple[str, Sequence]]


def _coerce(fact: FactLike) -> Fact:
    if isinstance(fact, Fact):
        return fact
    predicate, args = fact
    return Fact(predicate, tuple(args))


class KnowledgeBase:
    """Indexed set of ground facts.

    Indices:
        _by_pred          : predicate -> facts with that predicate
        _by_pred_arg0     : (predicate, first_arg) -> facts

    The arg0 index is what makes `query("at", "Rescue-1", None)` O(bucket)
    instead of O(total facts). Many hot paths use exactly that shape.
    """

    __slots__ = ("_facts", "_by_pred", "_by_pred_arg0")

    def __init__(self, facts: Iterable[FactLike] = ()) -> None:
        self._facts: Set[Fact] = set()
        self._by_pred: Dict[str, Set[Fact]] = {}
        self._by_pred_arg0: Dict[Tuple[str, object], Set[Fact]] = {}
        if facts:
            self.add_many(facts)

    @classmethod
    def from_facts(cls, facts: Iterable[Fact]) -> "KnowledgeBase":
        """Build an indexed KB from a known-unique iterable of Facts."""
        kb = cls.__new__(cls)
        kb._facts = set(facts)
        kb._by_pred = {}
        kb._by_pred_arg0 = {}
        for fact in kb._facts:
            kb._by_pred.setdefault(fact.predicate, set()).add(fact)
            if fact.args:
                kb._by_pred_arg0.setdefault(
                    (fact.predicate, fact.args[0]), set()
                ).add(fact)
        return kb

    # ------------------------------------------------------------------ #
    #  Mutation
    # ------------------------------------------------------------------ #

    def add(self, fact: FactLike) -> bool:
        f = _coerce(fact)
        if f in self._facts:
            return False
        self._facts.add(f)
        self._by_pred.setdefault(f.predicate, set()).add(f)
        if f.args:
            self._by_pred_arg0.setdefault(
                (f.predicate, f.args[0]), set()
            ).add(f)
        return True

    def add_many(self, facts: Iterable[FactLike]) -> int:
        added = 0
        for fact in facts:
            if self.add(fact):
                added += 1
        return added

    def remove(self, fact: FactLike) -> bool:
        f = _coerce(fact)
        if f not in self._facts:
            return False
        self._facts.discard(f)
        bucket = self._by_pred.get(f.predicate)
        if bucket is not None:
            bucket.discard(f)
            if not bucket:
                del self._by_pred[f.predicate]
        if f.args:
            key = (f.predicate, f.args[0])
            bucket0 = self._by_pred_arg0.get(key)
            if bucket0 is not None:
                bucket0.discard(f)
                if not bucket0:
                    del self._by_pred_arg0[key]
        return True

    def remove_matching(self, predicate: str, *args) -> int:
        """Bulk removal using the query pattern. Returns how many were removed."""
        matches = self.query(predicate, *args)
        removed = 0
        for fact in matches:
            if self.remove(fact):
                removed += 1
        return removed

    def clear(self) -> None:
        self._facts.clear()
        self._by_pred.clear()
        self._by_pred_arg0.clear()

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def contains(self, fact: FactLike) -> bool:
        return _coerce(fact) in self._facts

    def has_predicate(self, predicate: str) -> bool:
        """Return whether any fact has this predicate."""
        return bool(self._by_pred.get(predicate))

    def query(self, predicate: str, *args) -> List[Fact]:
        """Return every fact matching the pattern (None = wildcard)."""
        # Fast path: use the tightest index available.
        if args and args[0] is not None:
            source = self._by_pred_arg0.get((predicate, args[0]))
        else:
            source = self._by_pred.get(predicate)
        if not source:
            return []

        # Fast path: fully ground query.
        if args and all(a is not None for a in args):
            target = Fact(predicate, tuple(args))
            return [target] if target in source else []

        if not args:
            return list(source)

        wanted_len = len(args)
        matches: List[Fact] = []
        for f in source:
            if len(f.args) != wanted_len:
                continue
            if all(w is None or w == g for w, g in zip(args, f.args)):
                matches.append(f)
        return matches

    def facts(self) -> List[Fact]:
        """All facts, unsorted, for set-oriented internal operations."""
        return list(self._facts)

    def sorted_facts(self) -> List[Fact]:
        """All facts sorted for stable human-readable display."""
        return sorted(self._facts, key=str)

    # ------------------------------------------------------------------ #
    #  Dunder sugar
    # ------------------------------------------------------------------ #

    def __contains__(self, fact: FactLike) -> bool:
        return self.contains(fact)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"KnowledgeBase({len(self._facts)} facts)"