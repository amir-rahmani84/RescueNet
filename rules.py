"""Rule representation and the default rule set.

`can_reach` is no longer expressed as recursive rules. It is computed
directly by `reasoner.forward_chain` using BFS on `road_open`, which is
both cheaper and less brittle than repeated transitive-closure firings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

from kb import Fact, KnowledgeBase


Bindings = Dict[str, object]


def is_variable(term: object) -> bool:
    return isinstance(term, str) and term.startswith("?")


def match_fact(
    pattern: Fact,
    fact: Fact,
    bindings: Optional[Bindings] = None,
) -> Optional[Bindings]:
    if pattern.predicate != fact.predicate or len(pattern.args) != len(fact.args):
        return None
    result: Bindings = dict(bindings or {})
    for p, f in zip(pattern.args, fact.args):
        if is_variable(p):
            bound = result.get(p)
            if bound is None:
                result[p] = f
            elif bound != f:
                return None
        elif p != f:
            return None
    return result


def instantiate(pattern: Fact, bindings: Mapping[str, object]) -> Fact:
    args = tuple(
        bindings.get(a, a) if is_variable(a) else a
        for a in pattern.args
    )
    return Fact(pattern.predicate, args)


def match_premises(
    premises: Tuple[Fact, ...],
    kb: KnowledgeBase,
    bindings: Optional[Bindings] = None,
) -> Iterator[Bindings]:
    if not premises:
        yield dict(bindings or {})
        return
    head, tail = premises[0], premises[1:]
    # Push already-bound variables into the query so the index can filter.
    query_args = []
    for a in head.args:
        if is_variable(a) and a in (bindings or {}):
            query_args.append(bindings[a])   # type: ignore[index]
        else:
            query_args.append(None)
    for fact in kb.query(head.predicate, *query_args):
        extended = match_fact(head, fact, bindings)
        if extended is not None:
            yield from match_premises(tail, kb, extended)


@dataclass(frozen=True)
class Rule:
    name: str
    premises: Tuple[Fact, ...]
    conclusion: Fact

    def fires(self, kb: KnowledgeBase) -> Iterator[Bindings]:
        yield from match_premises(self.premises, kb)

    def derive(self, kb: KnowledgeBase) -> List[Fact]:
        return [instantiate(self.conclusion, b) for b in self.fires(kb)]


def _f(predicate: str, *args) -> Fact:
    return Fact(predicate, tuple(args))


# The two reachability rules (reach_self, reach_step) are gone on purpose.
DEFAULT_RULES: Tuple[Rule, ...] = (
    Rule(
        name="capability_match",
        premises=(
            _f("has_capability", "?agent", "?cap"),
            _f("requires_capability", "?task", "?cap"),
        ),
        conclusion=_f("can_perform", "?agent", "?task"),
    ),
    Rule(
        name="medical_agent_can_deliver",
        premises=(
            _f("has_capability", "?agent", "medical"),
            _f("task_type", "?task", "deliver"),
        ),
        conclusion=_f("can_perform", "?agent", "?task"),
    ),
    Rule(
        name="task_assignable",
        premises=(
            _f("task_pending", "?task"),
            _f("can_perform", "?agent", "?task"),
            _f("task_at", "?task", "?loc"),
            _f("can_reach", "?agent", "?loc"),
        ),
        conclusion=_f("can_assign", "?agent", "?task"),
    ),
    Rule(
        name="task_feasible",
        premises=(
            _f("task_pending", "?task"),
            _f("can_assign", "?agent", "?task"),
        ),
        conclusion=_f("task_feasible", "?task"),
    ),
    Rule(
        name="can_treat",
        premises=(
            _f("has", "?agent", "medical_kit"),
            _f("at", "?agent", "?loc"),
            _f("person_at", "?person", "?loc"),
            _f("needs_medical", "?person"),
        ),
        conclusion=_f("can_treat", "?agent", "?person"),
    ),
    Rule(
        name="can_evacuate",
        premises=(
            _f("has_capability", "?agent", "transport"),
            _f("at", "?agent", "?loc"),
            _f("person_at", "?person", "?loc"),
            _f("needs_evacuation", "?person"),
        ),
        conclusion=_f("can_evacuate", "?agent", "?person"),
    ),
    Rule(
        name="can_clear_road",
        premises=(
            _f("has_capability", "?agent", "repair"),
            _f("at", "?agent", "?loc"),
            _f("road_blocked", "?loc", "?next"),
        ),
        conclusion=_f("can_clear_road", "?agent", "?loc", "?next"),
    ),
)