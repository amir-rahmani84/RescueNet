"""Rule representation and the default rule set.

A rule is written as::

    IF   <premise 1>
         <premise 2>
         ...
    THEN <conclusion>

Premises and conclusions are ``Fact`` *patterns* whose arguments may be
variables -- strings beginning with ``?``.  A pattern matches a ground fact
when every constant lines up and every variable is bound consistently.

This module provides:

* ``Rule``              -- name + premises + conclusion
* ``match_fact``        -- unify one pattern against one ground fact
* ``match_premises``    -- find every binding satisfying a premise list
* ``instantiate``       -- substitute a binding into a conclusion pattern
* ``DEFAULT_RULES``     -- the rule set the agents reason with

The *forward-chaining engine* that applies these rules repeatedly until no
new facts appear is in ``reasoner.py``.  Everything needed for it is
already here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

from kb import Fact, KnowledgeBase


Bindings = Dict[str, object]


# --------------------------------------------------------------------------- #
#  Variables and unification
# --------------------------------------------------------------------------- #

def is_variable(term: object) -> bool:
    return isinstance(term, str) and term.startswith("?")


def match_fact(
    pattern: Fact,
    fact: Fact,
    bindings: Optional[Bindings] = None,
) -> Optional[Bindings]:
    """Try to unify ``pattern`` with ground ``fact``.

    Returns an extended binding dict on success, ``None`` on failure.  A
    variable already present in ``bindings`` acts as an equality constraint.
    """
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
    """Substitute ``bindings`` into ``pattern`` to produce a ground fact."""
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
    """Yield every binding under which *all* premises hold in ``kb``.

    Plain backtracking search over the premise list: pick a fact that
    satisfies the first premise, extend the bindings, recurse on the rest.
    """
    if not premises:
        yield dict(bindings or {})
        return

    head, tail = premises[0], premises[1:]
    for fact in kb.query(head.predicate, *([None] * len(head.args))):
        extended = match_fact(head, fact, bindings)
        if extended is not None:
            yield from match_premises(tail, kb, extended)


# --------------------------------------------------------------------------- #
#  Rules
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rule:
    """IF ``premises`` THEN ``conclusion``."""

    name: str
    premises: Tuple[Fact, ...]
    conclusion: Fact

    def fires(self, kb: KnowledgeBase) -> Iterator[Bindings]:
        """Yield every binding under which this rule's premises are satisfied."""
        yield from match_premises(self.premises, kb)

    def derive(self, kb: KnowledgeBase) -> List[Fact]:
        """Return the ground conclusions this rule can draw from ``kb``."""
        return [instantiate(self.conclusion, b) for b in self.fires(kb)]


def _f(predicate: str, *args) -> Fact:
    return Fact(predicate, tuple(args))


# --------------------------------------------------------------------------- #
#  The default rule set
# --------------------------------------------------------------------------- #

DEFAULT_RULES: Tuple[Rule, ...] = (
    # --- Reachability -------------------------------------------------- #
    Rule(
        name="reach_self",
        premises=(_f("at", "?agent", "?loc"),),
        conclusion=_f("can_reach", "?agent", "?loc"),
    ),
    Rule(
        name="reach_step",
        premises=(
            _f("can_reach", "?agent", "?loc"),
            _f("road_open", "?loc", "?next"),
        ),
        conclusion=_f("can_reach", "?agent", "?next"),
    ),

    # --- Capability matching ------------------------------------------- #
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

    # --- Task assignment ------------------------------------------------ #
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

    # --- Direct action eligibility -------------------------------------- #
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