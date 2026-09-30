"""Action representation.

An ``Action`` is a *schema*: a named template with parameters, preconditions,
effects, and a cost.  Variables are written with a leading ``?`` -- exactly
the same convention used in ``rules.py`` -- so an action's preconditions and
effects are ``Fact`` patterns just like rule premises and conclusions.

Binding the variables yields a ``BoundAction`` -- a concrete, ground action
that can be checked against a knowledge base and applied to it:

    Move(?agent, ?from, ?to)
        preconditions:  at(?agent, ?from),  road_open(?from, ?to)
        add effects:    at(?agent, ?to)
        del effects:    at(?agent, ?from)
        cost:           road_time(?from, ?to) + road_risk(?from, ?to)
        guard:          fuel(?agent, f) and road_time(?from, ?to, t) and f >= t

Cost may be a constant *or* a callable ``(kb, bindings) -> float``.  A
``guard`` callable is provided for numeric side-conditions ("enough fuel")
that are awkward to express as pure facts.  Everything else is symbolic.

This is the interface the A* planner will consume: it searches over
states, and each edge of the search graph is a bound action whose cost the
planner can read from the KB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

from kb import Fact, KnowledgeBase
from rules import Bindings, instantiate, match_premises


CostFn = Callable[[KnowledgeBase, Bindings], float]
GuardFn = Callable[[KnowledgeBase, Bindings], bool]


def _f(predicate: str, *args) -> Fact:
    return Fact(predicate, tuple(args))


# --------------------------------------------------------------------------- #
#  Action schema and bound action
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Action:
    """A parameterised action template."""

    name: str
    parameters: Tuple[str, ...] = ()
    preconditions: Tuple[Fact, ...] = ()
    add_effects: Tuple[Fact, ...] = ()
    del_effects: Tuple[Fact, ...] = ()
    cost: Union[float, CostFn] = 1.0
    guard: Optional[GuardFn] = None

    # -- binding --------------------------------------------------------- #

    def bindings(self, kb: KnowledgeBase) -> Iterator[Bindings]:
        """Yield every ground binding of ``parameters`` that satisfies this
        action's preconditions *and* its guard in ``kb``."""
        for b in match_premises(self.preconditions, kb):
            if not all(p in b for p in self.parameters):
                continue  # an unbound parameter is not a ground action
            if self.guard is not None and not self.guard(kb, b):
                continue
            yield b

    def instances(self, kb: KnowledgeBase) -> List["BoundAction"]:
        return [BoundAction(self, b) for b in self.bindings(kb)]

    def bind(self, **kwargs) -> "BoundAction":
        """Construct a specific bound action.  Keys use the ``?var`` names."""
        return BoundAction(self, dict(kwargs))

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.parameters)})"


@dataclass(frozen=True)
class BoundAction:
    """A ground action: a schema plus a variable -> constant binding."""

    action: Action
    bindings: Bindings

    def preconditions_hold(self, kb: KnowledgeBase) -> bool:
        for p in self.action.preconditions:
            if not kb.contains(instantiate(p, self.bindings)):
                return False
        guard = self.action.guard
        return guard is None or guard(kb, self.bindings)

    def cost(self, kb: KnowledgeBase) -> float:
        c = self.action.cost
        return float(c(kb, self.bindings)) if callable(c) else float(c)

    def apply(self, kb: KnowledgeBase) -> None:
        """Apply the action's effects to a KB (delete first, then add)."""
        for f in self.action.del_effects:
            kb.remove(instantiate(f, self.bindings))
        for f in self.action.add_effects:
            kb.add(instantiate(f, self.bindings))

    def __str__(self) -> str:
        args = ", ".join(
            str(self.bindings.get(p, p)) for p in self.action.parameters
        )
        return f"{self.action.name}({args})"


# --------------------------------------------------------------------------- #
#  Numeric helpers for the action definitions
# --------------------------------------------------------------------------- #

def _road_value(kb: KnowledgeBase, a: str, b: str, predicate: str, default):
    """Read the third argument of ``predicate(a, b, v)`` from the KB."""
    facts = kb.query(predicate, a, b)
    return facts[0].args[2] if facts else default


def _move_guard(kb: KnowledgeBase, b: Bindings) -> bool:
    """The agent must carry at least as much fuel as the road takes."""
    agent, src, dst = b.get("?agent"), b.get("?from"), b.get("?to")
    fuel_facts = kb.query("fuel", agent)
    if not fuel_facts:
        return False
    return fuel_facts[0].args[1] >= _road_value(kb, src, dst, "road_time", 1)


def _move_cost(kb: KnowledgeBase, b: Bindings) -> float:
    src, dst = b.get("?from"), b.get("?to")
    return (
        _road_value(kb, src, dst, "road_time", 1)
        + _road_value(kb, src, dst, "road_risk", 0.0)
    )


# --------------------------------------------------------------------------- #
#  The default action schemas
# --------------------------------------------------------------------------- #

MOVE = Action(
    name="Move",
    parameters=("?agent", "?from", "?to"),
    preconditions=(
        _f("at", "?agent", "?from"),
        _f("road_open", "?from", "?to"),
    ),
    add_effects=(_f("at", "?agent", "?to"),),
    del_effects=(_f("at", "?agent", "?from"),),
    guard=_move_guard,
    cost=_move_cost,
)

PICK_UP = Action(
    name="PickUp",
    parameters=("?agent", "?loc", "?resource"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("supply_at", "?loc", "?resource"),
    ),
    add_effects=(_f("has", "?agent", "?resource"),),
    del_effects=(_f("supply_at", "?loc", "?resource"),),
    cost=1.0,
)

DELIVER = Action(
    name="Deliver",
    parameters=("?agent", "?loc", "?resource"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("has", "?agent", "?resource"),
    ),
    add_effects=(_f("supply_at", "?loc", "?resource"),),
    del_effects=(_f("has", "?agent", "?resource"),),
    cost=1.0,
)

TREAT = Action(
    name="Treat",
    # Note: this schema reuses the *derived* predicate ``can_treat`` from
    # rules.py.  It can only be instantiated after forward chaining has run.
    parameters=("?agent", "?person"),
    preconditions=(_f("can_treat", "?agent", "?person"),),
    add_effects=(_f("treated", "?person"),),
    del_effects=(_f("needs_medical", "?person"),),
    cost=2.0,
)

RESCUE = Action(
    name="Rescue",
    parameters=("?agent", "?loc", "?person"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("person_at", "?person", "?loc"),
        _f("needs_evacuation", "?person"),
        _f("has_capability", "?agent", "transport"),
    ),
    add_effects=(_f("rescued", "?person"),),
    del_effects=(
        _f("needs_evacuation", "?person"),
        _f("person_at", "?person", "?loc"),
    ),
    cost=2.0,
)

EVACUATE = Action(
    name="Evacuate",
    parameters=("?agent", "?loc", "?person", "?safe"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("person_at", "?person", "?loc"),
        _f("can_reach", "?agent", "?safe"),   # derived by the reasoner
        _f("has_capability", "?agent", "transport"),
    ),
    add_effects=(_f("person_at", "?person", "?safe"),),
    del_effects=(_f("person_at", "?person", "?loc"),),
    cost=3.0,
)

REPAIR = Action(
    name="Repair",
    parameters=("?agent", "?loc"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("has_capability", "?agent", "repair"),
        _f("has", "?agent", "equipment"),
        _f("damaged", "?loc"),
    ),
    add_effects=(_f("repaired", "?loc"),),
    del_effects=(_f("damaged", "?loc"),),
    cost=3.0,
)

CLEAR_ROAD = Action(
    name="ClearRoad",
    parameters=("?agent", "?a", "?b"),
    # ``can_clear_road`` is derived in rules.py from at(agent, a) and
    # road_blocked(a, b); reusing it keeps the precondition symbolic.
    preconditions=(_f("can_clear_road", "?agent", "?a", "?b"),),
    add_effects=(_f("road_open", "?a", "?b"), _f("road_open", "?b", "?a")),
    del_effects=(
        _f("road_blocked", "?a", "?b"),
        _f("road_blocked", "?b", "?a"),
    ),
    cost=3.0,
)

# ``Refuel`` is deliberately omitted: topping up fuel is a *numeric* effect
# (fuel += n), which does not fit the pure add/delete fact model.  Numeric
# effects are a known limitation of the representation here and would need
# either special-cased effect functions or an explicit numeric-tracking
# layer in the KB.


DEFAULT_ACTIONS: Tuple[Action, ...] = (
    MOVE,
    PICK_UP,
    DELIVER,
    TREAT,
    RESCUE,
    EVACUATE,
    REPAIR,
    CLEAR_ROAD,
)


def applicable_actions(
    kb: KnowledgeBase,
    schemas: Sequence[Action] = DEFAULT_ACTIONS,
) -> List[BoundAction]:
    """Every ground action whose preconditions *and* guard hold in ``kb``."""
    out: List[BoundAction] = []
    for schema in schemas:
        out.extend(schema.instances(kb))
    return out