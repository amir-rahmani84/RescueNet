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
EffectFn = Callable[[KnowledgeBase, Bindings], None]


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
    effect: Optional[EffectFn] = None

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


NUMERIC_PRECONDITION_PREDICATES = frozenset({
    "resource_at",
    "inventory",
    "fuel",
    "elapsed",
    "delivered_at",
    "capacity",
})


@dataclass(frozen=True)
class BoundAction:
    """A ground action: a schema plus a variable -> constant binding."""

    action: Action
    bindings: Bindings

    def preconditions_hold(self, kb: KnowledgeBase) -> bool:
        for p in self.action.preconditions:
            ground = instantiate(p, self.bindings)
            if ground.predicate in NUMERIC_PRECONDITION_PREDICATES and ground.args:
                pattern = ground.args[:-1] + (None,)
                if not kb.query(ground.predicate, *pattern):
                    return False
            elif not kb.contains(ground):
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
        if self.action.effect is not None:
            self.action.effect(kb, self.bindings)

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
    facts = kb.query(predicate, a, b, None)
    return facts[0].args[2] if facts else default


def _move_guard(kb: KnowledgeBase, b: Bindings) -> bool:
    """The agent must carry at least as much fuel as the road takes."""
    agent, src, dst = b.get("?agent"), b.get("?from"), b.get("?to")
    fuel_facts = kb.query("fuel", agent, None)
    if not fuel_facts:
        return False
    return fuel_facts[0].args[1] >= _road_value(kb, src, dst, "road_time", 1)


def _move_cost(kb: KnowledgeBase, b: Bindings) -> float:
    src, dst = b.get("?from"), b.get("?to")
    return (
        _road_value(kb, src, dst, "road_time", 1)
        + _road_value(kb, src, dst, "road_risk", 0.0)
    )


def _replace_value(kb: KnowledgeBase, predicate: str, prefix: tuple,
                   value: int) -> None:
    for fact in list(kb.query(predicate, *prefix, None)):
        if fact.args[:len(prefix)] == prefix:
            kb.remove(fact)
    kb.add(Fact(predicate, prefix + (max(0, value),)))


def _inventory_qty(kb: KnowledgeBase, agent: str, resource: str) -> int:
    facts = kb.query("inventory", agent, resource, None)
    return int(facts[0].args[2]) if facts else 0


def _move_effect(kb: KnowledgeBase, b: Bindings) -> None:
    agent, src, dst = b["?agent"], b["?from"], b["?to"]
    fuel = int(kb.query("fuel", agent, None)[0].args[1])
    travel = int(_road_value(kb, src, dst, "road_time", 1))
    elapsed = int(kb.query("elapsed", agent, None)[0].args[1])
    _replace_value(kb, "fuel", (agent,), fuel - travel)
    _replace_value(kb, "elapsed", (agent,), elapsed + travel)


def _pickup_guard(kb: KnowledgeBase, b: Bindings) -> bool:
    agent, loc, resource = b["?agent"], b["?loc"], b["?resource"]
    facts = kb.query("resource_at", loc, resource, None)
    available = int(facts[0].args[2]) if facts else 0
    capacity = int(kb.query("capacity", agent, None)[0].args[1])
    load = sum(int(f.args[2]) for f in kb.query("inventory", agent, None, None))
    return available > 0 and load < capacity


def _pickup_effect(kb: KnowledgeBase, b: Bindings) -> None:
    agent, loc, resource = b["?agent"], b["?loc"], b["?resource"]
    facts = kb.query("resource_at", loc, resource, None)
    available = int(facts[0].args[2]) if facts else 0
    current = _inventory_qty(kb, agent, resource)
    _replace_value(kb, "resource_at", (loc, resource), available - 1)
    _replace_value(kb, "inventory", (agent, resource), current + 1)
    kb.add(_f("has", agent, resource))


def _deliver_guard(kb: KnowledgeBase, b: Bindings) -> bool:
    if _inventory_qty(kb, b["?agent"], b["?resource"]) <= 0:
        return False
    agent, loc = b["?agent"], b["?loc"]
    targets = kb.query("goal_target", agent, None)
    if targets and targets[0].args[1] != loc:
        return False
    return True


def _deliver_effect(kb: KnowledgeBase, b: Bindings) -> None:
    agent, loc, resource = b["?agent"], b["?loc"], b["?resource"]
    current = _inventory_qty(kb, agent, resource)
    delivered = kb.query("delivered_at", loc, resource, None)
    previous = int(delivered[0].args[2]) if delivered else 0
    _replace_value(kb, "inventory", (agent, resource), current - 1)
    _replace_value(kb, "delivered_at", (loc, resource), previous + 1)
    kb.add(_f("supply_at", loc, resource))
    if current <= 1:
        kb.remove(_f("has", agent, resource))


def _treat_effect(kb: KnowledgeBase, b: Bindings) -> None:
    agent, person = b["?agent"], b["?person"]
    current = _inventory_qty(kb, agent, "medical_kit")
    _replace_value(kb, "inventory", (agent, "medical_kit"), current - 1)
    if current <= 1:
        kb.remove(_f("has", agent, "medical_kit"))


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
    effect=_move_effect,
)

PICK_UP = Action(
    name="PickUp",
    parameters=("?agent", "?loc", "?resource"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("resource_at", "?loc", "?resource", "?available"),
    ),
    add_effects=(),
    del_effects=(),
    cost=1.0,
    guard=_pickup_guard,
    effect=_pickup_effect,
)

DELIVER = Action(
    name="Deliver",
    parameters=("?agent", "?loc", "?resource"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("inventory", "?agent", "?resource", "?available"),
    ),
    add_effects=(),
    del_effects=(),
    cost=1.0,
    guard=_deliver_guard,
    effect=_deliver_effect,
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
    effect=_treat_effect,
)

RESCUE = Action(
    name="Rescue",
    parameters=("?agent", "?loc", "?person"),
    preconditions=(
        _f("at", "?agent", "?loc"),
        _f("person_at", "?person", "?loc"),
        _f("needs_medical", "?person"),
        _f("has_capability", "?agent", "medical"),
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
    add_effects=(_f("person_at", "?person", "?safe"), _f("evacuated", "?person")),
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
        head = schema.preconditions[0] if schema.preconditions else None
        if head is not None and not kb.has_predicate(head.predicate):
            continue
        out.extend(schema.instances(kb))
    return out