"""A* state-space planner.

The planner searches over *possible states of the world*, not just roads.
A state is a frozen snapshot of an agent's **base** facts; each edge of the
search graph is a ``BoundAction`` applied to that state; the edge's cost is
the action's cost, read from the KB.

Derived facts (``can_reach``, ``can_treat``, ``can_clear_road``, ...) are
*not* stored in the state, because they are deterministic functions of the
base facts.  Whenever the planner needs to check a precondition or evaluate
a goal, it materialises a fresh KB from the state and re-runs forward
chaining.  This keeps states canonical -- so A*'s "have we been here?"
test is sound -- and never leaves stale derivations behind after an action
changes the world.

The public entry point is :func:`plan`.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from actions import Action, BoundAction, DEFAULT_ACTIONS, applicable_actions
from kb import Fact, KnowledgeBase
from reasoner import forward_chain
from rules import DEFAULT_RULES, Rule


# --------------------------------------------------------------------------- #
#  Types
# --------------------------------------------------------------------------- #

State = FrozenSet[Fact]
NumericState = FrozenSet[Fact]
GoalTest = Callable[[KnowledgeBase], bool]
Heuristic = Callable[[KnowledgeBase], float]

# These facts are required by numeric guards/effects, but they are not
# symbolic propositions that define the world state for action matching.
NUMERIC_PREDICATES = frozenset({
    "capacity",
    "delivered_at",
    "elapsed",
    "fuel",
    "inventory",
    "resource_at",
})
NUMERIC_KEY_PREDICATES = frozenset(
    f"{predicate}_band" for predicate in NUMERIC_PREDICATES
)


# --------------------------------------------------------------------------- #
#  State canonicalisation
# --------------------------------------------------------------------------- #

def _derived_predicates(rules: Sequence[Rule]) -> FrozenSet[str]:
    """Predicates the reasoner can *invent* -- these never belong in a state."""
    return frozenset(r.conclusion.predicate for r in rules)


def _numeric_band(predicate: str, value: int) -> int:
    """Keep goal-relevant small quantities exact and bucket larger values."""
    if predicate in {"inventory", "resource_at", "delivered_at"}:
        return min(value, 5)
    if predicate == "fuel":
        return min(value // 5, 4)
    if predicate == "elapsed":
        return min(value // 5, 20)
    return value


def _numeric_key_facts(kb: KnowledgeBase) -> FrozenSet[Fact]:
    return frozenset(
        Fact(
            f"{fact.predicate}_band",
            fact.args[:-1] + (_numeric_band(fact.predicate, int(fact.args[-1])),),
        )
        for fact in kb.facts()
        if fact.predicate in NUMERIC_PREDICATES
    )


def _base_state(kb: KnowledgeBase, derived: FrozenSet[str]) -> State:
    return frozenset(
        f for f in kb.facts()
        if f.predicate not in derived and f.predicate not in NUMERIC_PREDICATES
    ) | _numeric_key_facts(kb)


def _numeric_state(kb: KnowledgeBase) -> NumericState:
    return frozenset(
        f for f in kb.facts() if f.predicate in NUMERIC_PREDICATES
    )


def _materialise(
    state: State,
    rules: Sequence[Rule],
    numeric: NumericState = frozenset(),
    derived_cache: Optional[Dict[State, FrozenSet[Fact]]] = None,
) -> KnowledgeBase:
    """Rebuild a full KB (base + derived) from a base-fact state."""
    cache_key = frozenset(
        f for f in state if f.predicate not in NUMERIC_KEY_PREDICATES
    )
    if derived_cache is not None and cache_key in derived_cache:
        derived = derived_cache[cache_key]
    else:
        derived_kb = KnowledgeBase(
            f for f in state if f.predicate not in NUMERIC_KEY_PREDICATES
        )
        forward_chain(derived_kb, rules)
        derived = frozenset(f for f in derived_kb.facts() if f not in state)
        if derived_cache is not None:
            derived_cache[cache_key] = derived
    symbolic_state = (
        f for f in state if f.predicate not in NUMERIC_KEY_PREDICATES
    )
    kb = KnowledgeBase((*symbolic_state, *numeric, *derived))
    return kb


def _apply_to_state(
    state: State,
    numeric: NumericState,
    ba: BoundAction,
) -> Tuple[State, NumericState]:
    symbolic_state = (
        f for f in state if f.predicate not in NUMERIC_KEY_PREDICATES
    )
    kb = KnowledgeBase((*symbolic_state, *numeric))
    ba.apply(kb)
    next_symbolic = frozenset(
        f for f in kb.facts() if f.predicate not in NUMERIC_PREDICATES
    )
    return (
        next_symbolic | _numeric_key_facts(kb),
        _numeric_state(kb),
    )


# --------------------------------------------------------------------------- #
#  A* search
# --------------------------------------------------------------------------- #

def plan(
    kb: KnowledgeBase,
    goal: GoalTest,
    heuristic: Optional[Heuristic] = None,
    schemas: Sequence[Action] = DEFAULT_ACTIONS,
    rules: Sequence[Rule] = DEFAULT_RULES,
    max_expansions: int = 20_000,
) -> Optional[List[BoundAction]]:
    """Find a lowest-cost sequence of actions that satisfies ``goal``.

    ``kb``        -- the agent's current knowledge base.
    ``goal``      -- a predicate evaluated against a *materialised* KB, so
                     it may test derived facts (``can_reach``, ``treated``,
                     ...) as well as base ones.
    ``heuristic`` -- optional lower-bound estimator of remaining cost.
                     Defaults to the trivial 0 (Dijkstra behaviour).
    ``schemas``   -- the action templates the planner may use.
    ``rules``     -- the rule set used to derive implicit facts.

    Returns the plan as a list of ``BoundAction``, or ``None`` if no plan
    is found within ``max_expansions``.
    """
    derived = _derived_predicates(rules)
    start = _base_state(kb, derived)
    start_numeric = _numeric_state(kb)
    derived_cache: Dict[State, FrozenSet[Fact]] = {}

    start_kb = _materialise(start, rules, start_numeric, derived_cache)
    if goal(start_kb):
        return []

    h_start = heuristic(start_kb) if heuristic is not None else 0.0

    counter = itertools.count()
    frontier: List[
        Tuple[float, float, int, State, NumericState, Tuple[BoundAction, ...]]
    ] = [
        (h_start, 0.0, next(counter), start, start_numeric, ())
    ]
    best_g: Dict[State, float] = {start: 0.0}
    closed: set[State] = set()
    expansions = 0

    while frontier:
        f, g, _, state, numeric, path = heapq.heappop(frontier)

        if g > best_g.get(state, float("inf")):
            continue
        if state in closed:
            continue
        if expansions >= max_expansions:
            return None
        closed.add(state)
        expansions += 1

        state_kb = _materialise(state, rules, numeric, derived_cache)

        if goal(state_kb):
            return list(path)

        for ba in applicable_actions(state_kb, schemas):
            edge = ba.cost(state_kb)
            ng = g + edge

            next_state, next_numeric = _apply_to_state(state, numeric, ba)

            if ng >= best_g.get(next_state, float("inf")):
                continue

            best_g[next_state] = ng

            next_kb = _materialise(
                next_state, rules, next_numeric, derived_cache
            )
            h = heuristic(next_kb) if heuristic else 0.0

            heapq.heappush(
                frontier,
                (
                    ng + h,
                    ng,
                    next(counter),
                    next_state,
                    next_numeric,
                    path + (ba,),
                )
            )

    return None


#Have to change later for further expanding the scemas.
def plan_cost(plan_steps: Sequence[BoundAction], kb: KnowledgeBase) -> float:
    """Sum of the plan's step costs, evaluated against one KB.

    All of the current schemas have state-independent costs (road_time and
    road_risk only change when the world physically changes), so evaluating
    every step against the starting KB is exact here.
    """
    return sum(ba.cost(kb) for ba in plan_steps)


# --------------------------------------------------------------------------- #
#  Heuristics
# --------------------------------------------------------------------------- #

def location_heuristic(agent_id: str, target_location: str) -> Heuristic:
    """Estimate remaining cost from the agent's location to ``target_location``.

    Uses Dijkstra on the currently open road graph, with travel time + risk
    as edge weights.  If the target is currently unreachable we return 0.0
    rather than infinity: a ``ClearRoad`` action later in the plan could
    reopen the graph, so 0.0 keeps the estimate admissible.
    """
    def h(kb: KnowledgeBase) -> float:
        ats = kb.query("at", agent_id, None)
        if not ats:
            return 0.0
        current = ats[0].args[1]
        dist = _shortest_road_cost(kb, current, target_location)
        return 0.0 if dist == float("inf") else dist
    return h


def delivery_heuristic(
    agent_id: str,
    target_location: str,
    resource: str,
    quantity: int,
) -> Heuristic:
    """Estimate delivery cost through the cheapest available supply route."""
    def h(kb: KnowledgeBase) -> float:
        current_facts = kb.query("at", agent_id, None)
        if not current_facts:
            return 0.0
        current = current_facts[0].args[1]
        held = sum(
            int(f.args[2])
            for f in kb.query("inventory", agent_id, resource, None)
        )
        overhead = float(quantity)
        if held >= quantity:
            return _shortest_road_cost(kb, current, target_location) + overhead

        best = float("inf")
        for supply in kb.query("resource_at", None, resource, None):
            location, available = supply.args[0], int(supply.args[2])
            if held + available < quantity:
                continue
            route = (
                _shortest_road_cost(kb, current, location)
                + _shortest_road_cost(kb, location, target_location)
                + overhead
            )
            best = min(best, route)
        return 0.0 if best == float("inf") else best

    return h


def _shortest_road_cost(kb: KnowledgeBase, start: str, goal: str) -> float:
    """Dijkstra over ``road_open`` facts using travel_time + risk."""
    if start == goal:
        return 0.0
    dist: Dict[str, float] = {start: 0.0}
    pq: List[Tuple[float, str]] = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        if node == goal:
            return d
        for fact in kb.query("road_open", node, None):
            nxt = fact.args[1]
            step = (
                _road_attr(kb, node, nxt, "road_time", 1.0)
                + _road_attr(kb, node, nxt, "road_risk", 0.0)
            )
            nd = d + step
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))
    return float("inf")


def _road_attr(kb: KnowledgeBase, a: str, b: str,
               predicate: str, default: float) -> float:
    facts = kb.query(predicate, a, b, None)
    return float(facts[0].args[2]) if facts else default

