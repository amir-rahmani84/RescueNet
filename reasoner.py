"""Forward-chaining rule engine.

Two phases:

1. **Reachability.** `can_reach(agent, loc)` is derived by BFS over the
   `road_open` facts, once per agent. This replaces the recursive
   `reach_step` rule, which was the single most expensive part of the
   fixpoint loop.

2. **Rule fixpoint.** Remaining rules are applied until no new facts
   appear, using the standard semi-naive evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from kb import Fact, KnowledgeBase
from rules import DEFAULT_RULES, Rule, instantiate


@dataclass(frozen=True)
class Derivation:
    fact: Fact
    rule: str
    premises: Tuple[Fact, ...]

    def __str__(self) -> str:
        prem = ", ".join(str(p) for p in self.premises)
        return f"{self.fact}   <- [{self.rule}]   {prem}"


# --------------------------------------------------------------------------- #
#  Direct reachability
# --------------------------------------------------------------------------- #

def _derive_reachability(kb: KnowledgeBase) -> List[Derivation]:
    """One BFS per agent over `road_open` facts."""
    agent_locs: Dict[str, str] = {}
    for fact in kb.query("at", None, None):
        if len(fact.args) == 2:
            agent_locs[fact.args[0]] = fact.args[1]
    if not agent_locs:
        return []

    adj: Dict[str, List[str]] = {}
    for fact in kb.query("road_open", None, None):
        if len(fact.args) != 2:
            continue
        a, b = fact.args
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    derivations: List[Derivation] = []
    for agent, start in agent_locs.items():
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        start_fact = Fact("at", (agent, start))
        for loc in seen:
            fact = Fact("can_reach", (agent, loc))
            if not kb.contains(fact):
                derivations.append(Derivation(
                    fact=fact,
                    rule="reach_bfs",
                    premises=(start_fact,),
                ))
    return derivations


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def forward_chain(
    kb: KnowledgeBase,
    rules: Sequence[Rule] = DEFAULT_RULES,
    max_iterations: int = 100,
) -> List[Derivation]:
    """Apply reachability BFS + rules to a fixpoint. Mutates `kb` in place."""
    derivations: List[Derivation] = []
    known = set(kb.facts())

    # Phase 1 -- reachability.
    fresh_reach: List[Fact] = []
    for d in _derive_reachability(kb):
        if d.fact in known:
            continue
        known.add(d.fact)
        fresh_reach.append(d.fact)
        derivations.append(d)
    kb.add_many(fresh_reach)

    # Phase 2 -- rule fixpoint.
    for _ in range(max_iterations):
        fresh: List[Derivation] = []
        for rule in rules:
            for bindings in rule.fires(kb):
                fact = instantiate(rule.conclusion, bindings)
                if fact in known:
                    continue
                known.add(fact)
                premises = tuple(instantiate(p, bindings) for p in rule.premises)
                fresh.append(Derivation(fact, rule.name, premises))
        if not fresh:
            break
        for d in fresh:
            kb.add(d.fact)
        derivations.extend(fresh)

    return derivations


# --------------------------------------------------------------------------- #
#  Presentation helpers
# --------------------------------------------------------------------------- #

def explain(
    derivations: Sequence[Derivation],
    predicate: Optional[str] = None,
) -> str:
    lines = [
        f"    {d}" for d in derivations
        if predicate is None or d.fact.predicate == predicate
    ]
    return "\n".join(lines) if lines else "    (nothing derived)"