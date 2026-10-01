"""Forward-chaining rule engine.

``rules.py`` gives us the *representation* of a rule (premises + conclusion,
with ``?variables``).  What is missing is the loop that actually drives
inference:

    repeat:
        for every rule:
            if its premises hold in the KB:
                add the rule's conclusion
    until no new facts appear

This is classical data-driven (forward) reasoning.  With ground facts and
function-free rules the loop is guaranteed to terminate: the set of ground
facts over a finite universe of constants is finite.

The KB is mutated in place.  ``forward_chain`` also returns a list of
``Derivation`` records so the caller can show *why* each new fact was
produced (useful both for the terminal display and for debugging).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from kb import Fact, KnowledgeBase
from rules import DEFAULT_RULES, Rule, instantiate


@dataclass(frozen=True)
class Derivation:
    """A fact that was *derived* (rather than perceived), and its support.

    ``premises`` are the ground versions of the rule's premises under the
    binding that fired it.  This is a cheap, one-level explanation -- we do
    not chase the justification of *those* premises further.
    """

    fact: Fact
    rule: str
    premises: Tuple[Fact, ...]

    def __str__(self) -> str:
        prem = ", ".join(str(p) for p in self.premises)
        return f"{self.fact}   <- [{self.rule}]   {prem}"


def forward_chain(
    kb: KnowledgeBase,
    rules: Sequence[Rule] = DEFAULT_RULES,
    max_iterations: int = 100,
) -> List[Derivation]:
    """Apply ``rules`` to ``kb`` until a fixpoint is reached.

    Each pass evaluates every rule against the KB *as it was at the start of
    the pass*, then adds all freshly derived facts at once.  Rules that need
    a fact derived by another rule therefore fire on the next pass.  This is
    the standard semi-naive evaluation strategy.

    Returns every newly derived fact (in derivation order).  The KB is
    mutated in place.
    """
    derivations: List[Derivation] = []
    known: set = set(kb.facts())

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
#  Presentation helpers (used by the terminal demo, harmless otherwise)
# --------------------------------------------------------------------------- #

def explain(
    derivations: Sequence[Derivation],
    predicate: Optional[str] = None,
) -> str:
    """Human-readable list of derivations, optionally filtered by predicate."""
    lines = [
        f"    {d}" for d in derivations
        if predicate is None or d.fact.predicate == predicate
    ]
    return "\n".join(lines) if lines else "    (nothing derived)"