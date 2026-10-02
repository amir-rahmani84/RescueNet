"""Tunable decision policy -- the chromosome for evolutionary search.

Every weight the decision layer uses to rank goals, score assignments,
resolve negotiation, and shape planner costs lives here.  DEFAULT_POLICY
reproduces the hand-tuned constants, so importing this module is
behaviour-preserving until a caller explicitly plugs in a different
policy.

Two genomes are provided:

* ``Policy``      -- one genome driving all decisions (classic).
* ``TeamPolicy``  -- one Policy per agent plus a shared coordinator
                     policy, enabling per-agent specialisation.

Both implement ``as_vector()`` / ``from_vector()`` so the generic
evolutionary machinery in ``evolution.py`` can treat them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Union


@dataclass(frozen=True)
class Policy:
    """The decision-making genome (one agent)."""

    # Goal priority:
    #   priority = importance^a * urgency^b * success^c / (cost + 1)^d
    w_importance: float = 1.0
    w_urgency: float = 1.0
    w_cost: float = 1.0
    w_success: float = 1.0

    # Mission utility shaping.
    w_unassigned: float = 10.0
    w_cost_scale: float = 1.0

    # Negotiation: bonus for matching the top claim on a task.
    w_cooperation: float = 5.0

    # Concurrency cap.
    max_tasks_per_agent: int = 3

    # Risk appetite in [0, 1]: 0 = travel time only, 1 = full road risk.
    w_risk_aversion: float = 0.5

    # Supply selection: higher = prioritise getting to supply quickly.
    w_supply_proximity: float = 1.0

    # How many top-ranked goals an agent considers before choosing.
    # A large value disables the cap (reproduces original behaviour).
    goal_top_k: int = 100

    # Minimum fuel margin the agent keeps in reserve before moving.
    w_fuel_reserve: float = 0.0

    # -- vector conversions --------------------------------------------- #

    def as_vector(self) -> List[float]:
        return [float(getattr(self, f.name)) for f in fields(self)]

    @classmethod
    def from_vector(cls, vec) -> "Policy":
        names = [f.name for f in fields(cls)]
        if len(vec) != len(names):
            raise ValueError(
                f"Policy.from_vector expected {len(names)} values, got {len(vec)}"
            )
        kwargs = {n: float(x) for n, x in zip(names, vec)}
        # Re-cast the integer-valued fields.
        kwargs["max_tasks_per_agent"] = max(
            1, int(round(kwargs["max_tasks_per_agent"]))
        )
        kwargs["goal_top_k"] = max(1, int(round(kwargs["goal_top_k"])))
        return cls(**kwargs)


#: The hand-tuned baseline.  Any call site that takes ``policy=DEFAULT_POLICY``
#: behaves exactly like the pre-evolution code.
DEFAULT_POLICY = Policy()


@dataclass(frozen=True)
class TeamPolicy:
    """One Policy per agent, plus a shared coordinator policy."""

    rescue: Policy = DEFAULT_POLICY
    logistics: Policy = DEFAULT_POLICY
    engineer: Policy = DEFAULT_POLICY
    shared: Policy = DEFAULT_POLICY

    @classmethod
    def uniform(cls, p: Policy) -> "TeamPolicy":
        return cls(rescue=p, logistics=p, engineer=p, shared=p)

    def for_agent(self, agent_id: str) -> Policy:
        prefix = agent_id.split("-", 1)[0].lower()
        if prefix == "rescue":
            return self.rescue
        if prefix == "logistics":
            return self.logistics
        if prefix == "engineer":
            return self.engineer
        return self.shared

    def as_vector(self) -> List[float]:
        return (
            self.rescue.as_vector()
            + self.logistics.as_vector()
            + self.engineer.as_vector()
            + self.shared.as_vector()
        )

    @classmethod
    def from_vector(cls, vec) -> "TeamPolicy":
        n = len(fields(Policy))
        if len(vec) != 4 * n:
            raise ValueError(
                f"TeamPolicy.from_vector expected {4 * n} values, got {len(vec)}"
            )
        chunks = [vec[i:i + n] for i in range(0, 4 * n, n)]
        return cls(
            rescue=Policy.from_vector(chunks[0]),
            logistics=Policy.from_vector(chunks[1]),
            engineer=Policy.from_vector(chunks[2]),
            shared=Policy.from_vector(chunks[3]),
        )


#: Anything that quacks like a Policy genome.
Genome = Union[Policy, TeamPolicy]