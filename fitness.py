"""Multi-objective fitness extracted from a mission result.

Collapsing a mission into a single scalar loses the trade-offs that
matter most in the decision layer (safety vs. speed, throughput vs.
resource efficiency).  Instead, each mission is scored on four
independent axes and the GA does Pareto selection on the vector.

The four objectives:

* score       -- the existing scalar (completed - failed task value)
* completed   -- number of tasks finished
* speed       -- fraction of the deadline still available at the end
* efficiency  -- net completion ratio, penalising failures

All four are monotone-increasing: bigger is better.  The Pareto
machinery in ``evolution.py`` assumes that direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from simulation import MissionResult
from world import World


@dataclass(frozen=True)
class FitnessVector:
    score: float
    completed: float
    speed: float
    efficiency: float

    def as_tuple(self) -> tuple:
        return (self.score, self.completed, self.speed, self.efficiency)


def _mission_vector(world: World, result: MissionResult) -> FitnessVector:
    completed = float(len(result.completed))
    failed = float(len(result.failed))
    ticks = float(max(1, result.ticks))
    speed = max(0.0, 1.0 - ticks / float(max(1, world.deadline)))
    total = max(1.0, completed + failed + float(len(result.pending)))
    efficiency = (completed - failed) / total
    return FitnessVector(
        score=float(result.score),
        completed=completed,
        speed=speed,
        efficiency=efficiency,
    )


def evaluate_policy_vector(
    policy,
    seeds: Sequence[int],
    base_kwargs: dict,
) -> FitnessVector:
    """Mean fitness vector across a seed panel."""
    from main import MissionConfig, run_mission

    total = [0.0, 0.0, 0.0, 0.0]
    for seed in seeds:
        cfg = MissionConfig(seed=seed, policy=policy, **base_kwargs)
        world, _, result = run_mission(cfg)
        v = _mission_vector(world, result)
        for i, x in enumerate(v.as_tuple()):
            total[i] += x
    n = float(len(seeds))
    return FitnessVector(*(x / n for x in total))


def mean_vectors(vectors: List[FitnessVector]) -> FitnessVector:
    if not vectors:
        return FitnessVector(0.0, 0.0, 0.0, 0.0)
    n = float(len(vectors))
    return FitnessVector(
        sum(v.score for v in vectors) / n,
        sum(v.completed for v in vectors) / n,
        sum(v.speed for v in vectors) / n,
        sum(v.efficiency for v in vectors) / n,
    )


# -- Pareto machinery ------------------------------------------------------ #

def dominates(a: FitnessVector, b: FitnessVector) -> bool:
    """a Pareto-dominates b: at least as good everywhere, strictly better once."""
    at, bt = a.as_tuple(), b.as_tuple()
    return all(x >= y for x, y in zip(at, bt)) and any(
        x > y for x, y in zip(at, bt)
    )


def pareto_fronts(vectors: List[FitnessVector]) -> List[List[int]]:
    """NSGA-II fast non-dominated sort.  Returns a list of index groups."""
    n = len(vectors)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    dominated_count = [0] * n
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(vectors[i], vectors[j]):
                dominated_by[i].append(j)
                dominated_count[j] += 1
            elif dominates(vectors[j], vectors[i]):
                dominated_by[j].append(i)
                dominated_count[i] += 1

    for i, c in enumerate(dominated_count):
        if c == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt: List[int] = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(
    vectors: List[FitnessVector],
    indices: List[int],
) -> List[float]:
    """NSGA-II crowding distance within one front."""
    n = len(indices)
    if n <= 2:
        return [float("inf")] * n
    dist = [0.0] * n
    dims = len(vectors[0].as_tuple())
    for d in range(dims):
        order = sorted(
            range(n), key=lambda i: vectors[indices[i]].as_tuple()[d]
        )
        lo = vectors[indices[order[0]]].as_tuple()[d]
        hi = vectors[indices[order[-1]]].as_tuple()[d]
        span = (hi - lo) or 1.0
        dist[order[0]] = dist[order[-1]] = float("inf")
        for k in range(1, n - 1):
            prev_v = vectors[indices[order[k - 1]]].as_tuple()[d]
            next_v = vectors[indices[order[k + 1]]].as_tuple()[d]
            dist[order[k]] += (next_v - prev_v) / span
    return dist