"""Task-assignment CSP (Step 7).

Assigning pending tasks to agents is a small Constraint Satisfaction
Problem:

    Variables   -- the pending task ids.
    Domains     -- for each task, the agents whose KB contains
                   ``can_assign(agent, task)``.  That single derived
                   predicate already encodes capability, task-capability
                   matching, and reachability, so this layer never
                   re-implements those checks by hand.
    Constraints -- (a) an agent is not assigned more than
                   ``max_tasks_per_agent`` tasks concurrently, and
                   (b) the estimated time for an agent to visit all of
                   its assigned task targets must fit inside the
                   remaining mission budget.

The solver is textbook backtracking with MRV (minimum remaining values)
variable ordering.  A "leave unassigned" fallback value is added to every
domain so a *partial* assignment is returned when no total assignment
exists -- that is what a real coordinator wants when some tasks are simply
infeasible right now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from enums import TaskStatus
from models import Agent, Task
from pathfinding import shortest_travel_time
from world import World


@dataclass
class CSPResult:
    """Outcome of the task-assignment CSP."""

    assignment: Dict[str, str] = field(default_factory=dict)   # task -> agent
    unassigned: List[str] = field(default_factory=list)        # task ids
    backtracks: int = 0

    @property
    def complete(self) -> bool:
        return not self.unassigned


# --------------------------------------------------------------------------- #
#  Solver
# --------------------------------------------------------------------------- #

def solve_task_assignment(
    world: World,
    agents: Dict[str, Agent],
    *,
    max_tasks_per_agent: int = 2,
    enforce_deadline: bool = True,
    score: Optional[Callable[[Dict[str, str], List[str]], float]] = None,
) -> CSPResult:
    """Backtracking search for a valid task -> agent assignment.

    Callers should have already run ``forward_chain`` on each agent's KB
    (see ``main.py`` for the typical pipeline), so that ``can_assign`` is
    present for every feasible (agent, task) pair.
    """
    tasks: List[Task] = [
        t for t in world.tasks.values() if t.status is TaskStatus.PENDING
    ]
    task_by_id: Dict[str, Task] = {t.id: t for t in tasks}
    variables: List[str] = [t.id for t in tasks]

    # -- domains -------------------------------------------------------- #
    domains: Dict[str, List[str]] = {}
    for tid in variables:
        domains[tid] = [
            agent.id
            for agent in agents.values()
            if agent.knowledge.contains(("can_assign", (agent.id, tid)))
        ]

    # -- search state --------------------------------------------------- #
    assignment: Dict[str, str] = {}
    unassigned: Set[str] = set()
    result = CSPResult()
    best_score = float("-inf")

    def record_solution() -> None:
        nonlocal best_score
        candidate = CSPResult(
            assignment=dict(assignment),
            unassigned=sorted(unassigned),
            backtracks=result.backtracks,
        )
        candidate_score = (
            score(candidate.assignment, candidate.unassigned)
            if score is not None
            else 0.0
        )
        if score is None or candidate_score > best_score:
            best_score = candidate_score
            result.assignment = candidate.assignment
            result.unassigned = candidate.unassigned

    def consistent(tid: str, agent_id: str) -> bool:
        # (a) per-agent load cap
        load = sum(1 for a in assignment.values() if a == agent_id)
        if load >= max_tasks_per_agent:
            return False
        # (b) rough deadline feasibility
        if enforce_deadline and not _deadline_ok(
            assignment, tid, agent_id, task_by_id, agents, world
        ):
            return False
        return True

    def select_unassigned() -> Optional[str]:
        """MRV: the unassigned variable with the smallest remaining domain."""
        best: Optional[str] = None
        best_size = float("inf")
        for tid in variables:
            if tid in assignment or tid in unassigned:
                continue
            size = len(domains[tid])
            if size < best_size:
                best, best_size = tid, size
        return best

    def backtrack() -> bool:
        tid = select_unassigned()
        if tid is None:
            record_solution()
            return score is None
        # Try each candidate agent, then the "leave unassigned" fallback.
        for agent_id in list(domains[tid]) + [None]:
            if agent_id is not None and not consistent(tid, agent_id):
                continue
            if agent_id is None:
                unassigned.add(tid)
            else:
                assignment[tid] = agent_id
            if backtrack() and score is None:
                return True
            if agent_id is None:
                unassigned.discard(tid)
            else:
                del assignment[tid]
            result.backtracks += 1
        return False

    backtrack()
    return result


# --------------------------------------------------------------------------- #
#  Deadline constraint
# --------------------------------------------------------------------------- #

def _deadline_ok(
    partial: Dict[str, str],
    tid: str,
    agent_id: str,
    task_by_id: Dict[str, Task],
    agents: Dict[str, Agent],
    world: World,
) -> bool:
    """Rough check that an agent can finish its work before the deadline.

    Starting from the agent's current location, walk the assigned targets
    in the order they were added, accumulating shortest open-road travel
    cost plus a fixed per-action overhead.  If that already exceeds the
    remaining mission time, the candidate assignment is rejected.  The
    estimate deliberately ignores other agents (they act in parallel) and
    resource contention, so it's a lower bound on the agent's *own* time.
    """
    remaining = world.deadline - world.time
    if remaining <= 0:
        return False

    agent = agents[agent_id]
    targets = [task_by_id[t].target for t, a in partial.items() if a == agent_id]
    targets.append(task_by_id[tid].target)

    current = agent.location
    total = 0.0
    for target in targets:
        d = shortest_travel_time(world, current, target)
        if d == float("inf"):
            return False
        total += d + 1.0     # +1 tick for the on-site action itself
        current = target
        if total > remaining:
            return False
    return True