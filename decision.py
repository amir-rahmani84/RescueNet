"""Mission optimization and utility-based goal selection (Steps 8 and 9).

Every decision rule is parameterised by a ``Policy`` (see ``policy.py``).
The default policy reproduces the original hard-tuned constants, so
existing callers behave identically until they opt into a new genome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from constraints import CSPResult, solve_task_assignment
from enums import TaskStatus, TaskType
from models import Agent, Task
from pathfinding import shortest_travel_time
from policy import Policy, DEFAULT_POLICY
from world import World


TASK_VALUE = {
    TaskType.RESCUE: 100.0,
    TaskType.EVACUATE: 90.0,
    TaskType.REPAIR: 50.0,
    TaskType.CLEAR_ROAD: 60.0,
    TaskType.DELIVER: 30.0,
    TaskType.TRANSPORT: 30.0,
}

ACTION_COST = {
    TaskType.RESCUE: 2.0,
    TaskType.EVACUATE: 3.0,
    TaskType.REPAIR: 3.0,
    TaskType.CLEAR_ROAD: 3.0,
    TaskType.DELIVER: 1.0,
    TaskType.TRANSPORT: 1.0,
}


@dataclass(frozen=True)
class GoalCandidate:
    """A task evaluated from one agent's current perspective."""

    agent_id: str
    task_id: str
    importance: float
    urgency: float
    success_probability: float
    estimated_cost: float
    priority: float


def _task_cost(world: World, agent: Agent, task: Task) -> float:
    travel = shortest_travel_time(world, agent.location, task.target)
    if travel == float("inf"):
        return float("inf")
    return travel + ACTION_COST.get(task.type, 1.0)


def _urgency(world: World, task: Task) -> float:
    deadline = task.deadline if task.deadline is not None else world.deadline
    remaining = max(0, deadline - world.time)
    return 1.0 if remaining == 0 else 1.0 + 1.0 / remaining


def _success_probability(world: World, agent: Agent, task: Task) -> float:
    cost = _task_cost(world, agent, task)
    if cost == float("inf") or cost > agent.fuel + ACTION_COST.get(task.type, 1.0):
        return 0.0
    if task.quantity > agent.capacity:
        return 0.0
    return 1.0


def _priority(
    importance: float,
    urgency: float,
    success: float,
    cost: float,
    policy: Policy,
) -> float:
    """Weighted priority formula.  Reduces to the classic one when the
    policy has all weights == 1.0.

    Python's ``0 ** 0 == 1`` and all four factors are guaranteed positive
    for feasible goals, so this never divides by zero and never raises.
    """
    return (
        (importance ** policy.w_importance)
        * (urgency ** policy.w_urgency)
        * (success ** policy.w_success)
        / ((cost + 1.0) ** policy.w_cost)
    )


def evaluate_goal(
    world: World,
    agent: Agent,
    task: Task,
    policy: Policy = DEFAULT_POLICY,
) -> Optional[GoalCandidate]:
    """Score a task only when the agent's derived KB says it is assignable."""
    if task.status is not TaskStatus.PENDING:
        return None
    if task.deadline is not None and world.time > task.deadline:
        return None
    if not agent.knowledge.contains(("can_assign", (agent.id, task.id))):
        return None

    cost = _task_cost(world, agent, task)
    success = _success_probability(world, agent, task)
    if success == 0.0 or cost == float("inf"):
        return None
    importance = TASK_VALUE.get(task.type, 1.0)
    urgency = _urgency(world, task)
    priority = _priority(importance, urgency, success, cost, policy)
    return GoalCandidate(
        agent.id, task.id, importance, urgency, success, cost, priority
    )


def rank_goals(
    world: World,
    agent: Agent,
    policy: Policy = DEFAULT_POLICY,
) -> List[GoalCandidate]:
    """Return feasible goals in descending utility order."""
    candidates = [
        candidate
        for task in world.tasks.values()
        if (candidate := evaluate_goal(world, agent, task, policy)) is not None
    ]
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.priority, candidate.task_id),
    )


def select_goal(
    world: World,
    agent: Agent,
    policy: Policy = DEFAULT_POLICY,
) -> Optional[GoalCandidate]:
    """Select and store the highest-priority goal for an agent."""
    ranked = rank_goals(world, agent, policy)
    selected = ranked[0] if ranked else None
    agent.goal = selected.task_id if selected else None
    return selected


def mission_utility(
    world: World,
    assignment: Dict[str, str],
    unassigned: List[str],
    policy: Policy = DEFAULT_POLICY,
) -> float:
    """Estimate mission value: reward completed work minus operating cost."""
    utility = -policy.w_unassigned * len(unassigned)
    for task_id, agent_id in assignment.items():
        task = world.tasks[task_id]
        agent = world.agents[agent_id]
        cost = _task_cost(world, agent, task)
        if cost == float("inf"):
            return float("-inf")
        utility += TASK_VALUE.get(task.type, 1.0) - policy.w_cost_scale * cost
    return utility


def optimize_assignment(
    world: World,
    agents: Dict[str, Agent],
    *,
    max_tasks_per_agent: int = 2,
    enforce_deadline: bool = True,
    score: Optional[Callable[[Dict[str, str], List[str]], float]] = None,
    forbidden: Optional[Dict[str, Set[str]]] = None,
    policy: Policy = DEFAULT_POLICY,
) -> CSPResult:
    """Find the feasible task assignment with the highest mission utility.

    ``score`` and ``forbidden`` are passed through to the CSP unchanged.
    When ``score`` is omitted, the default scoring is ``mission_utility``
    parameterised by ``policy``.
    """
    if score is None:
        score = lambda assignment, unassigned: mission_utility(
            world, assignment, unassigned, policy
        )
    return solve_task_assignment(
        world,
        agents,
        max_tasks_per_agent=max_tasks_per_agent,
        enforce_deadline=enforce_deadline,
        score=score,
        forbidden=forbidden,
    )