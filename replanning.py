"""Plan invalidation detection and autonomous replanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from actions import BoundAction, actions_for_policy
from communication import CommunicationHub, MessageType
from decision import select_goal
from enums import TaskStatus, TaskType
from kb import KnowledgeBase
from models import Agent
from perception import perceive_agent
from planner import delivery_heuristic, location_heuristic, plan
from policy import Policy, DEFAULT_POLICY
from reasoner import forward_chain
from rules import instantiate
from world import World


@dataclass(frozen=True)
class Invalidation:
    index: int
    action: BoundAction
    reason: str

    def __str__(self) -> str:
        return f"step {self.index} ({self.action}): {self.reason}"


def make_goal_test(
    world: World, agent_id: str, task
) -> Callable[[KnowledgeBase], bool]:
    t = task.type
    target = task.target
    resource = task.resource

    if t is TaskType.REPAIR:
        return lambda kb: kb.contains(("repaired", (target,)))

    if t is TaskType.CLEAR_ROAD and task.secondary_target is not None:
        other = task.secondary_target
        return lambda kb: kb.contains(("road_open", (target, other)))

    if t in (TaskType.DELIVER, TaskType.TRANSPORT) and resource is not None:
        value = resource.value
        required = task.quantity
        return lambda kb: any(
            fact.args[2] >= required
            for fact in kb.query("delivered_at", target, value, None)
        )

    if t is TaskType.RESCUE:
        civilians = tuple(
            c.id for c in world.civilians.values() if c.location == target
        )
        if civilians:
            return lambda kb: any(
                kb.contains(("rescued", (cid,))) for cid in civilians
            )

    if t is TaskType.EVACUATE:
        civilians = tuple(
            c.id for c in world.civilians.values() if c.location == target
        )
        if civilians:
            return lambda kb: any(
                kb.contains(("evacuated", (cid,))) for cid in civilians
            )

    return lambda kb: bool(kb.query("at", agent_id, target))


def _explain_missing(step: BoundAction, kb: KnowledgeBase) -> str:
    for pattern in step.action.preconditions:
        ground = instantiate(pattern, step.bindings)
        if not kb.contains(ground):
            return f"missing precondition {ground}"
    guard = step.action.guard
    if guard is not None and not guard(kb, step.bindings):
        return "numeric guard no longer holds"
    return "preconditions no longer hold"


def detect_invalidation(
    agent: Agent, world: World
) -> Optional[Invalidation]:
    if not agent.plan:
        return None
    for index in range(agent.plan_index, len(agent.plan)):
        step = agent.plan[index]
        if step.preconditions_hold(agent.knowledge):
            continue
        return Invalidation(index, step, _explain_missing(step, agent.knowledge))
    return None


def _fmt_plan(actions) -> str:
    return " -> ".join(str(a) for a in actions) if actions else "(none)"


def _announce_replan(
    hub: CommunicationHub, world: World, agent: Agent, reason: str,
    old_goal: Optional[str], old_plan_repr: str,
) -> None:
    new_plan_repr = _fmt_plan(agent.plan)
    hub.broadcast(
        world, agent.id, MessageType.WARNING,
        f"replanned ({reason}): {old_plan_repr}  =>  {new_plan_repr}",
        payload={
            "reason": reason,
            "old_goal": old_goal,
            "new_goal": agent.goal,
            "old_plan": old_plan_repr,
            "new_plan": new_plan_repr,
        },
    )


def _build_plan_for_task(
    agent: Agent,
    world: World,
    task,
    policy: Policy = DEFAULT_POLICY,
) -> Optional[List[BoundAction]]:
    goal_test = make_goal_test(world, agent.id, task)
    action_set = actions_for_policy(policy)

    if (
        task.type in (TaskType.DELIVER, TaskType.TRANSPORT)
        and task.resource is not None
    ):
        heuristic = delivery_heuristic(
            agent.id, task.target, task.resource.value, task.quantity,
            policy=policy,
        )
        schemas = tuple(
            a for a in action_set
            if a.name in {"Move", "Refuel", "PickUp", "Deliver"}
        )
    else:
        heuristic = location_heuristic(agent.id, task.target)
        schemas = action_set

    new_plan = plan(
        agent.knowledge, goal_test,
        heuristic=heuristic,
        schemas=schemas,
    )
    return list(new_plan) if new_plan is not None else None


def replan(
    agent: Agent, world: World, hub: Optional[CommunicationHub] = None,
    *, reason: str = "world changed",
    preferred_task_id: Optional[str] = None,
    policy: Policy = DEFAULT_POLICY,
) -> List[BoundAction]:
    old_goal = agent.goal
    old_plan_repr = _fmt_plan(agent.plan)

    perceive_agent(agent, world)
    forward_chain(agent.knowledge)

    task = None
    if preferred_task_id is not None and preferred_task_id in world.tasks:
        candidate = world.tasks[preferred_task_id]
        if (
            candidate.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
            and candidate.assigned_to in (None, agent.id)
        ):
            task = candidate
            agent.goal = candidate.id

    if task is None:
        selected = select_goal(world, agent, policy=policy)
        if selected is not None:
            task = world.tasks[selected.task_id]

    if task is None:
        agent.goal = None
        agent.plan = []
        agent.plan_reason = reason
    else:
        new_plan = _build_plan_for_task(agent, world, task, policy=policy)
        agent.plan = new_plan if new_plan is not None else []
        agent.plan_reason = reason

    agent.plan_index = 0
    agent.plan_created_at = world.time

    if hub is not None and old_plan_repr != _fmt_plan(agent.plan):
        _announce_replan(hub, world, agent, reason, old_goal, old_plan_repr)

    return list(agent.plan)


def replan_all(
    world: World, hub: Optional[CommunicationHub] = None,
    *, reason: str = "world changed",
    policy: Policy = DEFAULT_POLICY,
) -> Dict[str, List[BoundAction]]:
    outcomes: Dict[str, List[BoundAction]] = {}
    for agent in world.agents.values():
        perceive_agent(agent, world)
        forward_chain(agent.knowledge)
        inval = detect_invalidation(agent, world)
        if inval is None:
            continue
        outcomes[agent.id] = replan(
            agent, world, hub, reason=str(inval), policy=policy,
        )
    return outcomes