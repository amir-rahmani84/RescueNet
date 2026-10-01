"""Plan invalidation detection and autonomous replanning.

When the world changes -- a road blocks, a task is discovered, fuel runs
low, a deadline shrinks -- an agent must not blindly continue executing a
plan that was built against an outdated picture of the world.  This module
supplies three operations:

* ``detect_invalidation(agent, world)``
      Walk the agent's *remaining* plan against a freshly derived view of
      the world.  Returns ``None`` when every step is still executable, or
      an ``Invalidation`` describing the first failure.

* ``replan(agent, world, hub)``
      Re-run the complete OBSERVE -> REPRESENT -> REASON -> SELECT -> PLAN
      loop for one agent, then broadcast the change to every peer through
      the communication hub.

* ``replan_all(world, hub)``
      Replan every agent whose current plan is invalid.

Nothing here invents a new planner, a new reasoner, or a new decision
procedure.  It reuses ``plan`` (A*), ``forward_chain``, ``perceive_agent``,
and ``select_goal`` -- the components built in earlier steps -- and only
adds the orchestration that was missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from actions import BoundAction, DELIVER, MOVE, PICK_UP
from communication import CommunicationHub, MessageType
from decision import select_goal
from enums import TaskType
from kb import KnowledgeBase
from models import Agent
from perception import perceive_agent
from planner import delivery_heuristic, location_heuristic, plan
from reasoner import forward_chain
from rules import instantiate
from world import World


# --------------------------------------------------------------------------- #
#  Invalidation record
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Invalidation:
    """Why a plan step is no longer executable against the current world."""

    index: int
    action: BoundAction
    reason: str

    def __str__(self) -> str:
        return f"step {self.index} ({self.action}): {self.reason}"


# --------------------------------------------------------------------------- #
#  Goal test builder (task -> A* goal predicate)
# --------------------------------------------------------------------------- #

def make_goal_test(
    world: World, agent_id: str, task
) -> Callable[[KnowledgeBase], bool]:
    """Build a goal predicate whose satisfaction means ``task`` is achieved.

    The predicate is evaluated against a *materialised* KB (base + derived),
    so it may test either base effects (``supply_at``, ``repaired``, ...) or
    derived ones (``treated``, ``rescued``).

    Falls back to "just reach the target location" for any task type the
    action schemas cannot yet fully realise.
    """
    t = task.type
    target = task.target
    resource = task.resource

    if t is TaskType.REPAIR:
        return lambda kb: kb.contains(("repaired", (target,)))

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

    # Fallback: at least be at the target.
    return lambda kb: bool(kb.query("at", agent_id, target))


# --------------------------------------------------------------------------- #
#  Invalidation detection
# --------------------------------------------------------------------------- #

def _explain_missing(step: BoundAction, kb: KnowledgeBase) -> str:
    """Name the first precondition of ``step`` that does not hold in ``kb``."""
    for pattern in step.action.preconditions:
        ground = instantiate(pattern, step.bindings)
        if not kb.contains(ground):
            return f"missing precondition {ground}"
    guard = step.action.guard
    if guard is not None and not guard(kb, step.bindings):
        return "numeric guard no longer holds"
    return "preconditions no longer hold"


def _fresh_kb(agent: Agent, world: World) -> KnowledgeBase:
    """Refresh the agent's KB from the current world, then forward-chain."""
    perceive_agent(agent, world)
    forward_chain(agent.knowledge)
    return agent.knowledge


def _simulated_kb(agent: Agent, world: World) -> KnowledgeBase:
    """A *copy* of the agent's fresh KB, safe to mutate during validation."""
    perceive_agent(agent, world)
    forward_chain(agent.knowledge)
    copy = KnowledgeBase(agent.knowledge.facts())
    forward_chain(copy)                 # no-op, but keeps invariants clear
    return copy


def detect_invalidation(agent: Agent, world: World) -> Optional[Invalidation]:
    """Walk the agent's plan against a fresh view of the world.

    Returns ``None`` when the plan (from ``agent.plan_index`` onward) is
    still executable, or the first ``Invalidation`` encountered.
    """
    if not agent.plan or agent.plan_index >= len(agent.plan):
        return None

    kb = _simulated_kb(agent, world)

    for i in range(agent.plan_index, len(agent.plan)):
        step = agent.plan[i]
        if not step.preconditions_hold(kb):
            return Invalidation(i, step, _explain_missing(step, kb))
        step.apply(kb)
        forward_chain(kb)               # derived predicates matter here

    return None


# --------------------------------------------------------------------------- #
#  Replanning
# --------------------------------------------------------------------------- #

def _fmt_plan(actions) -> str:
    return " -> ".join(str(a) for a in actions) if actions else "(none)"


def _announce_replan(
    hub: CommunicationHub,
    world: World,
    agent: Agent,
    reason: str,
    old_goal: Optional[str],
    old_plan_repr: str,
) -> None:
    """Broadcast a WARNING describing the replan to every peer."""
    new_plan_repr = _fmt_plan(agent.plan)
    hub.broadcast(
        world,
        agent.id,
        MessageType.WARNING,
        f"replanned ({reason}): {old_plan_repr}  =>  {new_plan_repr}",
        payload={
            "reason": reason,
            "old_goal": old_goal,
            "new_goal": agent.goal,
            "old_plan": old_plan_repr,
            "new_plan": new_plan_repr,
        },
    )


def replan(
    agent: Agent,
    world: World,
    hub: Optional[CommunicationHub] = None,
    *,
    reason: str = "world changed",
) -> List[BoundAction]:
    """Run the full OBSERVE -> REASON -> SELECT -> PLAN cycle for one agent.

    Updates ``agent.plan`` / ``agent.goal`` / ``agent.plan_created_at`` in
    place, and (when ``hub`` is supplied) broadcasts a WARNING describing
    the change.  Returns the new plan as a list of ``BoundAction``.
    """
    old_goal = agent.goal
    old_plan_repr = _fmt_plan(agent.plan)

    # OBSERVE + REASON
    _fresh_kb(agent, world)

    # SELECT GOAL
    selected = select_goal(world, agent)

    # PLAN
    if selected is None:
        agent.plan = []
        agent.plan_reason = reason
    else:
        task = world.tasks[selected.task_id]
        goal_test = make_goal_test(world, agent.id, task)
        if (
            task.type in (TaskType.DELIVER, TaskType.TRANSPORT)
            and task.resource is not None
        ):
            heuristic = delivery_heuristic(
                agent.id,
                task.target,
                task.resource.value,
                task.quantity,
            )
        else:
            heuristic = location_heuristic(agent.id, task.target)
        schemas = (
            (MOVE, PICK_UP, DELIVER)
            if task.type in (TaskType.DELIVER, TaskType.TRANSPORT)
            else None
        )
        plan_kwargs = {"heuristic": heuristic}
        if schemas is not None:
            plan_kwargs["schemas"] = schemas
        new_plan = plan(agent.knowledge, goal_test, **plan_kwargs)
        agent.plan = list(new_plan) if new_plan is not None else []
        agent.plan_reason = reason

    agent.plan_index = 0
    agent.plan_created_at = world.time

    # COMMUNICATE
    if hub is not None and old_plan_repr != _fmt_plan(agent.plan):
        _announce_replan(hub, world, agent, reason, old_goal, old_plan_repr)

    return list(agent.plan)


def replan_all(
    world: World,
    hub: Optional[CommunicationHub] = None,
    *,
    reason: str = "world changed",
) -> Dict[str, List[BoundAction]]:
    """Replan every agent whose current plan has been invalidated.

    Returns ``{agent_id: new_plan}`` for the agents that actually replanned.
    """
    outcomes: Dict[str, List[BoundAction]] = {}
    for agent in world.agents.values():
        inval = detect_invalidation(agent, world)
        if inval is None:
            continue
        outcomes[agent.id] = replan(
            agent, world, hub, reason=str(inval)
        )
    return outcomes