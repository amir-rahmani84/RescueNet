"""Step 14 -- the central agent loop.

The loop keeps the documented shape (OBSERVE -> ... -> WORLD CHANGES),
but each agent now perceives and forward-chains exactly once per turn.
Invalidation is a next-action check, and execution does not re-perceive.

Performance notes
-----------------
* Each agent tracks which (agent, task) goals it has *failed* to plan.
  Those goals are not retried on subsequent ticks, unless new events
  have fired in the world (which can make a previously impossible task
  possible again).  This removes the pathological "re-plan the same
  unplannable goal every single tick" loop.
* ``plan_max_expansions`` caps how deep A* may search per planning call.
  The planner's own default (20_000) is fine for one-shot demos but is
  far too generous for a tight simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from actions import DELIVER, MOVE, PICK_UP
from communication import (
    CommunicationHub,
    MessageType,
    NegotiationResult,
    claim_aware_score,
    negotiate_assignment,
    reassign_rejected,
    solicit_claims,
)
from decision import TASK_VALUE, optimize_assignment, rank_goals
from enums import TaskStatus
from perception import perceive_agent
from planner import delivery_heuristic, location_heuristic, plan
from reasoner import forward_chain
from replanning import detect_invalidation, make_goal_test
from world import World


@dataclass
class MissionResult:
    iterations: int = 0
    completed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def ticks(self) -> int:
        return self.iterations

    @ticks.setter
    def ticks(self, value: int) -> None:
        self.iterations = value


class Simulation:
    def __init__(
        self, world: World, *,
        hub: Optional[CommunicationHub] = None,
        max_tasks_per_agent: int = 2,
        max_ticks: int = 200,
        plan_max_expansions: int = 3000,
    ) -> None:
        self.world = world
        self.hub = hub or CommunicationHub()
        self.max_tasks_per_agent = max_tasks_per_agent
        self.max_ticks = max_ticks
        self.plan_max_expansions = plan_max_expansions
        self.negotiation: Optional[NegotiationResult] = None

        # (agent_id, task_id) pairs we have already failed to plan for.
        # Retried only when new events have fired (the world changed).
        self._failed_goals: Set[Tuple[str, str]] = set()
        self._last_event_count: int = 0

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def setup(self) -> NegotiationResult:
        for agent in self.world.agents.values():
            perceive_agent(agent, self.world)
            forward_chain(agent.knowledge)

        claims = solicit_claims(self.world, self.hub)
        proposal = optimize_assignment(
            self.world, self.world.agents,
            max_tasks_per_agent=self.max_tasks_per_agent,
            score=claim_aware_score(self.world, claims),
        )
        confirmed = negotiate_assignment(
            self.world, self.hub, proposal, claims=claims
        )
        confirmed = reassign_rejected(
            self.world, self.hub, confirmed,
            max_tasks_per_agent=self.max_tasks_per_agent,
            claims=claims,
        )

        for task_id, agent_id in confirmed.confirmed.items():
            self.world.agents[agent_id].goal = task_id

        self.negotiation = confirmed
        return confirmed

    # ------------------------------------------------------------------ #
    #  Tick
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        # If new events have fired since the last tick, the world may have
        # become more (or less) permissive.  Clear the failure cache so
        # agents get a fresh chance to plan for previously-impossible goals.
        if len(self.world.events) != self._last_event_count:
            self._failed_goals.clear()
            self._last_event_count = len(self.world.events)

        start_time = self.world.time
        for agent in list(self.world.agents.values()):
            self._agent_turn(agent)
        if self.world.time == start_time:
            self.world.advance_time(1)

    # ------------------------------------------------------------------ #
    #  One agent's turn
    # ------------------------------------------------------------------ #

    def _agent_turn(self, agent) -> None:
        # ---- OBSERVE + REPRESENT + REASON (once per turn) ------------ #
        perceive_agent(agent, self.world)
        forward_chain(agent.knowledge)

        # ---- SELECT GOAL --------------------------------------------- #
        if not self._goal_still_valid(agent):
            chosen = self._best_goal(agent)
            if chosen is None:
                agent.goal = None
                agent.plan = []
                agent.plan_index = 0
                return
            agent.goal = chosen

        # ---- PLAN ---------------------------------------------------- #
        just_planned = False
        if not agent.plan or agent.plan_index >= len(agent.plan):
            task = self.world.tasks[agent.goal]
            if self._plan_for(agent, task):
                just_planned = True
                self.hub.broadcast(
                    self.world, agent.id, MessageType.INFORM,
                    f"plan for {task.id}: "
                    f"{' -> '.join(str(a) for a in agent.plan) or '(none)'}",
                    task_id=task.id,
                    payload={"reason": "planning"},
                )
            else:
                self._failed_goals.add((agent.id, task.id))
                alt = self._best_goal(agent)
                if alt is None or alt == agent.goal:
                    agent.plan = []
                    agent.plan_index = 0
                    return
                agent.goal = alt
                if not self._plan_for(agent, self.world.tasks[alt]):
                    self._failed_goals.add((agent.id, alt))
                    agent.plan = []
                    agent.plan_index = 0
                    return
                just_planned = True

        # ---- CHECK CONSTRAINTS (next action only) -------------------- #
        if not just_planned and agent.plan:
            inval = detect_invalidation(agent, self.world)
            if inval is not None:
                task = self.world.tasks.get(agent.goal)
                if task is not None and self._plan_for(agent, task):
                    self.hub.broadcast(
                        self.world, agent.id, MessageType.WARNING,
                        f"replanned ({inval})",
                        task_id=task.id,
                        payload={"reason": str(inval)},
                    )
                else:
                    if task is not None:
                        self._failed_goals.add((agent.id, task.id))
                    agent.plan = []
                    agent.plan_index = 0
                    return

        # ---- ACT ---------------------------------------------------- #
        if not agent.plan or agent.plan_index >= len(agent.plan):
            return

        action = agent.plan[agent.plan_index]
        try:
            self.world.apply_action(agent.id, action)
        except ValueError as exc:
            if agent.goal is not None:
                self._failed_goals.add((agent.id, agent.goal))
            agent.plan = []
            agent.plan_index = 0
            return

        agent.plan_index += 1
        self._announce_task_completion(agent)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _goal_still_valid(self, agent) -> bool:
        if agent.goal is None:
            return False
        task = self.world.tasks.get(agent.goal)
        if task is None:
            return False
        if task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
            return False
        if task.assigned_to is not None and task.assigned_to != agent.id:
            return False
        return True

    def _best_goal(self, agent) -> Optional[str]:
        """Highest-priority goal this agent hasn't already failed to plan."""
        ranked = rank_goals(self.world, agent)
        for candidate in ranked:
            if (agent.id, candidate.task_id) not in self._failed_goals:
                return candidate.task_id
        return None

    def _plan_for(self, agent, task) -> bool:
        goal_test = make_goal_test(self.world, agent.id, task)
        if (
            task.type.value in ("deliver", "transport")
            and task.resource is not None
        ):
            heuristic = delivery_heuristic(
                agent.id, task.target, task.resource.value, task.quantity
            )
            schemas = (MOVE, PICK_UP, DELIVER)
        else:
            heuristic = location_heuristic(agent.id, task.target)
            schemas = None

        kwargs = {
            "heuristic": heuristic,
            "max_expansions": self.plan_max_expansions,
        }
        if schemas is not None:
            kwargs["schemas"] = schemas

        new_plan = plan(agent.knowledge, goal_test, **kwargs)
        if new_plan is None:
            agent.plan = []
            agent.plan_index = 0
            return False
        agent.plan = list(new_plan)
        agent.plan_index = 0
        agent.plan_created_at = self.world.time
        return True

    def _announce_task_completion(self, agent) -> None:
        if agent.goal is None:
            return
        task = self.world.tasks.get(agent.goal)
        if task is None or task.status is not TaskStatus.COMPLETED:
            return
        self.hub.broadcast(
            self.world, agent.id, MessageType.INFORM,
            f"completed {task.id} ({task.description})",
            task_id=task.id,
            payload={"status": "completed"},
        )
        agent.goal = None
        agent.plan = []
        agent.plan_index = 0

    # ------------------------------------------------------------------ #
    #  Mission loop
    # ------------------------------------------------------------------ #

    def run(self) -> MissionResult:
        if self.negotiation is None:
            self.setup()
        iterations = 0
        while self._mission_active() and iterations < self.max_ticks:
            self.tick()
            iterations += 1
        # A stopped mission cannot leave work actively assigned.  Preserve
        # unfinished tasks as pending so they can be retried in a later run.
        self.world.release_assignments()
        return self._result(iterations)

    def _mission_active(self) -> bool:
        if self.world.time >= self.world.deadline:
            return False
        for task in self.world.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                return True
        return False

    def _result(self, iterations: int) -> MissionResult:
        completed: List[str] = []
        failed: List[str] = []
        pending: List[str] = []
        score = 0.0
        for task in self.world.tasks.values():
            if task.status is TaskStatus.COMPLETED:
                completed.append(task.id)
                score += TASK_VALUE.get(task.type, 1.0)
            elif task.status is TaskStatus.FAILED:
                failed.append(task.id)
                score -= 10.0
            else:
                pending.append(task.id)
        return MissionResult(
            iterations=iterations,
            completed=sorted(completed),
            failed=sorted(failed),
            pending=sorted(pending),
            score=score,
        )