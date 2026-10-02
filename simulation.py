"""Step 14 -- the central agent loop.

The loop keeps the documented shape (OBSERVE -> ... -> WORLD CHANGES),
but each agent now perceives and forward-chains exactly once per turn.

Every decision is parameterised by a policy.  The Simulation accepts
either a ``Policy`` (uniform for the whole team) or a ``TeamPolicy``
(one Policy per agent id plus a shared coordinator policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from actions import actions_for_policy
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
from enums import TaskStatus, TaskType
from perception import perceive_agent
from planner import delivery_heuristic, location_heuristic, plan
from policy import Policy, TeamPolicy, DEFAULT_POLICY, Genome
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
        self,
        world: World,
        *,
        hub: Optional[CommunicationHub] = None,
        max_tasks_per_agent: Optional[int] = None,
        max_ticks: int = 200,
        plan_max_expansions: int = 20000,
        policy: Optional[Genome] = None,
    ) -> None:
        self.world = world
        self.hub = hub or CommunicationHub()

        # -- Policy normalisation ---------------------------------------- #
        if policy is None:
            self.team = TeamPolicy.uniform(DEFAULT_POLICY)
        elif isinstance(policy, TeamPolicy):
            self.team = policy
        elif isinstance(policy, Policy):
            self.team = TeamPolicy.uniform(policy)
        else:
            raise TypeError(f"Unknown policy type: {type(policy)!r}")

        # The "coordinator" policy drives global decisions: assignment,
        # negotiation, and claim bonuses.
        self.coordinator_policy = self.team.shared

        self.max_tasks_per_agent = (
            max_tasks_per_agent
            if max_tasks_per_agent is not None
            else self.coordinator_policy.max_tasks_per_agent
        )
        self.max_ticks = max_ticks
        self.plan_max_expansions = plan_max_expansions
        self.negotiation: Optional[NegotiationResult] = None

        self._failed_goals: Set[Tuple[str, str]] = set()
        self._last_event_count: int = 0

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def setup(self) -> NegotiationResult:
        for agent in self.world.agents.values():
            perceive_agent(agent, self.world)
            forward_chain(agent.knowledge)

        cp = self.coordinator_policy
        claims = solicit_claims(
            self.world, self.hub, policy=cp,
        )
        proposal = optimize_assignment(
            self.world, self.world.agents,
            max_tasks_per_agent=self.max_tasks_per_agent,
            score=claim_aware_score(
                self.world, claims,
                claim_bonus=cp.w_cooperation,
                policy=cp,
            ),
            policy=cp,
        )
        confirmed = negotiate_assignment(
            self.world, self.hub, proposal, claims=claims
        )
        confirmed = reassign_rejected(
            self.world, self.hub, confirmed,
            max_tasks_per_agent=self.max_tasks_per_agent,
            claims=claims,
            claim_bonus=cp.w_cooperation,
            policy=cp,
        )

        for task_id, agent_id in confirmed.confirmed.items():
            self.world.agents[agent_id].goal = task_id

        self.negotiation = confirmed
        return confirmed

    # ------------------------------------------------------------------ #
    #  Tick
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
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
        policy = self.team.for_agent(agent.id)

        # ---- OBSERVE + REPRESENT + REASON (once per turn) ------------ #
        perceive_agent(agent, self.world)
        forward_chain(agent.knowledge)

        # ---- SELECT GOAL --------------------------------------------- #
        if not self._goal_still_valid(agent):
            chosen = self._best_goal(agent, policy)
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
            if self._plan_for(agent, task, policy):
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
                alt = self._best_goal(agent, policy)
                if alt is None or alt == agent.goal:
                    agent.plan = []
                    agent.plan_index = 0
                    return
                agent.goal = alt
                if not self._plan_for(agent, self.world.tasks[alt], policy):
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
                if task is not None and self._plan_for(agent, task, policy):
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
        except ValueError:
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

    def _best_goal(self, agent, policy: Policy) -> Optional[str]:
        """Highest-priority goal this agent hasn't already failed to plan.

        The policy's ``goal_top_k`` caps how deep into the ranking the
        agent is willing to look.
        """
        ranked = rank_goals(self.world, agent, policy=policy)
        window = ranked[: max(1, policy.goal_top_k)]
        for candidate in window:
            if (agent.id, candidate.task_id) not in self._failed_goals:
                return candidate.task_id
        return None

    def _plan_for(self, agent, task, policy: Policy) -> bool:
        goal_test = make_goal_test(self.world, agent.id, task)
        action_set = actions_for_policy(policy)

        if (
            task.type.value in ("deliver", "transport")
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
            max_expansions=self.plan_max_expansions,
        )
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