"""Communication and conflict resolution for RescueNet.

Changes:
* CommunicationHub caps its message log so long missions do not grow
  unbounded memory.
* `reassign_rejected` refreshes each agent's KB before the follow-up CSP
  run, so the domains reflect the current world instead of stale facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from constraints import CSPResult
from decision import (
    GoalCandidate,
    mission_utility,
    optimize_assignment,
    rank_goals,
)
from enums import TaskStatus
from perception import perceive_agent
from reasoner import forward_chain
from world import World


class MessageType(Enum):
    REQUEST = "request"
    INFORM = "inform"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    CLAIM = "claim"
    WARNING = "warning"


@dataclass(frozen=True)
class Message:
    sender: str
    receiver: str
    type: MessageType
    content: str
    time: int
    task_id: Optional[str] = None
    payload: Optional[dict] = None


@dataclass(frozen=True)
class Claim:
    agent_id: str
    task_id: str
    priority: float
    estimated_cost: float


@dataclass(frozen=True)
class NegotiationResult:
    confirmed: Dict[str, str]
    rejected: Dict[str, str]
    round: int = 0
    exhausted: bool = False


class CommunicationHub:
    """In-memory message transport with a bounded log."""

    DEFAULT_MAX_MESSAGES = 10_000

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def _trim(self) -> None:
        if len(self.messages) > self.max_messages:
            # Amortised: only copies when we exceed by one.
            self.messages = self.messages[-self.max_messages:]

    def send(
        self, world: World, sender: str, receiver: str,
        message_type: MessageType, content: str,
        task_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> Message:
        if sender not in world.agents or receiver not in world.agents:
            raise ValueError("Both message endpoints must be known agents")
        message = Message(
            sender, receiver, message_type, content, world.time, task_id, payload
        )
        self.messages.append(message)
        self._trim()
        return message

    def broadcast(
        self, world: World, sender: str,
        message_type: MessageType, content: str,
        task_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> List[Message]:
        return [
            self.send(world, sender, receiver, message_type, content,
                      task_id, payload)
            for receiver in world.agents
            if receiver != sender
        ]

    def inbox(self, receiver: str) -> List[Message]:
        return [m for m in self.messages if m.receiver == receiver]


def _coordinator_id(world: World) -> str:
    return next(iter(world.agents))


def solicit_claims(
    world: World, hub: CommunicationHub, *, top_k: int = 2,
) -> Dict[str, List[Claim]]:
    claims: Dict[str, List[Claim]] = {}
    coordinator = _coordinator_id(world)
    for agent in world.agents.values():
        for candidate in rank_goals(world, agent)[:top_k]:
            claims.setdefault(candidate.task_id, []).append(
                Claim(
                    agent_id=agent.id,
                    task_id=candidate.task_id,
                    priority=candidate.priority,
                    estimated_cost=candidate.estimated_cost,
                )
            )
            hub.send(
                world, agent.id, coordinator, MessageType.CLAIM,
                f"I claim {candidate.task_id} "
                f"(priority {candidate.priority:.2f})",
                candidate.task_id,
                payload={
                    "round": 0,
                    "priority": candidate.priority,
                    "estimated_cost": candidate.estimated_cost,
                },
            )
    for bucket in claims.values():
        bucket.sort(key=lambda c: (-c.priority, c.agent_id))
    return claims


def claim_aware_score(
    world: World, claims: Dict[str, List[Claim]],
    *, claim_bonus: float = 5.0,
) -> Callable[[Dict[str, str], List[str]], float]:
    def score(assignment: Dict[str, str], unassigned: List[str]) -> float:
        base = mission_utility(world, assignment, unassigned)
        if base == float("-inf"):
            return base
        bonus = 0.0
        for task_id, agent_id in assignment.items():
            for claim in claims.get(task_id, ()):
                if claim.agent_id == agent_id:
                    bonus += claim_bonus * claim.priority
                    break
        return base + bonus
    return score


def negotiate_assignment(
    world: World, hub: CommunicationHub, result: CSPResult,
    coordinator_id: Optional[str] = None,
    *, claims: Optional[Dict[str, List[Claim]]] = None,
    round: int = 0,
) -> NegotiationResult:
    if not world.agents:
        return NegotiationResult({}, {}, round=round)

    coordinator = coordinator_id or _coordinator_id(world)
    if coordinator not in world.agents:
        raise ValueError(f"Unknown coordinator: {coordinator}")

    confirmed: Dict[str, str] = {}
    rejected: Dict[str, str] = {}

    for task_id, agent_id in result.assignment.items():
        task = world.tasks[task_id]
        agent = world.agents[agent_id]

        note = ""
        if claims and claims.get(task_id):
            top = claims[task_id][0]
            if top.agent_id == agent_id:
                note = f" (top claim, priority {top.priority:.2f})"

        hub.send(
            world, coordinator, agent_id, MessageType.OFFER,
            f"Assignment offer: {task.description}{note}",
            task_id,
            payload={"round": round, "coordinator": coordinator},
        )

        still_feasible = (
            task.status is TaskStatus.PENDING
            and agent.knowledge.contains(("can_assign", (agent_id, task_id)))
        )

        if not still_feasible:
            rejected[task_id] = agent_id
            hub.send(
                world, agent_id, coordinator, MessageType.REJECT,
                f"Cannot accept assignment: {task.description}",
                task_id,
                payload={"round": round, "reason": "not_feasible"},
            )
            continue

        confirmed[task_id] = agent_id
        task.assigned_to = agent_id
        task.mark(TaskStatus.IN_PROGRESS)
        hub.send(
            world, agent_id, coordinator, MessageType.ACCEPT,
            f"Accepted assignment: {task.description}",
            task_id,
            payload={"round": round},
        )

    for task_id in result.unassigned:
        hub.broadcast(
            world, coordinator, MessageType.INFORM,
            f"No feasible assignment yet for {task_id}",
            task_id,
            payload={"round": round, "reason": "no_domain"},
        )

    return NegotiationResult(confirmed, rejected, round=round)


def reassign_rejected(
    world: World, hub: CommunicationHub,
    previous: NegotiationResult, *,
    max_tasks_per_agent: int = 2,
    claims: Optional[Dict[str, List[Claim]]] = None,
    max_rounds: int = 3,
) -> NegotiationResult:
    if not previous.rejected:
        return previous

    coordinator = _coordinator_id(world)
    confirmed: Dict[str, str] = dict(previous.confirmed)
    rejected: Dict[str, str] = dict(previous.rejected)
    forbidden: Dict[str, Set[str]] = {
        tid: {aid} for tid, aid in previous.rejected.items()
    }
    round_index = previous.round

    for _ in range(max_rounds):
        if not rejected:
            break
        round_index += 1

        # Refresh each agent's KB so can_assign reflects the current world.
        for agent in world.agents.values():
            perceive_agent(agent, world)
            forward_chain(agent.knowledge)

        result = optimize_assignment(
            world, world.agents,
            max_tasks_per_agent=max_tasks_per_agent,
            score=claim_aware_score(world, claims or {}),
            forbidden=forbidden,
        )
        round_result = negotiate_assignment(
            world, hub, result,
            coordinator_id=coordinator,
            claims=claims,
            round=round_index,
        )

        confirmed.update(round_result.confirmed)
        new_rejected = dict(round_result.rejected)

        if new_rejected == rejected:
            rejected = new_rejected
            break

        for task_id, agent_id in new_rejected.items():
            forbidden.setdefault(task_id, set()).add(agent_id)
            hub.broadcast(
                world, coordinator, MessageType.WARNING,
                f"{agent_id} rejected {task_id}; excluding this pair "
                f"from round {round_index + 1}",
                task_id,
                payload={"round": round_index, "excluded": agent_id},
            )

        rejected = new_rejected

    exhausted = bool(rejected)
    if exhausted:
        hub.broadcast(
            world, coordinator, MessageType.WARNING,
            f"Negotiation exhausted after {round_index} rounds; "
            f"{len(rejected)} task(s) remain unassignable",
            payload={"round": round_index, "remaining": sorted(rejected)},
        )

    return NegotiationResult(
        confirmed=confirmed, rejected=rejected,
        round=round_index, exhausted=exhausted,
    )