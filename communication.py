"""Communication and conflict resolution for RescueNet (Steps 10 and 11)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from constraints import CSPResult
from enums import TaskStatus
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


@dataclass(frozen=True)
class NegotiationResult:
    """Assignments confirmed by agents after global optimization."""

    confirmed: Dict[str, str]
    rejected: Dict[str, str]


class CommunicationHub:
    """In-memory message transport with a world-time message log."""

    def __init__(self) -> None:
        self.messages: List[Message] = []

    def send(
        self,
        world: World,
        sender: str,
        receiver: str,
        message_type: MessageType,
        content: str,
        task_id: Optional[str] = None,
    ) -> Message:
        if sender not in world.agents or receiver not in world.agents:
            raise ValueError("Both message endpoints must be known agents")
        message = Message(
            sender, receiver, message_type, content, world.time, task_id
        )
        self.messages.append(message)
        return message

    def broadcast(
        self,
        world: World,
        sender: str,
        message_type: MessageType,
        content: str,
        task_id: Optional[str] = None,
    ) -> List[Message]:
        return [
            self.send(world, sender, receiver, message_type, content, task_id)
            for receiver in world.agents
            if receiver != sender
        ]

    def inbox(self, receiver: str) -> List[Message]:
        return [message for message in self.messages if message.receiver == receiver]


def negotiate_assignment(
    world: World,
    hub: CommunicationHub,
    result: CSPResult,
    coordinator_id: Optional[str] = None,
) -> NegotiationResult:
    """Announce and confirm the CSP's global assignment.

    Communication validates the current local facts, but never creates a new
    task assignment.  A rejected offer is returned to the coordinator so a
    later CSP run can reconsider it.
    """
    if not world.agents:
        return NegotiationResult({}, {})
    coordinator = coordinator_id or next(iter(world.agents))
    if coordinator not in world.agents:
        raise ValueError(f"Unknown coordinator: {coordinator}")

    confirmed: Dict[str, str] = {}
    rejected: Dict[str, str] = {}
    for task_id, agent_id in result.assignment.items():
        task = world.tasks[task_id]
        agent = world.agents[agent_id]
        offer = f"Assignment offer: {task.description}"
        if coordinator != agent_id:
            hub.send(world, coordinator, agent_id, MessageType.OFFER, offer, task_id)

        still_feasible = (
            task.status is TaskStatus.PENDING
            and agent.knowledge.contains(("can_assign", (agent_id, task_id)))
        )
        if not still_feasible:
            rejected[task_id] = agent_id
            if coordinator != agent_id:
                hub.send(
                    world,
                    agent_id,
                    coordinator,
                    MessageType.REJECT,
                    f"Cannot accept assignment: {task.description}",
                    task_id,
                )
            continue

        confirmed[task_id] = agent_id
        task.assigned_to = agent_id
        task.mark(TaskStatus.IN_PROGRESS)
        if coordinator != agent_id:
            hub.send(
                world,
                agent_id,
                coordinator,
                MessageType.ACCEPT,
                f"Accepted assignment: {task.description}",
                task_id,
            )

    for task_id in result.unassigned:
        hub.broadcast(
            world,
            coordinator,
            MessageType.INFORM,
            f"No feasible assignment yet for {task_id}",
            task_id,
        )
    return NegotiationResult(confirmed, rejected)