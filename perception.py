"""Translate the current world state into an agent's symbolic knowledge."""

from __future__ import annotations

from enums import Capability, TaskStatus, TaskType
from kb import Fact
from models import Agent
from world import World


REQUIRED_CAPABILITY = {
    TaskType.DELIVER: Capability.TRANSPORT,
    TaskType.TRANSPORT: Capability.TRANSPORT,
    TaskType.RESCUE: Capability.MEDICAL,
    TaskType.EVACUATE: Capability.TRANSPORT,
    TaskType.REPAIR: Capability.REPAIR,
}


def perceive_agent(agent: Agent, world: World) -> None:
    """Refresh ``agent``'s knowledge base from the observable world state."""
    kb = agent.knowledge
    kb.clear()

    kb.add(Fact("at", (agent.id, agent.location)))
    kb.add(Fact("fuel", (agent.id, agent.fuel)))
    for capability in agent.capabilities:
        kb.add(Fact("has_capability", (agent.id, capability.value)))
    for resource in agent.inventory:
        kb.add(Fact("has", (agent.id, resource.value)))

    for road in world.roads:
        predicate = "road_blocked" if road.blocked else "road_open"
        kb.add(Fact(predicate, (road.a, road.b)))
        kb.add(Fact(predicate, (road.b, road.a)))
        kb.add(Fact("road_time", (road.a, road.b, road.travel_time)))
        kb.add(Fact("road_time", (road.b, road.a, road.travel_time)))
        kb.add(Fact("road_risk", (road.a, road.b, road.risk)))
        kb.add(Fact("road_risk", (road.b, road.a, road.risk)))

    for location in world.locations.values():
        for resource, quantity in location.resources.items():
            if quantity > 0:
                kb.add(Fact("supply_at", (location.name, resource.value)))

    for civilian in world.civilians.values():
        if civilian.rescued:
            continue
        kb.add(Fact("person_at", (civilian.id, civilian.location)))
        if civilian.needs_medical:
            kb.add(Fact("needs_medical", (civilian.id,)))
        if civilian.needs_evacuation:
            kb.add(Fact("needs_evacuation", (civilian.id,)))

    for task in world.tasks.values():
        if task.status is not TaskStatus.PENDING:
            continue
        kb.add(Fact("task_pending", (task.id,)))
        kb.add(Fact("task_at", (task.id, task.target)))
        kb.add(Fact("task_type", (task.id, task.type.value)))
        required = REQUIRED_CAPABILITY.get(task.type)
        if required is not None:
            kb.add(Fact("requires_capability", (task.id, required.value)))
        if task.type is TaskType.REPAIR:
            kb.add(Fact("damaged", (task.target,)))
