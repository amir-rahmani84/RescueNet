"""Translate current world state into an agent's symbolic knowledge.

The KB is still rebuilt from scratch each call, but the KB itself is now
indexed (see `kb.py`), so the rebuild is cheap. The important change is
that positive effects (`treated`, `rescued`, `repaired`, progress on
deliveries) are now visible in the KB, so the agent's symbolic view stays
consistent with the world after an action executes.
"""

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
    """Refresh `agent`'s knowledge base from the observable world state."""
    kb = agent.knowledge
    kb.clear()

    # --- self --------------------------------------------------------- #
    kb.add(Fact("at", (agent.id, agent.location)))
    kb.add(Fact("fuel", (agent.id, agent.fuel)))
    kb.add(Fact("capacity", (agent.id, agent.capacity)))
    kb.add(Fact("elapsed", (agent.id, 0)))
    for capability in agent.capabilities:
        kb.add(Fact("has_capability", (agent.id, capability.value)))
    for resource, quantity in agent.inventory.items():
        kb.add(Fact("has", (agent.id, resource.value)))
        kb.add(Fact("inventory", (agent.id, resource.value, quantity)))

    # --- roads -------------------------------------------------------- #
    for road in world.roads:
        predicate = "road_blocked" if road.blocked else "road_open"
        kb.add(Fact(predicate, (road.a, road.b)))
        kb.add(Fact(predicate, (road.b, road.a)))
        kb.add(Fact("road_time", (road.a, road.b, road.travel_time)))
        kb.add(Fact("road_time", (road.b, road.a, road.travel_time)))
        kb.add(Fact("road_risk", (road.a, road.b, road.risk)))
        kb.add(Fact("road_risk", (road.b, road.a, road.risk)))

    # --- resources ---------------------------------------------------- #
    for location in world.locations.values():
        for resource, quantity in location.resources.items():
            if quantity > 0:
                kb.add(Fact("supply_at", (location.name, resource.value)))
                kb.add(Fact("resource_at",
                            (location.name, resource.value, quantity)))

    # --- civilians ---------------------------------------------------- #
    for civilian in world.civilians.values():
        kb.add(Fact("person_at", (civilian.id, civilian.location)))
        if civilian.needs_medical:
            kb.add(Fact("needs_medical", (civilian.id,)))
        if civilian.needs_evacuation:
            kb.add(Fact("needs_evacuation", (civilian.id,)))
        if civilian.treated:
            kb.add(Fact("treated", (civilian.id,)))
        if civilian.rescued:
            kb.add(Fact("rescued", (civilian.id,)))
        if civilian.evacuated:
            kb.add(Fact("evacuated", (civilian.id,)))

    # --- tasks -------------------------------------------------------- #
    for task in world.tasks.values():
        if task.status is TaskStatus.PENDING:
            kb.add(Fact("task_pending", (task.id,)))
        if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
            kb.add(Fact("task_at", (task.id, task.target)))
            kb.add(Fact("task_type", (task.id, task.type.value)))
            required = REQUIRED_CAPABILITY.get(task.type)
            if required is not None:
                kb.add(Fact("requires_capability", (task.id, required.value)))
            if task.type is TaskType.REPAIR and task.target not in {
                t.target for t in world.tasks.values()
                if t.type is TaskType.REPAIR and t.status is TaskStatus.COMPLETED
            }:
                kb.add(Fact("damaged", (task.target,)))

        # Progress and completion facts.
        if task.progress > 0 and task.resource is not None:
            kb.add(Fact("delivered_at",
                        (task.target, task.resource.value, task.progress)))
        if task.status is TaskStatus.COMPLETED and task.type is TaskType.REPAIR:
            kb.add(Fact("repaired", (task.target,)))