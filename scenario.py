"""Concrete scenario definitions.

Right now there is one scenario: the 7-location city from the spec. Later,
this module can hold multiple scenarios (needed for evolutionary search over
decision policies).
"""

from __future__ import annotations

from enums import Capability, ResourceType, TaskType
from models import Agent, Civilian, Location, Road, Task
from world import World


def build_agents() -> list[Agent]:
    """The three agents from the spec."""
    return [
        Agent(
            id="Rescue-1",
            capabilities={Capability.MEDICAL},
            capacity=2,
            location="E",
            inventory={ResourceType.MEDICAL_KIT: 1},
            fuel=20,
        ),
        Agent(
            id="Logistics-1",
            capabilities={Capability.TRANSPORT},
            capacity=5,
            location="A",
            inventory={},
            fuel=20,
        ),
        Agent(
            id="Engineer-1",
            capabilities={Capability.REPAIR},
            capacity=3,
            location="B",
            inventory={ResourceType.EQUIPMENT: 1},
            fuel=20,
        ),
    ]


def build_default_world() -> World:
    world = World(deadline=50)

    # --- Locations ---------------------------------------------------- #
    world.add_location(Location("A", {ResourceType.FOOD: 3, ResourceType.FUEL: 4}))
    world.add_location(Location("B", {ResourceType.EQUIPMENT: 2}))
    world.add_location(Location("C", {ResourceType.MEDICINE: 2}))
    world.add_location(Location("D", {ResourceType.MEDICAL_KIT: 3}))
    world.add_location(Location("E", {}))
    world.add_location(Location("F", {ResourceType.FOOD: 2}))
    world.add_location(Location("G", {}))

    # --- Roads -------------------------------------------------------- #
    #          a    b    time  risk
    world.add_road(Road("A", "B", 2, 0.1))
    world.add_road(Road("A", "D", 2, 0.1))
    world.add_road(Road("B", "C", 2, 0.1))
    world.add_road(Road("B", "E", 2, 0.2))
    world.add_road(Road("C", "F", 2, 0.1))
    world.add_road(Road("D", "E", 2, 0.1))
    world.add_road(Road("E", "F", 2, 0.2))
    world.add_road(Road("E", "G", 3, 0.5))   # remote, risky

    # --- Civilians ---------------------------------------------------- #
    world.add_civilian(Civilian("civ1", "F", needs_medical=True))
    world.add_civilian(Civilian("civ2", "G", needs_evacuation=True))

    # --- Agents (Step 2) ---------------------------------------------- #
    for agent in build_agents():
        world.add_agent(agent)

    # --- Tasks -------------------------------------------------------- #
    world.add_task(Task(
        id="deliver_medicine_D",
        type=TaskType.DELIVER,
        target="D",
        resource=ResourceType.MEDICINE,
        quantity=1,
        description="Deliver medicine to D",
    ))
    world.add_task(Task(
        id="rescue_civilian_F",
        type=TaskType.RESCUE,
        target="F",
        description="Rescue civilian at F",
    ))
    world.add_task(Task(
        id="deliver_food_G",
        type=TaskType.DELIVER,
        target="G",
        resource=ResourceType.FOOD,
        quantity=2,
        description="Deliver food to G",
    ))
    world.add_task(Task(
        id="repair_generator_C",
        type=TaskType.REPAIR,
        target="C",
        description="Repair generator at C",
    ))
    world.add_task(Task(
        id="evacuate_civilian_G",
        type=TaskType.EVACUATE,
        target="G",
        description="Evacuate civilian at G",
    ))
    world.add_task(Task(
        id="transport_equipment_B",
        type=TaskType.TRANSPORT,
        target="B",
        resource=ResourceType.EQUIPMENT,
        quantity=1,
        description="Transport equipment to B",
    ))

    return world