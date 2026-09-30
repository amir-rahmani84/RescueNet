"""Pure data models for the RescueNet world.

These classes describe *what exists*. They deliberately contain almost no
behaviour beyond validation, so that the simulation logic in ``world.py``
stays the single source of truth for how the world evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from enums import Capability, ResourceType, TaskStatus, TaskType
from kb import KnowledgeBase


# --------------------------------------------------------------------------- #
#  Locations & roads
# --------------------------------------------------------------------------- #

@dataclass
class Location:
    """A named place on the map.

    Resources are stored as a mapping ``{ResourceType: quantity}`` so we can
    consume partial amounts (e.g. take 2 of 5 medical kits).
    """

    name: str
    resources: Dict[ResourceType, int] = field(default_factory=dict)
    civilians: List[str] = field(default_factory=list)

    def has_resource(self, resource: ResourceType, qty: int = 1) -> bool:
        return self.resources.get(resource, 0) >= qty

    def add_resource(self, resource: ResourceType, qty: int) -> None:
        self.resources[resource] = self.resources.get(resource, 0) + qty

    def remove_resource(self, resource: ResourceType, qty: int) -> bool:
        """Remove `qty` of `resource`. Returns True on success, False otherwise."""
        if not self.has_resource(resource, qty):
            return False
        self.resources[resource] -= qty
        if self.resources[resource] <= 0:
            del self.resources[resource]
        return True


@dataclass
class Road:
    """An undirected road between two locations.

    ``travel_time`` is measured in simulation ticks.
    ``risk`` is a 0.0–1.0 hazard weight (used later by planners/utility).
    ``blocked`` marks the road as temporarily impassable.
    """

    a: str
    b: str
    travel_time: int = 1
    risk: float = 0.0
    blocked: bool = False

    def connects(self, x: str, y: str) -> bool:
        return {x, y} == {self.a, self.b}

    def other_end(self, node: str) -> str:
        if node == self.a:
            return self.b
        if node == self.b:
            return self.a
        raise ValueError(f"{node!r} is not an endpoint of road {self.a}-{self.b}")


# --------------------------------------------------------------------------- #
#  Civilians & tasks
# --------------------------------------------------------------------------- #

@dataclass
class Civilian:
    """A person in the world who may need rescue, treatment, or evacuation."""

    id: str
    location: str
    needs_medical: bool = False
    needs_evacuation: bool = False
    rescued: bool = False
    treated: bool = False


@dataclass
class Task:
    """A mission objective.

    ``target`` is the location the task concerns. ``resource`` and ``quantity``
    only apply to DELIVER / TRANSPORT tasks.
    """

    id: str
    type: TaskType
    target: str
    description: str = ""
    resource: Optional[ResourceType] = None
    quantity: int = 0
    status: TaskStatus = TaskStatus.PENDING
    deadline: Optional[int] = None
    assigned_to: Optional[str] = None

    def is_done(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def mark(self, status: TaskStatus) -> None:
        self.status = status


# --------------------------------------------------------------------------- #
#  Events
# --------------------------------------------------------------------------- #

@dataclass
class Event:
    """A world change that has occurred (or is scheduled to occur).

    The ``payload`` is a free-form dict so different event kinds can carry
    different data without forcing a class hierarchy yet.
    """

    time: int
    kind: str          # e.g. "road_blocked", "civilian_found", ...
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Agents
# --------------------------------------------------------------------------- #

@dataclass
class Agent:
    """An autonomous actor carrying its own knowledge.

    gives the agent identity, capabilities, capacity, position,
    inventory and fuel. adds ``knowledge``: an explicit, symbolic
    picture of the world that the agent reasons over instead of poking at
    arbitrary Python attributes.
    """

    id: str
    capabilities: Set[Capability] = field(default_factory=set)
    capacity: int = 1
    location: str = ""
    inventory: Dict[ResourceType, int] = field(default_factory=dict)
    fuel: int = 20

    # --- symbolic layer --------------------------------------- #
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)

    # --- filled in by later -------------------------------------- #
    goal: Optional[str] = None
    plan: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Capability / capacity helpers
    # ------------------------------------------------------------------ #

    def has_capability(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def load(self) -> int:
        """Total number of item units currently carried."""
        return sum(self.inventory.values())

    def free_capacity(self) -> int:
        return max(0, self.capacity - self.load())

    def can_carry(self, qty: int) -> bool:
        return self.load() + qty <= self.capacity

    def add_item(self, resource: ResourceType, qty: int = 1) -> bool:
        if qty <= 0 or not self.can_carry(qty):
            return False
        self.inventory[resource] = self.inventory.get(resource, 0) + qty
        return True

    def remove_item(self, resource: ResourceType, qty: int = 1) -> bool:
        if self.inventory.get(resource, 0) < qty:
            return False
        self.inventory[resource] -= qty
        if self.inventory[resource] <= 0:
            del self.inventory[resource]
        return True

