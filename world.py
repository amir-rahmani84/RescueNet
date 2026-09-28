"""The World: state + mechanical operations.

This is the "physics" of RescueNet. It knows nothing about agents' reasoning,
planning, or goals. It only knows how to change state in response to
legitimate operations, and how to report state to observers.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from enums import ResourceType, TaskStatus
from models import Agent, Civilian, Event, Location, Road, Task


class World:
    def __init__(self, deadline: int = 50) -> None:
        self.time: int = 0
        self.deadline: int = deadline

        self.locations: Dict[str, Location] = {}
        self.roads: List[Road] = []
        self.tasks: Dict[str, Task] = {}
        self.civilians: Dict[str, Civilian] = {}
        self.events: List[Event] = []

        self.agents: Dict[str, Agent] = {}

        # A simple queue of pending events, keyed by the time they should fire.
        self._scheduled_events: List[Event] = []

    # ------------------------------------------------------------------ #
    #  Setup helpers
    # ------------------------------------------------------------------ #

    def add_location(self, location: Location) -> None:
        if location.name in self.locations:
            raise ValueError(f"Location {location.name!r} already exists")
        self.locations[location.name] = location

    def add_road(self, road: Road) -> None:
        if road.a not in self.locations or road.b not in self.locations:
            raise ValueError(f"Road {road.a}-{road.b} references unknown location")
        self.roads.append(road)

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def add_civilian(self, civilian: Civilian) -> None:
        self.civilians[civilian.id] = civilian
        self.locations[civilian.location].civilians.append(civilian.id)

    def add_agent(self, agent: Agent) -> None:
        if agent.id in self.agents:
            raise ValueError(f"Agent {agent.id!r} already exists")
        if agent.location not in self.locations:
            raise ValueError(
                f"Agent {agent.id!r} placed at unknown location {agent.location!r}"
            )
        self.agents[agent.id] = agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def get_road(self, a: str, b: str) -> Optional[Road]:
        for road in self.roads:
            if road.connects(a, b):
                return road
        return None

    def is_road_open(self, a: str, b: str) -> bool:
        road = self.get_road(a, b)
        return road is not None and not road.blocked

    def neighbors(self, node: str) -> Iterable[Tuple[str, Road]]:
        """Yield (neighbour, road) pairs for every connected road."""
        for road in self.roads:
            if node == road.a:
                yield road.b, road
            elif node == road.b:
                yield road.a, road

    # ------------------------------------------------------------------ #
    #  Core operations
    # ------------------------------------------------------------------ #

    def advance_time(self, amount: int = 1) -> None:
        """Move the clock forward, firing any events whose time has come."""
        for _ in range(amount):
            self.time += 1
            self._fire_due_events()

    # Basic function might need to change it later 
    def move_agent(self, agent_id: str, destination: str) -> bool:
        """Move an agent along an open road.

        Returns True if the move succeeded. Consumes fuel and advances time
        by the road's travel_time.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return False

        origin = agent.location
        road = self.get_road(origin, destination)
        if road is None or road.blocked:
            return False

        fuel_cost = road.travel_time
        if getattr(agent, "fuel", 0) < fuel_cost:
            return False

        agent.location = destination
        agent.fuel -= fuel_cost
        self.advance_time(road.travel_time)
        return True

    def complete_task(self, task_id: str, agent_id: Optional[str] = None) -> bool:
        task = self.tasks.get(task_id)
        if task is None or task.is_done():
            return False
        task.status = TaskStatus.COMPLETED
        if agent_id is not None:
            task.assigned_to = agent_id
        return True

    def fail_task(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if task is None or task.is_done():
            return False
        task.status = TaskStatus.FAILED
        return True

    def block_road(self, a: str, b: str) -> bool:
        road = self.get_road(a, b)
        if road is None:
            return False
        road.blocked = True
        self.events.append(Event(self.time, "road_blocked", {"a": a, "b": b}))
        return True

    def unblock_road(self, a: str, b: str) -> bool:
        road = self.get_road(a, b)
        if road is None:
            return False
        road.blocked = False
        self.events.append(Event(self.time, "road_unblocked", {"a": a, "b": b}))
        return True

    def add_resource(self, location: str, resource: ResourceType, qty: int) -> None:
        self.locations[location].add_resource(resource, qty)

    def remove_resource(self, location: str, resource: ResourceType, qty: int) -> bool:
        return self.locations[location].remove_resource(resource, qty)

    # ------------------------------------------------------------------ #
    #  Event scheduling
    # ------------------------------------------------------------------ #

    def schedule_event(self, event: Event) -> None:
        """Queue an event to fire at a specific simulation time."""
        self._scheduled_events.append(event)

    def _fire_due_events(self) -> None:
        due = [e for e in self._scheduled_events if e.time <= self.time]
        self._scheduled_events = [e for e in self._scheduled_events if e.time > self.time]
        for event in due:
            self._apply_event(event)

    def _apply_event(self, event: Event) -> None:
        """Translate a raw event into world-state changes."""
        if event.kind == "road_blocked":
            road = self.get_road(event.payload["a"], event.payload["b"])
            if road:
                road.blocked = True
        elif event.kind == "road_unblocked":
            road = self.get_road(event.payload["a"], event.payload["b"])
            if road:
                road.blocked = False
        elif event.kind == "civilian_found":
            civ = Civilian(
                id=event.payload["id"],
                location=event.payload["location"],
                needs_medical=event.payload.get("needs_medical", False),
                needs_evacuation=event.payload.get("needs_evacuation", False),
            )
            self.add_civilian(civ)
        elif event.kind == "resource_consumed":
            self.locations[event.payload["location"]].remove_resource(
                event.payload["resource"], event.payload["qty"]
            )
        # Record it in the log regardless.
        self.events.append(event)