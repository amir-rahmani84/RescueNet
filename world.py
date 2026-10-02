"""Authoritative world state and mutation operations.

Optimizations:

* Adjacency list and (a,b) road index so neighbor/road lookups are O(1).
* `road_version` counter so pathfinding can invalidate its all-pairs cache
  exactly when something road-related changes.
* Scheduled events are stored in a heap; `_fire_due_events` pops only due
  entries instead of scanning the whole schedule.
* Task deadlines are stored in a heap; `_fail_expired_tasks` pops only
  expired tasks instead of scanning all tasks every tick.
* `execute_action` is split into `can_execute` (pure check against a
  caller-supplied KB) and `apply_action` (mutate world state only). This
  lets `Simulation` perceive and forward-chain once per agent turn instead
  of three times.
* Task completion now runs *before* `advance_time(1)`, so a task that
  finishes at its exact deadline is marked completed, not failed.
* Unknown event kinds are recorded on ``World.unknown_events`` and warned
  about, so a missing handler is visible instead of silently dropped.
"""

from __future__ import annotations

import heapq
import itertools
import warnings
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
)

from enums import ResourceType, TaskStatus, TaskType
from events import EventScheduler
from models import Agent, Civilian, Event, Location, Road, Task


EventListener = Callable[["World", Event], None]


#: Every event kind ``World`` knows how to apply.  Cross-checked by
#: ``event_audit`` against ``events.EVENT_KINDS`` and
#: ``display._EVENT_FORMATTERS``.
HANDLED_EVENT_KINDS: FrozenSet[str] = frozenset({
    "road_blocked",
    "road_unblocked",
    "civilian_found",
    "resource_consumed",
    "resource_depleted",
    "fuel_low",
    "road_risk_increased",
    "civilian_condition_changed",
    "task_added",
})


class World:
    def __init__(self, deadline: int = 50,
                 scheduler: Optional[EventScheduler] = None) -> None:
        self.time: int = 0
        self.deadline: int = deadline

        self.locations: Dict[str, Location] = {}
        self.roads: List[Road] = []
        self.tasks: Dict[str, Task] = {}
        self.civilians: Dict[str, Civilian] = {}
        self.events: List[Event] = []
        # Events received with a kind we don't recognise.  Reported by
        # ``event_audit`` and by any listener that cares to look.
        self.unknown_events: List[Event] = []
        self.agents: Dict[str, Agent] = {}

        # Indices.
        self._road_index: Dict[Tuple[str, str], Road] = {}
        self._adj: Dict[str, List[Tuple[str, Road]]] = {}

        # Versioned travel cache (owned by pathfinding).
        self.road_version: int = 0
        self._travel_cache: Dict[Tuple[str, str], float] = {}
        self._travel_cache_version: int = -1

        # Event / deadline scheduling.
        self._event_seq = itertools.count()
        self._scheduled: List[Tuple[int, int, Event]] = []
        self._scheduler: Optional[EventScheduler] = scheduler
        self._event_listeners: List[EventListener] = []

        self._task_deadlines: List[Tuple[int, int, str]] = []
        self._task_seq = itertools.count()

    # ------------------------------------------------------------------ #
    #  Mutation helpers
    # ------------------------------------------------------------------ #

    def _bump_road_version(self) -> None:
        self.road_version += 1

    # ------------------------------------------------------------------ #
    #  Setup API
    # ------------------------------------------------------------------ #

    def add_location(self, location: Location) -> None:
        self.locations[location.name] = location

    def add_road(self, road: Road) -> None:
        if road.a not in self.locations or road.b not in self.locations:
            raise ValueError("Both road endpoints must be known locations")
        self.roads.append(road)
        self._road_index[(road.a, road.b)] = road
        self._road_index[(road.b, road.a)] = road
        self._adj.setdefault(road.a, []).append((road.b, road))
        self._adj.setdefault(road.b, []).append((road.a, road))
        self._bump_road_version()

    def add_agent(self, agent: Agent) -> None:
        if agent.location not in self.locations:
            raise ValueError(f"Unknown agent location: {agent.location}")
        self.agents[agent.id] = agent

    def add_civilian(self, civilian: Civilian) -> None:
        if civilian.location not in self.locations:
            raise ValueError(f"Unknown civilian location: {civilian.location}")
        previous = self.civilians.get(civilian.id)
        if previous is not None and civilian.id in self.locations[previous.location].civilians:
            self.locations[previous.location].civilians.remove(civilian.id)
        self.civilians[civilian.id] = civilian
        if civilian.id not in self.locations[civilian.location].civilians:
            self.locations[civilian.location].civilians.append(civilian.id)

    def add_task(self, task: Task) -> None:
        if task.target not in self.locations:
            raise ValueError(f"Unknown task location: {task.target}")
        self.tasks[task.id] = task
        if task.deadline is not None:
            heapq.heappush(
                self._task_deadlines,
                (task.deadline, next(self._task_seq), task.id),
            )

    # ------------------------------------------------------------------ #
    #  Lookups
    # ------------------------------------------------------------------ #

    def neighbors(self, location: str) -> Iterable[Tuple[str, Road]]:
        return self._adj.get(location, ())

    def get_road(self, a: str, b: str) -> Optional[Road]:
        return self._road_index.get((a, b))

    def block_road(self, a: str, b: str) -> None:
        road = self.get_road(a, b)
        if road is None:
            raise ValueError(f"Unknown road: {a}-{b}")
        if not road.blocked:
            road.blocked = True
            self._bump_road_version()

    def unblock_road(self, a: str, b: str) -> None:
        road = self.get_road(a, b)
        if road is None:
            raise ValueError(f"Unknown road: {a}-{b}")
        if road.blocked:
            road.blocked = False
            self._bump_road_version()

    # ------------------------------------------------------------------ #
    #  Clock
    # ------------------------------------------------------------------ #

    def advance_time(self, ticks: int = 1) -> None:
        if ticks < 0:
            raise ValueError("ticks cannot be negative")
        for _ in range(ticks):
            self.time += 1
            self._fire_due_events()
            self._fail_expired_tasks()

    # ------------------------------------------------------------------ #
    #  Agent / task operations
    # ------------------------------------------------------------------ #

    def move_agent(self, agent_id: str, destination: str) -> None:
        agent = self.agents[agent_id]
        road = self.get_road(agent.location, destination)
        if road is None or road.blocked:
            raise ValueError(f"No open road from {agent.location} to {destination}")
        if agent.fuel < road.travel_time:
            raise ValueError(f"{agent_id} does not have enough fuel")
        agent.fuel -= road.travel_time
        agent.location = destination
        self.advance_time(road.travel_time)

    def release_assignments(self) -> None:
        for task in self.tasks.values():
            if task.status is TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.PENDING
                task.assigned_to = None

    def complete_task(self, task_id: str, agent_id: Optional[str] = None) -> None:
        task = self.tasks[task_id]
        if agent_id is not None and task.assigned_to not in (None, agent_id):
            raise ValueError(f"Task {task_id} is assigned to another agent")
        task.mark(TaskStatus.COMPLETED)
        task.assigned_to = agent_id or task.assigned_to

    def fail_task(self, task_id: str, reason: str = "") -> None:
        task = self.tasks[task_id]
        task.mark(TaskStatus.FAILED)
        task.failure_reason = reason

    def pick_up(self, agent_id: str, location: str, resource: ResourceType,
                quantity: int = 1) -> None:
        agent = self.agents[agent_id]
        if agent.location != location:
            raise ValueError("Agent is not at the pickup location")
        if not self.locations[location].remove_resource(resource, quantity):
            raise ValueError("Resource pickup is not possible")
        if not agent.add_item(resource, quantity):
            self.locations[location].add_resource(resource, quantity)
            raise ValueError("Agent capacity is insufficient")

    def deliver(self, agent_id: str, location: str, resource: ResourceType,
                quantity: int = 1) -> None:
        agent = self.agents[agent_id]
        if agent.location != location or not agent.remove_item(resource, quantity):
            raise ValueError("Resource delivery is not possible")
        self.locations[location].add_resource(resource, quantity)
        if agent.goal in self.tasks:
            task = self.tasks[agent.goal]
            if task.resource is resource and task.target == location:
                task.progress += quantity

    def treat(self, civilian_id: str, agent_id: Optional[str] = None) -> None:
        civilian = self.civilians[civilian_id]
        if agent_id is not None:
            if not self.agents[agent_id].remove_item(ResourceType.MEDICAL_KIT):
                raise ValueError("Medical kit is required")
        civilian.needs_medical = False
        civilian.treated = True

    def rescue(self, civilian_id: str) -> None:
        civilian = self.civilians[civilian_id]
        civilian.rescued = True
        civilian.needs_evacuation = False

    def evacuate(self, civilian_id: str, destination: str) -> None:
        civilian = self.civilians[civilian_id]
        if destination not in self.locations:
            raise ValueError(f"Unknown evacuation destination: {destination}")
        origin = civilian.location
        if civilian.id in self.locations[origin].civilians:
            self.locations[civilian.location].civilians.remove(civilian.id)
        civilian.location = destination
        civilian.needs_evacuation = False
        civilian.evacuated = True
        civilian.evacuated_from = origin
        self.locations[destination].civilians.append(civilian.id)

    def repair(self, location: str) -> None:
        for task in self.tasks.values():
            if task.type is TaskType.REPAIR and task.target == location:
                task.progress = 1

    # ------------------------------------------------------------------ #
    #  Action execution (split)
    # ------------------------------------------------------------------ #

    def can_execute(self, agent_id: str, action) -> bool:
        """Pure precondition check against the agent's *current* KB.

        Callers are expected to have just called `perceive_agent` and
        `forward_chain` on this agent, so derived predicates are present.
        """
        return action.preconditions_hold(self.agents[agent_id].knowledge)

    def apply_action(self, agent_id: str, action) -> None:
        """Mutate world state. Preconditions must already hold."""
        agent = self.agents[agent_id]
        name = action.action.name
        bindings = action.bindings

        if name == "Move":
            self.move_agent(agent_id, bindings["?to"])
            self._complete_agent_task(agent)
            return

        if name == "PickUp":
            self.pick_up(agent_id, bindings["?loc"],
                         ResourceType(bindings["?resource"]))
        elif name == "Deliver":
            self.deliver(agent_id, bindings["?loc"],
                         ResourceType(bindings["?resource"]))
        elif name == "Treat":
            self.treat(bindings["?person"], agent_id)
        elif name == "Rescue":
            self.rescue(bindings["?person"])
        elif name == "Evacuate":
            self.evacuate(bindings["?person"], bindings["?safe"])
        elif name == "Repair":
            self.repair(bindings["?loc"])
        elif name == "ClearRoad":
            self.unblock_road(bindings["?a"], bindings["?b"])
        else:
            raise ValueError(f"Unknown action: {name}")

        # Complete *before* advancing time, so an on-time finish counts.
        self._complete_agent_task(agent)
        self.advance_time(1)

    def _complete_agent_task(self, agent: Agent) -> None:
        if agent.goal in self.tasks and self.task_satisfied(self.tasks[agent.goal]):
            self.complete_task(agent.goal, agent.id)

    def task_satisfied(self, task: Task) -> bool:
        if task.type is TaskType.REPAIR:
            return task.progress > 0
        if task.type is TaskType.RESCUE:
            return any(c.rescued and c.location == task.target
                       for c in self.civilians.values())
        if task.type is TaskType.EVACUATE:
            return any(c.evacuated and c.evacuated_from == task.target
                       for c in self.civilians.values())
        return task.progress >= task.quantity

    # ------------------------------------------------------------------ #
    #  Scheduled events and deadlines
    # ------------------------------------------------------------------ #

    def set_event_scheduler(self, scheduler: EventScheduler) -> None:
        self._scheduler = scheduler

    def add_event_listener(self, callback: EventListener) -> None:
        self._event_listeners.append(callback)

    def schedule_event(self, event: Event) -> None:
        heapq.heappush(
            self._scheduled,
            (event.time, next(self._event_seq), event),
        )

    def _fire_due_events(self) -> None:
        due: List[Event] = []
        while self._scheduled and self._scheduled[0][0] <= self.time:
            _, _, event = heapq.heappop(self._scheduled)
            due.append(event)
        if self._scheduler is not None:
            due.extend(self._scheduler.due(self.time))
        for event in due:
            self._apply_event(event)

    def _fail_expired_tasks(self) -> None:
        while (self._task_deadlines
               and self._task_deadlines[0][0] < self.time):
            _, _, task_id = heapq.heappop(self._task_deadlines)
            task = self.tasks.get(task_id)
            if task is None:
                continue
            if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                self.fail_task(task.id, "task deadline expired")

    # ------------------------------------------------------------------ #
    #  Event application
    # ------------------------------------------------------------------ #

    def _apply_event(self, event: Event) -> None:
        if event.kind not in HANDLED_EVENT_KINDS:
            warnings.warn(
                f"World received unknown event kind {event.kind!r}; "
                f"recorded but not applied.",
                stacklevel=2,
            )
            self.unknown_events.append(event)
            self.events.append(event)
            for listener in self._event_listeners:
                listener(self, event)
            return

        if event.kind == "road_blocked":
            road = self.get_road(event.payload["a"], event.payload["b"])
            if road and not road.blocked:
                road.blocked = True
                self._bump_road_version()
        elif event.kind == "road_unblocked":
            road = self.get_road(event.payload["a"], event.payload["b"])
            if road and road.blocked:
                road.blocked = False
                self._bump_road_version()
        elif event.kind == "civilian_found":
            civ = Civilian(
                id=event.payload["id"],
                location=event.payload["location"],
                needs_medical=event.payload.get("needs_medical", False),
                needs_evacuation=event.payload.get("needs_evacuation", False),
            )
            self.add_civilian(civ)
        elif event.kind == "resource_consumed":
            resource = event.payload["resource"]
            if isinstance(resource, str):
                resource = ResourceType(resource)
            self.locations[event.payload["location"]].remove_resource(
                resource,
                event.payload.get("quantity", event.payload.get("qty", 0)),
            )
        elif event.kind == "resource_depleted":
            resource = event.payload["resource"]
            if isinstance(resource, str):
                resource = ResourceType(resource)
            self.locations[event.payload["location"]].remove_resource(
                resource, event.payload["quantity"]
            )
        elif event.kind == "fuel_low":
            agent = self.agents.get(event.payload["agent_id"])
            if agent is not None:
                agent.fuel = min(agent.fuel, event.payload["fuel"])
        elif event.kind == "road_risk_increased":
            road = self.get_road(event.payload["a"], event.payload["b"])
            if road is not None and event.payload["risk"] > road.risk:
                road.risk = event.payload["risk"]
                self._bump_road_version()
        elif event.kind == "civilian_condition_changed":
            civilian = self.civilians.get(event.payload["id"])
            if civilian is not None:
                if "needs_medical" in event.payload:
                    civilian.needs_medical = event.payload["needs_medical"]
                if "needs_evacuation" in event.payload:
                    civilian.needs_evacuation = event.payload["needs_evacuation"]
        elif event.kind == "task_added":
            p = event.payload
            task = Task(
                id=p["id"],
                type=TaskType(p["type"]),
                target=p["target"],
                resource=(ResourceType(p["resource"])
                          if p.get("resource") else None),
                quantity=p.get("quantity", 0),
                description=p.get("description", ""),
                deadline=p.get("deadline"),
            )
            self.add_task(task)

        self.events.append(event)
        for listener in self._event_listeners:
            listener(self, event)