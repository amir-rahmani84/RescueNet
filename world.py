# --- new import at the top of world.py ---
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from enums import ResourceType, TaskStatus, TaskType
from events import EventScheduler
from models import Agent, Civilian, Event, Location, Road, Task


# Type alias for listeners
EventListener = Callable[["World", Event], None]


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

        self.agents: Dict[str, Agent] = {}

        self._scheduled_events: List[Event] = []
        self._scheduler: Optional[EventScheduler] = scheduler
        self._event_listeners: List[EventListener] = []

    # ------------------------------------------------------------------ #
    #  World state API                                                    #
    # ------------------------------------------------------------------ #

    def add_location(self, location: Location) -> None:
        self.locations[location.name] = location

    def add_road(self, road: Road) -> None:
        if road.a not in self.locations or road.b not in self.locations:
            raise ValueError("Both road endpoints must be known locations")
        self.roads.append(road)

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

    def neighbors(self, location: str) -> Iterable[Tuple[str, Road]]:
        for road in self.roads:
            if road.a == location:
                yield road.b, road
            elif road.b == location:
                yield road.a, road

    def get_road(self, a: str, b: str) -> Optional[Road]:
        return next((road for road in self.roads if road.connects(a, b)), None)

    def block_road(self, a: str, b: str) -> None:
        road = self.get_road(a, b)
        if road is None:
            raise ValueError(f"Unknown road: {a}-{b}")
        road.blocked = True

    def unblock_road(self, a: str, b: str) -> None:
        road = self.get_road(a, b)
        if road is None:
            raise ValueError(f"Unknown road: {a}-{b}")
        road.blocked = False

    def advance_time(self, ticks: int = 1) -> None:
        if ticks < 0:
            raise ValueError("ticks cannot be negative")
        for _ in range(ticks):
            self.time += 1
            self._fire_due_events()
            self._fail_expired_tasks()

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

    def execute_action(self, agent_id: str, action) -> None:
        """Execute a bound action against authoritative world state."""
        from perception import perceive_agent
        from reasoner import forward_chain

        agent = self.agents[agent_id]
        perceive_agent(agent, self)
        # Derived preconditions (can_treat, can_reach, can_clear_road, ...)
        # only become visible after forward chaining, so run it before the
        # precondition check.
        forward_chain(agent.knowledge)
        if not action.preconditions_hold(agent.knowledge):
            raise ValueError(f"Action preconditions failed: {action}")
        name = action.action.name
        bindings = action.bindings
        if name == "Move":
            self.move_agent(agent_id, bindings["?to"])
        elif name == "PickUp":
            self.pick_up(agent_id, bindings["?loc"], ResourceType(bindings["?resource"]))
        elif name == "Deliver":
            self.deliver(agent_id, bindings["?loc"], ResourceType(bindings["?resource"]))
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
        if name != "Move":
            self.advance_time(1)
        self._complete_agent_task(agent)

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

    def _fail_expired_tasks(self) -> None:
        for task in self.tasks.values():
            if (task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
                    and task.deadline is not None
                    and self.time > task.deadline):
                self.fail_task(task.id, "task deadline expired")

    # ------------------------------------------------------------------ #
    #  Dynamic event support                                             #
    # ------------------------------------------------------------------ #

    def set_event_scheduler(self, scheduler: EventScheduler) -> None:
        """Attach a scheduler for stochastic / randomised events."""
        self._scheduler = scheduler

    def add_event_listener(self, callback: EventListener) -> None:
        """Register a callback invoked immediately after an event is applied."""
        self._event_listeners.append(callback)

    # ------------------------------------------------------------------ #
    #  Event scheduling (existing signature, scheduler-aware)            #
    # ------------------------------------------------------------------ #

    def schedule_event(self, event: Event) -> None:
        """Queue an event to fire at a specific simulation time."""
        self._scheduled_events.append(event)

    def _fire_due_events(self) -> None:
        due = [e for e in self._scheduled_events if e.time <= self.time]
        self._scheduled_events = [
            e for e in self._scheduled_events if e.time > self.time
        ]
        if self._scheduler is not None:
            due.extend(self._scheduler.due(self.time))
        for event in due:
            self._apply_event(event)

    # ------------------------------------------------------------------ #
    #  Event application + task_added + listener notification            #
    # ------------------------------------------------------------------ #

    def _apply_event(self, event: Event) -> None:
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
        elif event.kind == "deadline_shortened":
            self.deadline = min(self.deadline, event.payload["deadline"])
        elif event.kind == "road_risk_increased":
            road = self.get_road(event.payload["a"], event.payload["b"])
            if road is not None:
                road.risk = max(road.risk, event.payload["risk"])
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

        # Log it, then notify listeners (so agents learn *immediately*).
        self.events.append(event)
        for listener in self._event_listeners:
            listener(self, event)