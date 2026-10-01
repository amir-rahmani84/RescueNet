"""Canonical event vocabulary and constructors for RescueNet.

Also hosts ``EventScheduler``: the object that decides *when* events fire.
Two flavours are supported:

* **Scheduled** events fire exactly at their recorded time.
* **Stochastic** events are described by a probability plus a factory; on
  every world tick each stochastic entry is rolled, and on success the
  factory produces an Event that fires immediately.  The RNG is seeded so
  a mission is reproducible.

This module owns event creation and payload shape.  ``World`` owns applying
events to mutable state, so scheduled and immediately recorded events use the
same representation.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

from enums import ResourceType, TaskType
from models import Event


# --------------------------------------------------------------------------- #
#  Canonical event constructors
# --------------------------------------------------------------------------- #

def road_blocked(time: int, a: str, b: str) -> Event:
    return Event(time, "road_blocked", {"a": a, "b": b})


def road_unblocked(time: int, a: str, b: str) -> Event:
    return Event(time, "road_unblocked", {"a": a, "b": b})


def civilian_found(
    time: int,
    civilian_id: str,
    location: str,
    *,
    needs_medical: bool = False,
    needs_evacuation: bool = False,
) -> Event:
    return Event(
        time,
        "civilian_found",
        {
            "id": civilian_id,
            "location": location,
            "needs_medical": needs_medical,
            "needs_evacuation": needs_evacuation,
        },
    )


def resource_consumed(
    time: int, location: str, resource: ResourceType, quantity: int
) -> Event:
    return Event(
        time,
        "resource_consumed",
        {"location": location, "resource": resource.value, "quantity": quantity},
    )


def fuel_low(time: int, agent_id: str, fuel: int) -> Event:
    return Event(time, "fuel_low", {"agent_id": agent_id, "fuel": fuel})


def deadline_shortened(time: int, deadline: int) -> Event:
    return Event(time, "deadline_shortened", {"deadline": deadline})


def resource_depleted(
    time: int, location: str, resource: ResourceType, quantity: int
) -> Event:
    return Event(
        time,
        "resource_depleted",
        {"location": location, "resource": resource.value, "quantity": quantity},
    )


def road_risk_increased(time: int, a: str, b: str, risk: float) -> Event:
    return Event(time, "road_risk_increased", {"a": a, "b": b, "risk": risk})


def civilian_condition_changed(
    time: int,
    civilian_id: str,
    *,
    needs_medical: bool | None = None,
    needs_evacuation: bool | None = None,
) -> Event:
    payload = {"id": civilian_id}
    if needs_medical is not None:
        payload["needs_medical"] = needs_medical
    if needs_evacuation is not None:
        payload["needs_evacuation"] = needs_evacuation
    return Event(time, "civilian_condition_changed", payload)


def task_added(
    time: int,
    task_id: str,
    task_type: TaskType,
    target: str,
    *,
    resource: Optional[ResourceType] = None,
    quantity: int = 0,
    description: str = "",
    deadline: Optional[int] = None,
) -> Event:
    """A new mission objective appears mid-simulation."""
    return Event(
        time,
        "task_added",
        {
            "id": task_id,
            "type": task_type.value,
            "target": target,
            "resource": resource.value if resource is not None else None,
            "quantity": quantity,
            "description": description,
            "deadline": deadline,
        },
    )


# --------------------------------------------------------------------------- #
#  EventScheduler
# --------------------------------------------------------------------------- #

# ``factory(time)`` -- returns the Event to fire when the stochastic roll wins.
StochasticFactory = Callable[[int], Event]


class EventScheduler:
    """Owns the *timing* of a mission's dynamic events.

    The world holds a reference to a scheduler and asks it, once per tick,
    which events are due.  The scheduler never touches world state itself --
    it just decides what fires and when.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._scheduled: List[Event] = []
        self._stochastic: List[Tuple[float, StochasticFactory]] = []
        self._rng = random.Random(seed)

    # -- scheduling ----------------------------------------------------- #

    def schedule(self, event: Event) -> None:
        """Queue an event to fire at its recorded time."""
        self._scheduled.append(event)

    def add_stochastic(
        self,
        probability: float,
        factory: StochasticFactory,
    ) -> None:
        """Roll ``probability`` each tick; on success, ``factory`` produces an Event.

        ``probability`` must lie in [0, 1].  The same RNG is used for every
        stochastic entry, so a seeded scheduler is fully reproducible.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self._stochastic.append((probability, factory))

    # -- firing --------------------------------------------------------- #

    def due(self, time: int) -> List[Event]:
        """Return every event that should fire now, and forget it.

        Scheduled events fire as soon as the clock reaches (or passes) their
        time.  Stochastic entries are rolled independently for each tick.
        """
        due = [e for e in self._scheduled if e.time <= time]
        self._scheduled = [e for e in self._scheduled if e.time > time]
        for probability, factory in self._stochastic:
            if self._rng.random() < probability:
                due.append(factory(time))
        due.sort(key=lambda e: e.time)
        return due

    def pending(self) -> List[Event]:
        """Snapshot of currently-scheduled events (for display)."""
        return list(self._scheduled)