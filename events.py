"""Canonical event vocabulary and constructors for RescueNet.

This module owns event creation and payload shape.  ``World`` owns applying
events to mutable state, so scheduled and immediately recorded events use the
same representation.
"""

from __future__ import annotations

from enums import ResourceType
from models import Event


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