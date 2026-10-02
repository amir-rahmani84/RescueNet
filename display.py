"""Terminal rendering for the world and mission (Step 15).

Design goal: make each agent's *internal state* legible.  Not just where
they are, but what they believe, what they want, and what they plan to do
next.  The terminal is the observability layer for the AI mechanics.

Public API kept compatible with the earlier version so `main.py` and
`simulation.py` keep working without changes:

    render_map, render_tasks, render_agents, render_knowledge,
    render_events, render_messages, render_world

New in Step 15:

    render_dashboard(world, hub)   -- one-shot full mission view
    render_mission_summary(world, result)  -- final scoreboard
"""

from __future__ import annotations

from typing import Dict, List, Optional

from communication import CommunicationHub, Message, MessageType
from enums import TaskStatus
from models import Agent, Event
from world import World


_STATUS_GLYPH = {
    TaskStatus.COMPLETED: "✓",
    TaskStatus.IN_PROGRESS: "~",
    TaskStatus.PENDING: "!",
    TaskStatus.FAILED: "✗",
}

_MESSAGE_TAG = {
    MessageType.REQUEST: "REQ",
    MessageType.INFORM: "INF",
    MessageType.OFFER: "OFF",
    MessageType.ACCEPT: "ACC",
    MessageType.REJECT: "REJ",
    MessageType.CLAIM: "CLM",
    MessageType.WARNING: "WRN",
}

RULE = "=" * 64
THIN = "-" * 64


# --------------------------------------------------------------------------- #
#  Map and tasks
# --------------------------------------------------------------------------- #

def render_map(world: World) -> str:
    """Hand-drawn ASCII map matching the scenario layout."""
    lines = [
        "    A ---- B ---- C",
        "    |      |      |",
        "    D ---- E ---- F",
        "           |",
        "           G",
    ]
    return "\n".join(lines)


def render_tasks(world: World) -> str:
    if not world.tasks:
        return "    (none)"
    lines: List[str] = []
    for task in world.tasks.values():
        glyph = _STATUS_GLYPH.get(task.status, "?")
        owner = f"  <- {task.assigned_to}" if task.assigned_to else ""
        lines.append(f"    {glyph} {task.description}{owner}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Agents
# --------------------------------------------------------------------------- #

def _capabilities(agent: Agent) -> str:
    return ", ".join(sorted(c.value for c in agent.capabilities)) or "-"


def _inventory(agent: Agent) -> str:
    if not agent.inventory:
        return "-"
    return ", ".join(
        f"{res.value} x{qty}"
        for res, qty in sorted(agent.inventory.items(),
                               key=lambda kv: kv[0].value)
    )


def render_agents(world: World) -> str:
    if not world.agents:
        return "    (none)"
    blocks: List[str] = []
    for agent in world.agents.values():
        lines = [
            f"    {agent.id}",
            f"        location:     {agent.location}",
            f"        capabilities: {_capabilities(agent)}",
            f"        fuel:         {agent.fuel}",
            f"        load:         {agent.load()} / {agent.capacity}",
        ]
        inv = _inventory(agent)
        if inv != "-":
            lines.append(f"        inventory:    {inv}")
        if agent.goal:
            lines.append(f"        goal:         {agent.goal}")
        if agent.plan:
            plan_str = " -> ".join(str(a) for a in agent.plan)
            lines.append(f"        plan:         {plan_str}")
            lines.append(
                f"        progress:     step "
                f"{agent.plan_index}/{len(agent.plan)}"
            )
        elif agent.goal:
            lines.append("        plan:         (none -- waiting for a plan)")
        if agent.plan_reason:
            lines.append(f"        last reason:  {agent.plan_reason}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def render_knowledge(
    agent: Agent,
    predicates: Optional[List[str]] = None,
) -> str:
    """Dump an agent's KB, grouped by predicate, optionally filtered."""
    kb = agent.knowledge
    grouped: Dict[str, List[str]] = {}
    for fact in kb.sorted_facts():
        if predicates is not None and fact.predicate not in predicates:
            continue
        arg_text = ", ".join(str(a) for a in fact.args) if fact.args else ""
        grouped.setdefault(fact.predicate, []).append(arg_text)

    lines = [f"KNOWLEDGE — {agent.id}  ({len(kb)} facts)"]
    if not grouped:
        lines.append("    (empty)")
        return "\n".join(lines)

    for predicate in sorted(grouped):
        lines.append(f"    {predicate}:")
        for arg_text in grouped[predicate]:
            lines.append(f"        {arg_text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Events
# --------------------------------------------------------------------------- #

_EVENT_FORMATTERS = {
    "road_blocked": lambda p: f"Road {p['a']} -> {p['b']} has become blocked.",
    "road_unblocked": lambda p: f"Road {p['a']} -> {p['b']} has reopened.",
    "civilian_found": lambda p: (
        f"New civilian {p['id']} at {p['location']}"
        + (" (needs medical)" if p.get("needs_medical") else "")
        + (" (needs evacuation)" if p.get("needs_evacuation") else "")
    ),
    "fuel_low": lambda p: f"{p['agent_id']} is low on fuel ({p['fuel']}).",
    "road_risk_increased": lambda p: (
        f"Road {p['a']} -> {p['b']} risk now {p['risk']}."
    ),
    "resource_consumed": lambda p: (
        f"{p['quantity']} x {p['resource']} consumed at {p['location']}."
    ),
    "resource_depleted": lambda p: (
        f"{p['resource']} depleted at {p['location']}."
    ),
    "civilian_condition_changed": lambda p: (
        f"Condition of {p['id']} changed."
    ),
    "task_added": lambda p: (
        f"New task {p['id']} ({p.get('type')} -> {p.get('target')})"
    ),
}


def _format_event(event: Event) -> str:
    fmt = _EVENT_FORMATTERS.get(event.kind)
    if fmt is None:
        return f"{event.kind}: {event.payload}"
    try:
        return fmt(event.payload)
    except (KeyError, TypeError):
        return f"{event.kind}: {event.payload}"


def render_events(world: World, last_n: int = 5) -> str:
    recent = world.events[-last_n:]
    if not recent:
        return "    (no events yet)"
    lines: List[str] = []
    for event in recent:
        lines.append(f"    [t={event.time}] {_format_event(event)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Communication
# --------------------------------------------------------------------------- #

def render_messages(hub: CommunicationHub, last_n: int = 10) -> str:
    recent = hub.messages[-last_n:]
    if not recent:
        return "    (no messages yet)"
    lines: List[str] = []
    for message in recent:
        tag = _MESSAGE_TAG.get(
            message.type, message.type.value[:3].upper()
        )
        lines.append(
            f"    [t={message.time}] [{tag}] "
            f"{message.sender} -> {message.receiver}"
        )
        lines.append(f"        {message.content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Composed views
# --------------------------------------------------------------------------- #

def render_world(world: World) -> str:
    parts = [
        RULE,
        "RESCUENET",
        RULE,
        f"Time: {world.time} / {world.deadline}",
        "",
        "WORLD",
        render_map(world),
        "",
        "TASKS",
        render_tasks(world),
        "",
        "AGENTS",
        render_agents(world),
        "",
        "EVENTS",
        render_events(world),
        RULE,
    ]
    return "\n".join(parts)


def render_dashboard(
    world: World,
    hub: Optional[CommunicationHub] = None,
    *,
    last_events: int = 5,
    last_messages: int = 10,
) -> str:
    """One-shot view of the whole mission state.

    Combines world, tasks, agents, events, and (optionally) the
    communication log into a single string.  This is the canonical
    "mission view" used by the Step 17 integrated demo.
    """
    parts: List[str] = [
        RULE,
        "RESCUENET -- MISSION DASHBOARD",
        RULE,
        f"Time: {world.time} / {world.deadline}",
        "",
        "WORLD",
        render_map(world),
        "",
        "TASKS",
        render_tasks(world),
        "",
        "AGENTS",
        render_agents(world),
        "",
        "EVENTS",
        render_events(world, last_n=last_events),
    ]
    if hub is not None:
        parts.extend([
            "",
            "COMMUNICATION",
            render_messages(hub, last_n=last_messages),
        ])
    parts.append(RULE)
    return "\n".join(parts)


def render_mission_summary(world: World, result) -> str:
    """Final scoreboard for the mission (Step 17)."""
    parts: List[str] = [
        RULE,
        "MISSION SUMMARY",
        RULE,
        f"Time reached:  {world.time} / {world.deadline}",
        f"Iterations:    {result.iterations}",
        f"Final score:   {result.score:.2f}",
        "",
    ]
    if result.completed:
        parts.append("Completed tasks:")
        for task_id in result.completed:
            parts.append(f"    ✓ {task_id}")
    if result.failed:
        parts.append("Failed tasks:")
        for task_id in result.failed:
            parts.append(f"    ✗ {task_id}")
    if result.pending:
        parts.append("Still pending:")
        for task_id in result.pending:
            parts.append(f"    ! {task_id}")
    if not (result.completed or result.failed or result.pending):
        parts.append("    (no tasks recorded)")
    parts.append(RULE)
    return "\n".join(parts)