"""Terminal rendering for the world."""

from __future__ import annotations

from typing import Dict, List, Optional

from enums import TaskStatus
from models import Agent
from world import World
from communication import CommunicationHub


_STATUS_GLYPH = {
    TaskStatus.COMPLETED: "✓",
    TaskStatus.IN_PROGRESS: "~",
    TaskStatus.PENDING: "!",
    TaskStatus.FAILED: "✗",
}


def render_map(world: World) -> str:
    """Hand-drawn ASCII map matching the scenario layout.

    In a later step we can replace this with an auto-layout, but a fixed
    drawing is clearer for a 7-node scenario.
    """
    lines = []
    lines.append("    A ---- B ---- C")
    lines.append("    |      |      |")
    lines.append("    D ---- E ---- F")
    lines.append("           |")
    lines.append("           G")
    return "\n".join(lines)


def render_tasks(world: World) -> str:
    lines = []
    for task in world.tasks.values():
        glyph = _STATUS_GLYPH.get(task.status, "?")
        lines.append(f"    {glyph} {task.description}")
    return "\n".join(lines) if lines else "    (none)"


def render_agents(world: World) -> str:
    """Show each agent's identity, position, capability, fuel and load."""
    if not world.agents:
        return "    (none)"

    lines: List[str] = []
    for agent in world.agents.values():
        caps = ", ".join(sorted(c.value for c in agent.capabilities)) or "-"
        lines.append(f"    {agent.id}")
        lines.append(f"        location:     {agent.location}")
        lines.append(f"        capabilities: {caps}")
        lines.append(f"        fuel:         {agent.fuel}")
        lines.append(f"        load:         {agent.load()} / {agent.capacity}")
        if agent.inventory:
            items = ", ".join(
                f"{res.value} x{qty}"
                for res, qty in sorted(agent.inventory.items(),
                                       key=lambda kv: kv[0].value)
            )
            lines.append(f"        inventory:    {items}")
        if agent.goal:
            lines.append(f"        goal:         {agent.goal}")
        if agent.plan:
            lines.append(f"        plan:         {' -> '.join(map(str, agent.plan))}")
    return "\n".join(lines)


def render_knowledge(
    agent: Agent,
    predicates: Optional[List[str]] = None,
) -> str:
    """Dump an agent's knowledge base, grouped by predicate.

    ``predicates`` optionally filters to a subset, e.g.
    ``["at", "can_reach", "can_assign"]``.
    """
    kb = agent.knowledge

    grouped: Dict[str, List[str]] = {}
    for fact in kb.facts():
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


def render_events(world: World, last_n: int = 5) -> str:
    recent = world.events[-last_n:]
    if not recent:
        return "    (no events yet)"
    lines = []
    for e in recent:
        lines.append(f"    [t={e.time}] {e.kind}: {e.payload}")
    return "\n".join(lines)


def render_messages(hub: CommunicationHub, last_n: int = 10) -> str:
    recent = hub.messages[-last_n:]
    if not recent:
        return "    (no messages yet)"
    lines = []
    for message in recent:
        lines.append(
            f"    [t={message.time}] {message.sender} -> {message.receiver} "
            f"[{message.type.value}] {message.content}"
        )
    return "\n".join(lines)


def render_world(world: World) -> str:
    parts = [
        "=" * 44,
        "RESCUENET",
        "=" * 44,
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
        "=" * 44,
    ]
    return "\n".join(parts)