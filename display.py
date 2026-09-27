"""Terminal rendering for the world."""

from __future__ import annotations

from enums import TaskStatus
from world import World


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


def render_events(world: World, last_n: int = 5) -> str:
    recent = world.events[-last_n:]
    if not recent:
        return "    (no events yet)"
    lines = []
    for e in recent:
        lines.append(f"    [t={e.time}] {e.kind}: {e.payload}")
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
        "EVENTS",
        render_events(world),
        "=" * 44,
    ]
    return "\n".join(parts)