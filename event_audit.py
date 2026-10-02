"""Event-vocabulary audit for RescueNet.

Verifies that the project's event pipeline is internally consistent:

* every event kind the ``events`` module can actually construct
  (discovered by parsing its source AST, so a new factory is picked up
  automatically) is handled by ``World._apply_event`` and rendered by
  ``display._EVENT_FORMATTERS``;
* the declared contracts (``events.EVENT_KINDS``, ``world.HANDLED_EVENT_KINDS``)
  match that discovered reality;
* every handler / formatter refers to a kind that can actually fire
  (no dead code, no orphan formatter).

Library use::

    from event_audit import audit_event_coverage
    print(audit_event_coverage(world).render())

CLI::

    python -m event_audit
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set

import events as events_module
from display import _EVENT_FORMATTERS
from events import EVENT_KINDS
from world import HANDLED_EVENT_KINDS, World


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #

def _discover_emitted_kinds() -> Set[str]:
    """Extract every string literal passed as the 2nd arg to ``Event(...)``
    inside a top-level function of ``events.py``."""
    source = inspect.getsource(events_module)
    tree = ast.parse(source)
    kinds: Set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name != "Event":
                continue
            if len(sub.args) < 2:
                continue
            arg = sub.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                kinds.add(arg.value)
    return kinds


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EventAudit:
    declared: FrozenSet[str]
    discovered: FrozenSet[str]
    handled: FrozenSet[str]
    formatted: FrozenSet[str]
    fired: Optional[FrozenSet[str]] = None    # kinds seen in a live world

    # -- discrepancies ----------------------------------------------------- #

    @property
    def declared_but_not_discovered(self) -> List[str]:
        return sorted(self.declared - self.discovered)

    @property
    def discovered_but_not_declared(self) -> List[str]:
        return sorted(self.discovered - self.declared)

    @property
    def emitted_but_unhandled(self) -> List[str]:
        return sorted(self.discovered - self.handled)

    @property
    def handled_but_not_emitted(self) -> List[str]:
        return sorted(self.handled - self.discovered)

    @property
    def emitted_but_unformatted(self) -> List[str]:
        return sorted(self.discovered - self.formatted)

    @property
    def formatted_but_not_emitted(self) -> List[str]:
        return sorted(self.formatted - self.discovered)

    @property
    def never_fired(self) -> List[str]:
        if self.fired is None:
            return []
        return sorted(self.discovered - self.fired)

    @property
    def is_healthy(self) -> bool:
        return not (
            self.declared_but_not_discovered
            or self.discovered_but_not_declared
            or self.emitted_but_unhandled
            or self.handled_but_not_emitted
            or self.emitted_but_unformatted
            or self.formatted_but_not_emitted
        )

    # -- presentation ------------------------------------------------------ #

    def render(self) -> str:
        rule = "=" * 64
        lines: List[str] = [rule, "EVENT VOCABULARY AUDIT", rule]

        lines.append(f"  declared   : {len(self.declared)}")
        lines.append(f"  discovered : {len(self.discovered)}  (from events.py AST)")
        lines.append(f"  handled    : {len(self.handled)}")
        lines.append(f"  formatted  : {len(self.formatted)}")
        if self.fired is not None:
            lines.append(f"  fired (this run): {len(self.fired)}")
        lines.append("")

        def section(title: str, items: List[str]) -> None:
            lines.append(title)
            if not items:
                lines.append("    (none)")
            else:
                for name in items:
                    lines.append(f"    !! {name}")
            lines.append("")

        section(
            ">> Declared in EVENT_KINDS but no factory emits it",
            self.declared_but_not_discovered,
        )
        section(
            ">> A factory emits it but EVENT_KINDS does not declare it",
            self.discovered_but_not_declared,
        )
        section(
            ">> Emitted but NOT handled by World (silent drop risk)",
            self.emitted_but_unhandled,
        )
        section(
            ">> Handled by World but no factory emits it (dead branch)",
            self.handled_but_not_emitted,
        )
        section(
            ">> Emitted but NOT formatted in display (invisible in UI)",
            self.emitted_but_unformatted,
        )
        section(
            ">> Formatted in display but no factory emits it (orphan)",
            self.formatted_but_not_emitted,
        )
        if self.fired is not None:
            section(
                ">> Available but never fired in the observed world",
                self.never_fired,
            )

        lines.append("-" * 64)
        if self.is_healthy:
            lines.append("RESULT: event vocabulary is fully consistent.")
        else:
            lines.append("RESULT: inconsistencies detected (see above).")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def audit_event_coverage(world: Optional[World] = None) -> EventAudit:
    """Build an ``EventAudit``.

    If ``world`` is given, the report also lists kinds that are *available*
    (declared + handled) but never actually fired in that world's history.
    """
    fired: Optional[FrozenSet[str]] = None
    if world is not None:
        fired = frozenset(e.kind for e in world.events)

    return EventAudit(
        declared=frozenset(EVENT_KINDS),
        discovered=frozenset(_discover_emitted_kinds()),
        handled=frozenset(HANDLED_EVENT_KINDS),
        formatted=frozenset(_EVENT_FORMATTERS),
        fired=fired,
    )


def _main() -> None:
    print(audit_event_coverage().render())


if __name__ == "__main__":
    _main()