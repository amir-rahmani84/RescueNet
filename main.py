"""Driver for Steps 1-5 of RescueNet.

Step 1 -- the mechanical world (map, roads, resources, tasks, events).
Step 2 -- three agents that can move around it.
Step 3 -- each agent carries a knowledge base populated from the world.
Step 4 -- a forward-chaining reasoner that derives new facts from rules.
Step 5 -- a symbolic action representation (preconditions / effects / cost).

Nothing here is autonomous yet: agents still do exactly what we tell them.
The point is to show that the substrate the later steps will build on
actually works -- and that the reasoner can now hand the action layer
information the agent was never explicitly told.
"""

from __future__ import annotations

from actions import (
    DEFAULT_ACTIONS,
    MOVE,
    applicable_actions,
)
from display import render_agents, render_knowledge, render_world
from models import Event
from reasoner import explain, forward_chain
from scenario import build_default_world


BANNER = "=" * 64


def section(title: str) -> None:
    print()
    print(BANNER)
    print(title)
    print(BANNER)


def main() -> None:
    world = build_default_world()
    world.schedule_event(Event(time=6, kind="road_blocked",
                               payload={"a": "E", "b": "G"}))
    world.schedule_event(Event(time=10, kind="civilian_found",
                               payload={"id": "civ3", "location": "C",
                                        "needs_medical": True}))

    # ---------------------------------------------------------------- #
    #  Step 1 -- the world                                             #
    # ---------------------------------------------------------------- #
    section("STEP 1 -- the world (mechanics only)")
    print(render_world(world))
    world.advance_time(5)
    print(f"\n>> clock advanced to t={world.time}")

    # ---------------------------------------------------------------- #
    #  Step 2 -- agents move by hand                                   #
    # ---------------------------------------------------------------- #
    section("STEP 2 -- agents (no intelligence yet)")
    world.move_agent("Rescue-1", "F")
    world.move_agent("Logistics-1", "B")
    world.move_agent("Logistics-1", "E")
    print(render_agents(world))
    print(f"\n>> clock is now t={world.time}; the t=6 road-block fired mid-move")

    # ---------------------------------------------------------------- #
    #  Step 3 -- perception                                            #
    # ---------------------------------------------------------------- #
    section("STEP 3 -- every agent perceives the world as facts")
    for agent in world.agents.values():
        agent.perceive(world)
    print(render_knowledge(world.agents["Rescue-1"]))

    # ---------------------------------------------------------------- #
    #  Step 4 -- forward chaining                                      #
    # ---------------------------------------------------------------- #
    section("STEP 4 -- forward-chaining reasoner")
    derived_by_agent = {}
    for agent in world.agents.values():
        derived = forward_chain(agent.knowledge)
        derived_by_agent[agent.id] = derived
        print(f"\n--- {agent.id}: derived {len(derived)} new facts ---")
        print(explain(derived))

    print("\n>> Facts Rescue-1 now believes but was never explicitly told:")
    for predicate in ("can_reach", "can_treat", "can_assign", "task_feasible"):
        print(f"  {predicate}:")
        print(explain(derived_by_agent["Rescue-1"], predicate=predicate))

    # ---------------------------------------------------------------- #
    #  Step 5 -- actions                                               #
    # ---------------------------------------------------------------- #
    section("STEP 5 -- actions: preconditions, effects, cost")
    kb = world.agents["Rescue-1"].knowledge

    print(f"Applicable actions for Rescue-1 (schemas: "
          f"{', '.join(a.name for a in DEFAULT_ACTIONS)}):")
    for ba in applicable_actions(kb):
        print(f"    {str(ba):<40}  cost={ba.cost(kb):.2f}")

    print("\n>> Bind Move(Rescue-1, F, E) explicitly and apply it to the KB")
    move = MOVE.bind(**{"?agent": "Rescue-1", "?from": "F", "?to": "E"})
    print(f"    action:             {move}")
    print(f"    preconditions hold? {move.preconditions_hold(kb)}")
    print(f"    cost:               {move.cost(kb):.2f}")
    print(f"    at(Rescue-1, F) before: "
          f"{kb.contains(('at', ('Rescue-1', 'F')))}")
    move.apply(kb)
    print(f"    at(Rescue-1, F) after:  "
          f"{kb.contains(('at', ('Rescue-1', 'F')))}")
    print(f"    at(Rescue-1, E) after:  "
          f"{kb.contains(('at', ('Rescue-1', 'E')))}")

    print("\n>> Re-run forward chaining after the Move -- new reachability")
    print(explain(forward_chain(kb), predicate="can_reach"))

    print("\n>> An action whose preconditions *fail*: Move(Rescue-1, E, G)")
    print("   (E-G was blocked by the t=6 event; road_open(E,G) is absent)")
    blocked = MOVE.bind(**{"?agent": "Rescue-1", "?from": "E", "?to": "G"})
    print(f"    at(Rescue-1, E) present?  "
          f"{kb.contains(('at', ('Rescue-1', 'E')))}")
    print(f"    road_open(E, G) present?  "
          f"{kb.contains(('road_open', ('E', 'G')))}")
    print(f"    preconditions hold?       "
          f"{blocked.preconditions_hold(kb)}")

    print("\n>> Note: numeric effects (e.g. Refuel: fuel += n) do not fit")
    print("   the pure add/delete fact model and are left for a later step.")


if __name__ == "__main__":
    main()