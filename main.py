"""Driver for Steps 1–3.

Step 1 -- the mechanical world (map, roads, resources, tasks, events).
Step 2 -- three agents that can move around it.
Step 3 -- each agent carries a knowledge base populated from the world.

Nothing here is intelligent yet: agents are moved by hand, and rules are
only *checked* against the knowledge base (the fixpoint engine that chains
them together is Step 4).
"""

from __future__ import annotations

from display import render_agents, render_knowledge, render_world
from models import Event
from rules import DEFAULT_RULES, instantiate
from scenario import build_default_world


def main() -> None:
    world = build_default_world()

    # Schedule a couple of future events so we can watch them fire.
    world.schedule_event(Event(time=6, kind="road_blocked",
                               payload={"a": "E", "b": "G"}))
    world.schedule_event(Event(time=10, kind="civilian_found",
                               payload={"id": "civ3", "location": "C",
                                        "needs_medical": True}))

    print(render_world(world))

    # ------------------------------------------------------------------ #
    #  Step 1 -- mechanical world
    # ------------------------------------------------------------------ #
    print("\n>> Advance time to 5")
    world.advance_time(5)
    print(render_world(world))

    # ------------------------------------------------------------------ #
    #  Step 2 -- agents move around manually
    # ------------------------------------------------------------------ #
    print("\n>> Rescue-1 moves E -> F")
    print(f"   move succeeded: {world.move_agent('Rescue-1', 'F')}")
    print(render_agents(world))

    print("\n>> Logistics-1 moves A -> B -> E")
    world.move_agent("Logistics-1", "B")
    world.move_agent("Logistics-1", "E")
    print(render_agents(world))

    print("\n>> Engineer-1 tries B -> C while the road is blocked")
    world.block_road("B", "C")
    print(f"   move succeeded: {world.move_agent('Engineer-1', 'C')}")
    world.unblock_road("B", "C")
    print(render_agents(world))

    # ------------------------------------------------------------------ #
    #  Step 3 -- knowledge representation
    # ------------------------------------------------------------------ #
    print(f"\n(clock is now t={world.time}; "
          "the t=6 and t=10 events have fired)")
    print(render_world(world))

    for agent in world.agents.values():
        agent.perceive(world)

    print()
    print(render_knowledge(world.agents["Rescue-1"]))

    print("\n>> Rules whose premises hold in Rescue-1's knowledge")
    print("   (one round only -- Step 4 adds the chaining loop)")
    kb = world.agents["Rescue-1"].knowledge
    for rule in DEFAULT_RULES:
        for bindings in rule.fires(kb):
            conclusion = instantiate(rule.conclusion, bindings)
            print(f"    [{rule.name}]  ->  {conclusion}")


if __name__ == "__main__":
    main()