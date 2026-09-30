"""Driver for Steps 1-7 of RescueNet.

Step 1 -- the mechanical world (map, roads, resources, tasks, events).
Step 2 -- three agents that can move around it.
Step 3 -- each agent carries a knowledge base populated from the world.
Step 4 -- a forward-chaining reasoner that derives new facts from rules.
Step 5 -- a symbolic action representation (preconditions / effects / cost).
Step 6 -- an A* planner that searches over possible states of the world.
Step 7 -- a backtracking CSP that assigns pending tasks to agents.
"""

from __future__ import annotations

from actions import (
    DEFAULT_ACTIONS,
    MOVE,
    applicable_actions,
)
from constraints import solve_task_assignment
from display import render_agents, render_knowledge, render_world
from models import Event
from planner import location_heuristic, plan, plan_cost
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

    # ---------------------------------------------------------------- #
    #  Step 6 -- A* planner                                            #
    # ---------------------------------------------------------------- #
    section("STEP 6 -- A* planner: search over possible world states")

    rescue_kb = world.agents["Rescue-1"].knowledge

    print(">> Situation (from Rescue-1's KB):")
    print("     at(Rescue-1, E)")
    print("     has(Rescue-1, medical_kit)")
    print("     person_at(civ1, F),  needs_medical(civ1)")
    print("     road_open(E, F) is true; E-G is blocked.")
    print()
    print(">> Goal: derive a plan that makes treated(civ1) hold.")
    print("   (The planner must *sequence* Move then Treat -- Treat alone")
    print("    is not applicable because can_treat is not yet derivable.)")

    goal_treat = lambda k: k.contains(("treated", ("civ1",)))
    h_treat = location_heuristic("Rescue-1", "F")

    plan_treat = plan(rescue_kb, goal_treat, heuristic=h_treat)
    if plan_treat is None:
        print("    (no plan found)")
    else:
        print(f"\n    plan ({len(plan_treat)} steps):")
        for step in plan_treat:
            print(f"        {str(step):<38}  cost={step.cost(rescue_kb):.2f}")
        print(f"    {'TOTAL':<42}  cost={plan_cost(plan_treat, rescue_kb):.2f}")

    print()
    print(">> Cross-check: is Logistics-1 able to reach G?")
    print("   Road E-G is blocked, Logistics-1 has no repair capability,")
    print("   so the planner should give up.")
    log_kb = world.agents["Logistics-1"].knowledge
    goal_g = lambda k: k.contains(("at", ("Logistics-1", "G")))
    h_g = location_heuristic("Logistics-1", "G")
    plan_g = plan(log_kb, goal_g, heuristic=h_g, max_expansions=3000)
    if plan_g is None:
        print("    planner returned: None (goal unreachable)")
    else:
        print(f"    planner returned: {[str(s) for s in plan_g]}")

    # ---------------------------------------------------------------- #
    #  Step 7 -- CSP: assign tasks to agents                           #
    # ---------------------------------------------------------------- #
    section("STEP 7 -- CSP: assign pending tasks to agents")

    # Refresh each agent's KB so can_assign reflects the current world.
    for agent in world.agents.values():
        agent.perceive(world)
        forward_chain(agent.knowledge)

    print(">> Variables   : pending tasks")
    print("   Domains     : agents with can_assign(agent, task) in their KB")
    print("                 (this already encodes capability + reachability)")
    print("   Constraints : per-agent load cap (<= 2) and deadline feasibility")
    print()

    result = solve_task_assignment(world, world.agents, max_tasks_per_agent=2)

    print(">> Assignment:")
    for tid in sorted(result.assignment):
        task = world.tasks[tid]
        print(f"    {tid:<26} -> {result.assignment[tid]:<14}  "
              f"({task.description})")

    if result.unassigned:
        print("\n>> Unassigned (no valid agent under current constraints):")
        for tid in result.unassigned:
            task = world.tasks[tid]
            print(f"    {tid:<26}    -              ({task.description})")

    print(f"\n   backtracks: {result.backtracks}")

    print()
    print(">> Same problem, but with max_tasks_per_agent = 1:")
    strict = solve_task_assignment(world, world.agents, max_tasks_per_agent=1)
    for tid in sorted(strict.assignment):
        print(f"    {tid:<26} -> {strict.assignment[tid]}")
    if strict.unassigned:
        print("   left unassigned:")
        for tid in strict.unassigned:
            print(f"    {tid}")
    print(f"   backtracks: {strict.backtracks}")


if __name__ == "__main__":
    main()