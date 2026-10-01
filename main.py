"""Terminal demonstration for the RescueNet autonomous mission system.

Step 1 -- the mechanical world (map, roads, resources, tasks, events).
Step 2 -- three agents that can move around it.
Step 3 -- each agent carries a knowledge base populated from the world.
Step 4 -- a forward-chaining reasoner that derives new facts from rules.
Step 5 -- a symbolic action representation (preconditions / effects / cost).
Step 6 -- an A* planner that searches over possible states of the world.
Step 7 -- a backtracking CSP that assigns pending tasks to agents.
Steps 8-9 -- utility optimization and individual goal evaluation.
Steps 10-11 -- agent claims, CSP assignment, and rejection feedback loop.
Advanced events -- release commitments and trigger a new assignment pass.
"""

from __future__ import annotations

from actions import (
    DEFAULT_ACTIONS,
    DELIVER,
    MOVE,
    PICK_UP,
    applicable_actions,
)
from constraints import solve_task_assignment
from communication import (
    CommunicationHub,
    negotiate_assignment,
    reassign_rejected,
    solicit_claims,
)
from decision import (
    mission_utility,
    optimize_assignment,
    rank_goals,
    select_goal,
)
from display import render_agents, render_knowledge, render_world
from display import render_events, render_messages
from events import (
    civilian_found,
    deadline_shortened,
    fuel_low,
    road_blocked,
    road_risk_increased,
)
from perception import perceive_agent
from planner import location_heuristic, plan, plan_cost
from reasoner import explain, forward_chain
from scenario import build_default_world
from events import EventScheduler, task_added
from replanning import detect_invalidation, make_goal_test, replan, replan_all
from world import World

BANNER = "=" * 64


def section(title: str) -> None:
    print()
    print(BANNER)
    print(title)
    print(BANNER)


def main() -> None:
    world = build_default_world()
    world.schedule_event(road_blocked(6, "E", "G"))
    world.schedule_event(civilian_found(
        10, "civ3", "C", needs_medical=True
    ))
    world.schedule_event(fuel_low(11, "Logistics-1", 4))
    world.schedule_event(deadline_shortened(12, 20))
    world.schedule_event(road_risk_increased(13, "A", "B", 0.9))

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
        perceive_agent(agent, world)
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
        perceive_agent(agent, world)
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

    # ---------------------------------------------------------------- #
    #  Steps 8-9 -- optimization and goal selection                    #
    # ---------------------------------------------------------------- #
    section("STEPS 8-9 -- utility optimization and goal selection")
    optimized = optimize_assignment(world, world.agents, max_tasks_per_agent=2)
    print(">> Highest-utility feasible assignment:")
    utility = mission_utility(
        world, optimized.assignment, optimized.unassigned
    )
    print(f"   mission utility: {utility:.2f}")
    for tid in sorted(optimized.assignment):
        print(f"    {tid:<26} -> {optimized.assignment[tid]}")
    if optimized.unassigned:
        print(f"   unassigned: {', '.join(sorted(optimized.unassigned))}")

    print("\n>> Ranked goals for each agent:")
    for agent in world.agents.values():
        ranked = rank_goals(world, agent)
        selected = select_goal(world, agent)
        if selected is None:
            print(f"    {agent.id:<14} no feasible goals")
            continue
        print(f"    {agent.id:<14} selects {selected.task_id}")
        for candidate in ranked[:3]:
            print(f"        {candidate.task_id:<26} priority={candidate.priority:.2f} "
                  f"cost={candidate.estimated_cost:.2f}")

    # ---------------------------------------------------------------- #
    #  Steps 10-11 -- claims, CSP, and rejection feedback loop         #
    # ---------------------------------------------------------------- #
    section("STEPS 10-11 -- claims, assignment, and conflict resolution")
    hub = CommunicationHub()

    # Step 10: agents announce their top preferences.  Claims are a
    # signal to the CSP, not a decision.
    claims = solicit_claims(world, hub)
    print(">> Claimed tasks (agent priorities, descending):")
    for task_id, bucket in sorted(claims.items()):
        for claim in bucket:
            print(f"    {claim.agent_id:<14} -> {task_id:<26} "
                  f"priority={claim.priority:.2f}")
    if not claims:
        print("    (no claims)")

    # The CSP still owns the decision.  Claims only influence the score
    # used to rank candidate solutions.
    from communication import claim_aware_score
    negotiated_input = optimize_assignment(
        world, world.agents,
        max_tasks_per_agent=2,
        score=claim_aware_score(world, claims),
    )

    # Step 11: offers, accepts, rejects.
    negotiated = negotiate_assignment(
        world, hub, negotiated_input,
        claims=claims,
    )
    print("\n>> Offers/accepts/rejects from round 0:")
    for task_id, agent_id in sorted(negotiated.confirmed.items()):
        print(f"    {task_id:<26} accepted by {agent_id}")
    if negotiated.rejected:
        print("\n>> Rejected (will be fed back into the CSP):")
        for task_id, agent_id in sorted(negotiated.rejected.items()):
            print(f"    {task_id:<26} rejected by {agent_id}")

    # Step 11/13: rejected pairs become domain pruning for a fresh CSP
    # run.  The loop terminates when nothing new can be placed.
    negotiated = reassign_rejected(
        world, hub, negotiated,
        max_tasks_per_agent=2,
        claims=claims,
    )
    print(f"\n>> After {negotiated.round} round(s) of feedback:")
    for task_id, agent_id in sorted(negotiated.confirmed.items()):
        print(f"    {task_id:<26} confirmed by {agent_id}")
    if negotiated.rejected:
        print("   still unassignable:")
        for task_id, agent_id in sorted(negotiated.rejected.items()):
            print(f"    {task_id:<26} (last rejected by {agent_id})")
    if negotiated.exhausted:
        print("   >> negotiation exhausted; remaining tasks are infeasible "
              "under current constraints")

    print("\n>> Communication log:")
    print(render_messages(hub, last_n=20))

    # ---------------------------------------------------------------- #
    #  Advanced dynamic events -- state changes force new decisions      #
    # ---------------------------------------------------------------- #
    section("ADVANCED EVENTS -- fuel, deadline, and road risk changes")
    print(">> Before events:")
    print(f"    deadline: {world.deadline}; Logistics-1 fuel: "
          f"{world.agents['Logistics-1'].fuel}; A-B risk: "
          f"{world.get_road('A', 'B').risk}")
    world.advance_time(8)
    print(">> After advancing to the scheduled events:")
    print(f"    time: {world.time}; deadline: {world.deadline}; "
          f"Logistics-1 fuel: {world.agents['Logistics-1'].fuel}; "
          f"A-B risk: {world.get_road('A', 'B').risk}")
    print("\n>> Event log:")
    print(render_events(world, last_n=8))

    world.release_assignments()
    for agent in world.agents.values():
        perceive_agent(agent, world)
        forward_chain(agent.knowledge)
    updated = optimize_assignment(world, world.agents, max_tasks_per_agent=2)
    print("\n>> Re-optimized assignment after environmental changes:")
    for task_id, agent_id in sorted(updated.assignment.items()):
        print(f"    {task_id:<26} -> {agent_id}")
    if updated.unassigned:
        print(f"    unassigned: {', '.join(sorted(updated.unassigned))}")


        # ---------------------------------------------------------------- #
    #  Step 12 -- dynamic events during the simulation                  #
    # ---------------------------------------------------------------- #
    section("STEP 12 -- dynamic events (scheduled + stochastic)")

    world_ev = build_default_world()
    scheduler = EventScheduler(seed=11)

    # -- deterministic, pre-scheduled events ------------------------- #
    scheduler.schedule(road_blocked(3, "E", "G"))
    scheduler.schedule(deadline_shortened(9, 35))

    # -- stochastic events: rolled every tick against a seeded RNG --- #
    discovered = [0]
    def _new_civilian(t: int):
        discovered[0] += 1
        loc = ("A", "C", "D")[discovered[0] % 3]
        return civilian_found(
            t, f"civ_stoch_{discovered[0]}", loc, needs_medical=True
        )
    scheduler.add_stochastic(0.5, _new_civilian)

    world_ev.set_event_scheduler(scheduler)

    # -- install a listener so agents are notified the instant an event fires #
    def _broadcast_event(w: World, evt) -> None:
        coordinator = next(iter(w.agents))
        for agent in w.agents.values():
            if agent.id == coordinator:
                continue
            # Logged only in this demo; the real loop (Step 14) would use
            # the hub here.  Kept as a placeholder so the shape is visible.
            _ = (agent.id, evt.kind)

    world_ev.add_event_listener(_broadcast_event)

    print(">> Scheduled:      t=3 road E->G blocked;  t=9 deadline -> 35")
    print(">> Stochastic:     p=0.5 per tick, new injured civilian appears")
    print()

    start_events = len(world_ev.events)
    start_civilians = len(world_ev.civilians)
    world_ev.advance_time(12)

    new_events = world_ev.events[start_events:]
    print(f">> After 12 ticks:  time={world_ev.time}/{world_ev.deadline}, "
          f"events={len(new_events)}, "
          f"civilians+={len(world_ev.civilians) - start_civilians}")
    for e in new_events:
        print(f"     [t={e.time}] {e.kind}: {e.payload}")

    # ---------------------------------------------------------------- #
    #  Step 13 -- plan invalidation and autonomous replanning           #
    # ---------------------------------------------------------------- #
    section("STEP 13 -- plan invalidation and autonomous replanning")

    world_rp = build_default_world()
    hub = CommunicationHub()

    log = world_rp.agents["Logistics-1"]
    perceive_agent(log, world_rp)
    forward_chain(log.knowledge)

    # Give Logistics-1 a commitment to make the demo deterministic.
    goal_task = world_rp.tasks["deliver_food_G"]
    log.goal = goal_task.id
    log.plan = plan(
        log.knowledge,
        make_goal_test(world_rp, log.id, goal_task),
        heuristic=location_heuristic(log.id, goal_task.target),
        schemas=(MOVE, PICK_UP, DELIVER),
    ) or []
    log.plan_index = 0
    log.plan_created_at = world_rp.time

    def _fmt(actions):
        return " -> ".join(str(a) for a in actions) if actions else "(none)"

    plan_traverses_eg = any(
        {a.bindings.get("?from"), a.bindings.get("?to")} == {"E", "G"}
        for a in log.plan
        if a.action.name == "Move"
    )

    print(f">> Logistics-1's commitment: {log.goal}")
    print(f"   plan:                 {_fmt(log.plan)}")
    print(f"   plan traverses E-G:   {plan_traverses_eg}")

    print()
    print(">> EVENT at t=1: road E->G becomes blocked")
    world_rp.block_road("E", "G")

    inval = detect_invalidation(log, world_rp)
    print(f"\n>> detect_invalidation => "
          f"{'still valid' if inval is None else inval}")

    print(f"\n>> running replan(...)")
    new_plan = replan(log, world_rp, hub,
                      reason=str(inval) if inval else "manual trigger")

    print(f">> new goal: {log.goal}")
    print(f">> new plan: {_fmt(new_plan)}")

    print(f"\n>> WARNING broadcast from Logistics-1:")
    for msg in hub.messages:
        print(f"   [{msg.type.value}] {msg.sender} -> {msg.receiver}: "
              f"{msg.content}")

    # ---------------------------------------------------------------- #
    #  Show replan_all sweeping the whole team                         #
    # ---------------------------------------------------------------- #
    print()
    print(">> replan_all(world, hub):")
    swept = replan_all(world_rp, hub,
                       reason="E-G blocked; reassess team commitments")
    if not swept:
        print("   (no agent had an invalid plan)")
    for agent_id, new_plan in swept.items():
        print(f"   {agent_id}: {_fmt(new_plan)}")


if __name__ == "__main__":
    main()