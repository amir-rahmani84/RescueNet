"""RescueNet -- integrated autonomous mission simulation.

Everything the project provides runs as a single simulation:

    scenario  ->  world  ->  agents
              ->  perception  ->  forward chaining  ->  planning
              ->  negotiation  ->  execution  ->  replanning
              ->  world changes  ->  summary

Task identifiers like ``dyn7_001`` are hard to read, so every report in
this driver renders the task's *description* instead (with the raw id in
parentheses for cross-referencing).

Run:
    python main.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from display import (
    RULE,
    render_agents,
    render_dashboard,
    render_events,
    render_messages,
)
from event_audit import audit_event_coverage
from events import (
    EventScheduler,
    TaskGenerator,
    civilian_found,
    fuel_low,
    road_blocked,
    road_risk_increased,
)
from scenario import build_default_world
from simulation import MissionResult, Simulation
from world import World


# --------------------------------------------------------------------------- #
#  Task-aware rendering helpers
# --------------------------------------------------------------------------- #

def describe_task(world: World, task_id: str) -> str:
    """Human-readable label for a task id.

    Prefers the task's description; falls back to the id when the task
    has no description or is no longer in the world.
    """
    task = world.tasks.get(task_id)
    if task is None:
        return f"(unknown task {task_id})"
    label = task.description or task.id
    return f"{label}   ({task.id})"


def render_task_catalog(world: World) -> str:
    """List every task in the world with status, description, and owner."""
    if not world.tasks:
        return "    (no tasks)"
    lines: List[str] = []
    for task in world.tasks.values():
        owner = f"  <- {task.assigned_to}" if task.assigned_to else ""
        label = task.description or task.id
        lines.append(
            f"    [{task.status.value:<11}] {label}"
            f"   ({task.id}){owner}"
        )
    return "\n".join(lines)


def render_mission_summary(world: World, result: MissionResult) -> str:
    """Final scoreboard that shows task descriptions instead of raw ids."""
    lines: List[str] = [
        RULE,
        "MISSION SUMMARY",
        RULE,
        f"Time reached:  {world.time} / {world.deadline}",
        f"Iterations:    {result.iterations}",
        f"Final score:   {result.score:.2f}",
        "",
    ]
    sections = (
        ("Completed tasks:", result.completed, "✓"),
        ("Failed tasks:",    result.failed,    "✗"),
        ("Still pending:",   result.pending,   "!"),
    )
    wrote_any = False
    for title, ids, glyph in sections:
        if not ids:
            continue
        wrote_any = True
        lines.append(title)
        for tid in ids:
            lines.append(f"    {glyph} {describe_task(world, tid)}")
    if not wrote_any:
        lines.append("    (no tasks recorded)")
    lines.append(RULE)
    return "\n".join(lines)


def render_spawned_tasks(world: World, prefix: str) -> str:
    """List only the tasks that were spawned mid-mission (by id prefix)."""
    spawned = [t for t in world.tasks.values() if t.id.startswith(prefix)]
    if not spawned:
        return "    (none spawned this run)"
    lines: List[str] = []
    for task in spawned:
        lines.append(
            f"    [{task.status.value:<11}] {task.description or task.id}"
            f"   ({task.id})   target={task.target}"
        )
    return "\n".join(lines)


def render_agent_goals(world: World) -> str:
    """Show each agent's goal as the task description, not the raw id."""
    lines: List[str] = []
    for agent in world.agents.values():
        if agent.goal is None:
            lines.append(f"    {agent.id:<14} (no goal)")
            continue
        lines.append(f"    {agent.id:<14} {describe_task(world, agent.goal)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Mission configuration
# --------------------------------------------------------------------------- #

@dataclass
class MissionConfig:
    """Everything that defines one mission."""

    seed: int = 7
    deadline: int = 200
    max_ticks: int = 400
    max_tasks_per_agent: int = 3
    task_probability: float = 0.1
    plan_max_expansions: int = 3000
    noise: bool = True

    @property
    def task_prefix(self) -> str:
        return f"dyn{self.seed}"


# --------------------------------------------------------------------------- #
#  World assembly
# --------------------------------------------------------------------------- #

def build_world(cfg: MissionConfig) -> World:
    """Assemble the default scenario + stochastic sources + disruptions."""
    world = build_default_world()
    world.deadline = cfg.deadline
    if not cfg.noise:
        return world

    scheduler = EventScheduler(seed=cfg.seed)

    scheduler.add_stochastic(
        cfg.task_probability,
        TaskGenerator(
            locations=tuple(world.locations.keys()),
            id_prefix=cfg.task_prefix,
            seed=cfg.seed * 31 + 7,
            min_deadline_offset=15,
            max_deadline_offset=40,
        ),
    )

    scheduler.schedule(road_blocked(6, "E", "G"))
    scheduler.schedule(civilian_found(
        9, f"civ_seed{cfg.seed}", "C", needs_medical=True))
    scheduler.schedule(road_risk_increased(13, "A", "B", 0.7))
    scheduler.schedule(fuel_low(20, "Logistics-1", 6))

    world.set_event_scheduler(scheduler)
    return world


# --------------------------------------------------------------------------- #
#  One mission, end-to-end
# --------------------------------------------------------------------------- #

def run_mission(cfg: MissionConfig) -> Tuple[World, Simulation, MissionResult]:
    world = build_world(cfg)
    sim = Simulation(
        world,
        max_tasks_per_agent=cfg.max_tasks_per_agent,
        max_ticks=cfg.max_ticks,
        plan_max_expansions=cfg.plan_max_expansions,
    )
    sim.setup()
    result = sim.run()
    return world, sim, result


# --------------------------------------------------------------------------- #
#  Showcase run
# --------------------------------------------------------------------------- #

def run_showcase(cfg: MissionConfig) -> None:
    print(RULE)
    print("RESCUENET -- INTEGRATED MISSION SIMULATION")
    print(RULE)
    print(f"seed={cfg.seed}   deadline={cfg.deadline}   "
          f"task_probability={cfg.task_probability}   "
          f"max_ticks={cfg.max_ticks}")
    print()
    print(">> Running the full pipeline:")
    print("   scenario -> world -> agents -> perception -> reasoning")
    print("   -> planning -> negotiation -> execution -> replanning")
    print("   -> world changes -> summary")
    print()

    world, sim, result = run_mission(cfg)

    print("TASK CATALOG (all tasks, by description)")
    print(render_task_catalog(world))
    print()

    print("TASKS SPAWNED MID-MISSION")
    print(render_spawned_tasks(world, cfg.task_prefix))
    print()

    print(render_mission_summary(world, result))
    print()

    print("AGENT GOALS (as task descriptions)")
    print(render_agent_goals(world))
    print()

    print("FINAL AGENT STATE")
    print(render_agents(world))

    print()
    print("EVENTS THAT FIRED DURING THE MISSION")
    print(render_events(world, last_n=15))

    print()
    print("COMMUNICATION TRANSCRIPT (last 25 messages)")
    print(render_messages(sim.hub, last_n=25))

    print()
    print(render_dashboard(world, sim.hub, last_events=6, last_messages=8))


# --------------------------------------------------------------------------- #
#  Seed sweep
# --------------------------------------------------------------------------- #

def run_sweep(seeds: Sequence[int]) -> None:
    print()
    print(RULE)
    print("SEED SWEEP -- pipeline robustness across stochastic rollouts")
    print(RULE)

    header = (f"{'seed':>4}  {'score':>8}  {'done':>5}  "
              f"{'fail':>5}  {'pend':>5}  {'tasks':>6}  {'ticks':>6}")
    print(header)
    print("-" * len(header))

    scores: List[float] = []
    completed: List[int] = []
    per_seed: List[Tuple[int, World, MissionResult]] = []

    for seed in seeds:
        cfg = MissionConfig(seed=seed)
        world, _sim, result = run_mission(cfg)
        scores.append(result.score)
        completed.append(len(result.completed))
        per_seed.append((seed, world, result))
        print(
            f"{seed:>4}  {result.score:>8.2f}  "
            f"{len(result.completed):>5}  {len(result.failed):>5}  "
            f"{len(result.pending):>5}  {len(world.tasks):>6}  "
            f"{result.ticks:>6}"
        )

    print("-" * len(header))
    if len(scores) > 1:
        print(f"{'mean':>4}  {statistics.mean(scores):>8.2f}  "
              f"{statistics.mean(completed):>5.2f}")
        print(f"{'std':>4}  {statistics.stdev(scores):>8.2f}  "
              f"{statistics.stdev(completed):>5.2f}")
    print(f"{'best':>4}  {max(scores):>8.2f}  {max(completed):>5}")
    print(f"{'worst':>4}  {min(scores):>8.2f}  {min(completed):>5}")

    # --- per-seed task breakdown, by description ---------------------- #
    print()
    print("PER-SEED TASK OUTCOMES (by description)")
    print("-" * len(header))
    for seed, world, result in per_seed:
        print(f"seed {seed}  (score={result.score:.2f})")
        if result.completed:
            print("    ✓ completed:")
            for tid in result.completed:
                print(f"        {describe_task(world, tid)}")
        if result.failed:
            print("    ✗ failed:")
            for tid in result.failed:
                print(f"        {describe_task(world, tid)}")
        if result.pending:
            print("    ! pending:")
            for tid in result.pending:
                print(f"        {describe_task(world, tid)}")
        if not (result.completed or result.failed or result.pending):
            print("    (no tasks recorded)")


# --------------------------------------------------------------------------- #
#  Event audit
# --------------------------------------------------------------------------- #

def run_audit() -> None:
    print()
    print(audit_event_coverage().render())


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    run_showcase(MissionConfig(seed=7))
    run_sweep(seeds=[1, 2, 3, 4, 5, 6, 7, 8])
    run_audit()

    print()
    print(RULE)
    print("SIMULATION COMPLETE")
    print(RULE)


if __name__ == "__main__":
    main()