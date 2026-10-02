"""Adversarial scenario genomes for co-evolution.

Each ScenarioGenome encodes the timing and rate of stochastic events
that the world can throw at the agents.  Co-evolution alternates
between:

  1. Evolving policies to maximise mission score against the current
     scenario population.
  2. Evolving scenarios to minimise mission score against the current
     policy population.

The final scenario population is a stress-test suite that the policies
were unable to beat -- a reusable evaluation panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ScenarioGenome:
    """The timing (in ticks) and rate of disruptions."""

    block_time: int = 6
    civilian_time: int = 9
    risk_time: int = 13
    fuel_time: int = 20
    task_probability: float = 0.1

    def as_vector(self) -> List[float]:
        return [
            float(self.block_time),
            float(self.civilian_time),
            float(self.risk_time),
            float(self.fuel_time),
            float(self.task_probability),
        ]

    @classmethod
    def from_vector(cls, vec) -> "ScenarioGenome":
        return cls(
            block_time=max(1, int(round(vec[0]))),
            civilian_time=max(1, int(round(vec[1]))),
            risk_time=max(1, int(round(vec[2]))),
            fuel_time=max(1, int(round(vec[3]))),
            task_probability=min(0.5, max(0.0, float(vec[4]))),
        )


#: (name, low, high) in ScenarioGenome field order.
SCENARIO_GENE_BOUNDS = (
    ("block_time",        1.0,   60.0),
    ("civilian_time",     1.0,   60.0),
    ("risk_time",         1.0,   60.0),
    ("fuel_time",         1.0,   60.0),
    ("task_probability",  0.0,    0.4),
)


def build_world_from_genome(
    genome: ScenarioGenome,
    seed: int,
    deadline: int = 200,
):
    """Assemble a world whose event schedule is controlled by the genome."""
    from events import (
        EventScheduler,
        TaskGenerator,
        civilian_found,
        fuel_low,
        road_blocked,
        road_risk_increased,
    )
    from scenario import build_default_world

    world = build_default_world()
    world.deadline = deadline

    scheduler = EventScheduler(seed=seed)
    scheduler.add_stochastic(
        genome.task_probability,
        TaskGenerator(
            locations=tuple(world.locations.keys()),
            id_prefix=f"coevo{seed}",
            seed=seed * 31 + 7,
            min_deadline_offset=15,
            max_deadline_offset=40,
        ),
    )
    scheduler.schedule(road_blocked(genome.block_time, "E", "G"))
    scheduler.schedule(civilian_found(
        genome.civilian_time, f"civ_coevo{seed}", "C",
        needs_medical=True,
    ))
    scheduler.schedule(road_risk_increased(genome.risk_time, "A", "B", 0.7))
    scheduler.schedule(fuel_low(genome.fuel_time, "Logistics-1", 6))

    world.set_event_scheduler(scheduler)
    return world