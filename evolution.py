"""Generational GA over decision policies (and, optionally, scenarios).

Modes
-----
* single  -- one Policy per individual (classic)
* team    -- one TeamPolicy per individual (per-agent specialisation)

Selection is NSGA-II style: Pareto-rank the population on the four-
objective fitness vector, then break ties with crowding distance.  This
preserves behavioural diversity instead of collapsing every genome
toward a scalar optimum.

Fitness is evaluated in parallel across a persistent process pool
created once per run.

Public entry points: :func:`evolve`, :func:`coevolve`.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from typing import (Any, Dict, List, Optional, Sequence, Tuple)

from fitness import (
    FitnessVector,
    crowding_distance,
    mean_vectors,
    pareto_fronts,
)
from policy import Policy, TeamPolicy, DEFAULT_POLICY


# --------------------------------------------------------------------------- #
#  Gene bounds
# --------------------------------------------------------------------------- #

#: (name, low, high) for every Policy field, in declaration order.
POLICY_GENE_BOUNDS: Tuple[Tuple[str, float, float], ...] = (
    ("w_importance",         0.0,   3.0),
    ("w_urgency",            0.0,   3.0),
    ("w_cost",               0.0,   3.0),
    ("w_success",            0.0,   3.0),
    ("w_unassigned",         0.0,  30.0),
    ("w_cost_scale",         0.0,   3.0),
    ("w_cooperation",        0.0,  20.0),
    ("max_tasks_per_agent",  1.0,   4.0),
    ("w_risk_aversion",      0.0,   1.0),
    ("w_supply_proximity",   0.0,   3.0),
    ("goal_top_k",           1.0,  20.0),
    ("w_fuel_reserve",       0.0,  10.0),
)

assert tuple(n for n, _, _ in POLICY_GENE_BOUNDS) == tuple(
    f.name for f in fields(Policy)
), "POLICY_GENE_BOUNDS order must match Policy field order"


def gene_bounds_for(mode: str):
    if mode == "single":
        return POLICY_GENE_BOUNDS
    if mode == "team":
        return POLICY_GENE_BOUNDS * 4
    raise ValueError(f"unknown mode: {mode!r}")


# --------------------------------------------------------------------------- #
#  Individuals and config
# --------------------------------------------------------------------------- #

@dataclass
class Individual:
    genome: Any                                # Policy | TeamPolicy
    vector: Optional[FitnessVector] = None
    pareto_rank: int = 0
    crowding: float = 0.0
    #: Scalar shadow for display / backward compat.
    fitness: float = float("-inf")


@dataclass
class EvolutionConfig:
    population_size: int = 24
    generations: int = 15
    elite: int = 4
    seeds: Sequence[int] = (1, 2, 3, 4)
    tournament_k: int = 3
    crossover_alpha: float = 0.3
    mutation_rate: float = 0.2
    mutation_scale: float = 0.15
    rng_seed: int = 42
    #: 0 = auto (cpu_count // 2), 1 = serial, N > 1 = cap at N workers.
    workers: int = 0
    #: "single" (one Policy) or "team" (one TeamPolicy).
    mode: str = "single"
    #: Mission kwargs passed to every evaluation.
    base_kwargs: Dict = field(default_factory=lambda: dict(
        deadline=200,
        max_ticks=400,
        task_probability=0.1,
        plan_max_expansions=1500,
        noise=True,
    ))


# --------------------------------------------------------------------------- #
#  Vector-space operators
# --------------------------------------------------------------------------- #

def _clamp(vec: Sequence[float], bounds) -> List[float]:
    return [min(hi, max(lo, x)) for x, (_, lo, hi) in zip(vec, bounds)]


def random_vector(rng: random.Random, bounds) -> List[float]:
    return [rng.uniform(lo, hi) for _, lo, hi in bounds]


def crossover_vectors(a: Sequence[float], b: Sequence[float],
                      rng: random.Random, alpha: float) -> List[float]:
    child = []
    for x, y in zip(a, b):
        lo, hi = min(x, y), max(x, y)
        spread = (hi - lo) or 1e-6
        child.append(rng.uniform(lo - alpha * spread, hi + alpha * spread))
    return child


def mutate_vector(vec: List[float], rng: random.Random,
                  bounds, rate: float, scale: float) -> List[float]:
    out = list(vec)
    for i, (_, lo, hi) in enumerate(bounds):
        if rng.random() < rate:
            out[i] += rng.gauss(0.0, scale) * (hi - lo)
    return out


# --------------------------------------------------------------------------- #
#  Genome construction
# --------------------------------------------------------------------------- #

def _genome_from_vector(mode: str, vec) -> Any:
    if mode == "single":
        return Policy.from_vector(vec)
    if mode == "team":
        return TeamPolicy.from_vector(vec)
    raise ValueError(f"unknown mode: {mode!r}")


def _seed_genome(mode: str) -> Any:
    if mode == "single":
        return DEFAULT_POLICY
    if mode == "team":
        return TeamPolicy.uniform(DEFAULT_POLICY)
    raise ValueError(f"unknown mode: {mode!r}")


# --------------------------------------------------------------------------- #
#  Fitness worker (top-level so it is picklable)
# --------------------------------------------------------------------------- #

def _evaluate_pair(args) -> FitnessVector:
    genome, seed, base_kwargs = args
    from fitness import _mission_vector
    from main import MissionConfig, run_mission
    cfg = MissionConfig(seed=seed, policy=genome, **base_kwargs)
    world, _, result = run_mission(cfg)
    return _mission_vector(world, result)


def _resolve_workers(cfg: EvolutionConfig) -> int:
    if cfg.workers == 1:
        return 1
    if cfg.workers > 0:
        requested = cfg.workers
    else:
        requested = max(1, (os.cpu_count() or 2) // 2)
    useful = max(1, cfg.population_size * len(cfg.seeds))
    return max(1, min(requested, useful))


def _evaluate_population(
    population: List[Individual],
    cfg: EvolutionConfig,
    pool: Optional[ProcessPoolExecutor],
) -> None:
    pending = [ind for ind in population if ind.vector is None]
    if not pending:
        return

    tasks = [
        (ind.genome, seed, cfg.base_kwargs)
        for ind in pending
        for seed in cfg.seeds
    ]

    if pool is None:
        flat = [_evaluate_pair(t) for t in tasks]
    else:
        flat = list(pool.map(_evaluate_pair, tasks))

    n = len(cfg.seeds)
    for i, ind in enumerate(pending):
        vecs = flat[i * n:(i + 1) * n]
        ind.vector = mean_vectors(vecs)
        ind.fitness = ind.vector.score


# --------------------------------------------------------------------------- #
#  Pareto selection
# --------------------------------------------------------------------------- #

def rank_population(population: List[Individual]) -> None:
    vectors = [ind.vector for ind in population]
    fronts = pareto_fronts(vectors)
    for rank, front in enumerate(fronts):
        cd = crowding_distance(vectors, front)
        for idx, cd_val in zip(front, cd):
            population[idx].pareto_rank = rank
            population[idx].crowding = cd_val


def sort_by_pareto(population: List[Individual]) -> List[Individual]:
    rank_population(population)
    return sorted(
        population,
        key=lambda ind: (ind.pareto_rank, -ind.crowding, -ind.fitness),
    )


def tournament(
    population: List[Individual],
    rng: random.Random,
    k: int,
) -> Individual:
    k = min(k, len(population))
    return min(
        rng.sample(population, k),
        key=lambda ind: (ind.pareto_rank, -ind.crowding),
    )


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #

def evolve(
    cfg: EvolutionConfig,
    *,
    log: bool = True,
) -> Tuple[Any, List[List[Individual]]]:
    """Run the GA and return (best_genome, per_generation_history)."""
    rng = random.Random(cfg.rng_seed)
    bounds = gene_bounds_for(cfg.mode)
    workers = _resolve_workers(cfg)

    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    if log:
        mode_str = "serial" if workers == 1 else f"{workers} workers"
        print(f"[evolve] mode={cfg.mode}  fitness evaluation: {mode_str}")

    population: List[Individual] = [
        Individual(genome=_seed_genome(cfg.mode))
    ]
    while len(population) < cfg.population_size:
        vec = random_vector(rng, bounds)
        population.append(
            Individual(genome=_genome_from_vector(cfg.mode, vec))
        )

    history: List[List[Individual]] = []

    try:
        for gen in range(cfg.generations):
            _evaluate_population(population, cfg, pool)
            population = sort_by_pareto(population)
            history.append(list(population))

            if log:
                front0 = [ind for ind in population if ind.pareto_rank == 0]
                best = population[0]
                print(
                    f"[gen {gen:02d}] front0={len(front0):2d}  "
                    f"best_score={best.vector.score:8.2f}  "
                    f"best_done={best.vector.completed:5.2f}  "
                    f"best_speed={best.vector.speed:5.3f}"
                )

            first_front = [ind for ind in population if ind.pareto_rank == 0]
            elite_count = max(cfg.elite, len(first_front))
            next_pop: List[Individual] = [
                Individual(genome=ind.genome, vector=ind.vector)
                for ind in first_front[:elite_count]
            ]

            while len(next_pop) < cfg.population_size:
                p1 = tournament(population, rng, cfg.tournament_k)
                p2 = tournament(population, rng, cfg.tournament_k)
                child_vec = crossover_vectors(
                    p1.genome.as_vector(),
                    p2.genome.as_vector(),
                    rng, cfg.crossover_alpha,
                )
                child_vec = mutate_vector(
                    child_vec, rng, bounds,
                    cfg.mutation_rate, cfg.mutation_scale,
                )
                child_vec = _clamp(child_vec, bounds)
                next_pop.append(
                    Individual(
                        genome=_genome_from_vector(cfg.mode, child_vec)
                    )
                )

            population = next_pop

        _evaluate_population(population, cfg, pool)
        population = sort_by_pareto(population)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    return population[0].genome, history


# --------------------------------------------------------------------------- #
#  Co-evolution
# --------------------------------------------------------------------------- #

@dataclass
class CoevolutionConfig:
    policy: EvolutionConfig = field(default_factory=EvolutionConfig)
    rounds: int = 8
    scenario_population_size: int = 12
    sample_k: int = 4
    eval_seeds: Sequence[int] = (101, 102, 103)
    hall_of_fame_size: int = 8
    rng_seed: int = 1234


def _negate(v: FitnessVector) -> FitnessVector:
    return FitnessVector(-v.score, -v.completed, -v.speed, -v.efficiency)


def _eval_policy_against(
    genome,
    scenarios,
    seeds: Sequence[int],
    base_kwargs: Dict,
) -> FitnessVector:
    from adversarial import build_world_from_genome
    from fitness import _mission_vector
    from simulation import Simulation

    vecs: List[FitnessVector] = []
    for scenario in scenarios:
        for seed in seeds:
            world = build_world_from_genome(
                scenario, seed,
                deadline=base_kwargs.get("deadline", 200),
            )
            sim = Simulation(
                world,
                max_ticks=base_kwargs.get("max_ticks", 400),
                plan_max_expansions=base_kwargs.get(
                    "plan_max_expansions", 1500
                ),
                policy=genome,
            )
            sim.setup()
            result = sim.run()
            vecs.append(_mission_vector(world, result))
    return mean_vectors(vecs)


def coevolve(cfg: CoevolutionConfig, *, log: bool = True):
    """Alternate policy evolution and scenario evolution.

    Returns ``(best_policy, scenario_hof, history)``.
    """
    from adversarial import (
        ScenarioGenome,
        SCENARIO_GENE_BOUNDS,
    )

    rng = random.Random(cfg.rng_seed)
    policy_cfg = cfg.policy
    bounds = gene_bounds_for(policy_cfg.mode)

    # Seed both populations.
    policies: List[Individual] = [
        Individual(genome=_seed_genome(policy_cfg.mode))
    ]
    while len(policies) < policy_cfg.population_size:
        v = random_vector(rng, bounds)
        policies.append(
            Individual(genome=_genome_from_vector(policy_cfg.mode, v))
        )

    scenarios: List[ScenarioGenome] = [ScenarioGenome()]
    while len(scenarios) < cfg.scenario_population_size:
        v = random_vector(rng, SCENARIO_GENE_BOUNDS)
        scenarios.append(ScenarioGenome.from_vector(v))

    policy_hof: List[Individual] = []
    scenario_hof: List[ScenarioGenome] = []
    history: List = []

    for round_idx in range(cfg.rounds):
        # -- 1. Score policies against sampled scenarios. --------------- #
        sample_scenarios = [scenarios[0]] + rng.sample(
            scenarios, min(cfg.sample_k, len(scenarios))
        )
        for ind in policies:
            ind.vector = _eval_policy_against(
                ind.genome, sample_scenarios,
                cfg.eval_seeds, policy_cfg.base_kwargs,
            )
            ind.fitness = ind.vector.score
        policies = sort_by_pareto(policies)

        # -- 2. Score scenarios against sampled policies (flip sign). --- #
        policy_sample = policies[: max(1, cfg.sample_k)] + policy_hof
        scenario_individuals: List[Individual] = []
        for s in scenarios:
            vecs: List[FitnessVector] = []
            for ind in policy_sample:
                v = _eval_policy_against(
                    ind.genome, [s],
                    cfg.eval_seeds, policy_cfg.base_kwargs,
                )
                vecs.append(_negate(v))
            s_vec = mean_vectors(vecs)
            scenario_individuals.append(
                Individual(genome=s, vector=s_vec, fitness=s_vec.score)
            )
        scenario_individuals = sort_by_pareto(scenario_individuals)

        # -- 3. Update halls of fame. ----------------------------------- #
        policy_hof.extend(policies[: max(1, cfg.hall_of_fame_size // 4)])
        policy_hof = policy_hof[-cfg.hall_of_fame_size:]
        scenario_hof.extend(
            [i.genome for i in scenario_individuals[:2]]
        )
        scenario_hof = scenario_hof[-cfg.hall_of_fame_size:]

        history.append((round_idx, list(policies),
                        list(scenario_individuals)))

        if log:
            best_p = policies[0]
            hardest = scenario_individuals[0]
            print(
                f"[coevo {round_idx:02d}] policy_score={best_p.vector.score:8.2f}"
                f"  worst_scenario_fitness={hardest.vector.score:8.2f}"
            )

        # -- 4. Reproduce both populations. ----------------------------- #
        policies = _reproduce_policies(policies, policy_cfg, bounds, rng)

        next_scenarios = _reproduce_scenarios(
            scenario_individuals, cfg.scenario_population_size,
            rng, SCENARIO_GENE_BOUNDS,
        )
        scenarios = [ind.genome for ind in next_scenarios]
        if not any(s == ScenarioGenome() for s in scenarios):
            scenarios[0] = ScenarioGenome()

    return policies[0].genome, scenario_hof, history


def _reproduce_policies(
    population: List[Individual],
    cfg: EvolutionConfig,
    bounds,
    rng: random.Random,
) -> List[Individual]:
    first_front = [ind for ind in population if ind.pareto_rank == 0]
    next_pop: List[Individual] = [
        Individual(genome=ind.genome, vector=ind.vector)
        for ind in first_front[: max(cfg.elite, len(first_front))]
    ]
    while len(next_pop) < cfg.population_size:
        p1 = tournament(population, rng, cfg.tournament_k)
        p2 = tournament(population, rng, cfg.tournament_k)
        child = crossover_vectors(
            p1.genome.as_vector(), p2.genome.as_vector(),
            rng, cfg.crossover_alpha,
        )
        child = mutate_vector(
            child, rng, bounds, cfg.mutation_rate, cfg.mutation_scale,
        )
        child = _clamp(child, bounds)
        next_pop.append(
            Individual(genome=_genome_from_vector(cfg.mode, child))
        )
    return next_pop


def _reproduce_scenarios(
    population: List[Individual],
    size: int,
    rng: random.Random,
    bounds,
) -> List[Individual]:
    from adversarial import ScenarioGenome

    first_front = [ind for ind in population if ind.pareto_rank == 0]
    next_pop: List[Individual] = [
        Individual(genome=ind.genome, vector=ind.vector)
        for ind in first_front[: max(2, len(first_front))]
    ]
    while len(next_pop) < size:
        p1 = tournament(population, rng, 3)
        p2 = tournament(population, rng, 3)
        child = crossover_vectors(
            p1.genome.as_vector(), p2.genome.as_vector(),
            rng, 0.3,
        )
        child = mutate_vector(child, rng, bounds, 0.25, 0.2)
        child = _clamp(child, bounds)
        next_pop.append(
            Individual(genome=ScenarioGenome.from_vector(child))
        )
    return next_pop


__all__ = [
    "POLICY_GENE_BOUNDS",
    "gene_bounds_for",
    "Individual",
    "EvolutionConfig",
    "CoevolutionConfig",
    "evolve",
    "coevolve",
    "random_vector",
    "crossover_vectors",
    "mutate_vector",
    "tournament",
    "rank_population",
    "sort_by_pareto",
]