"""Unit tests for the task-assignment CSP (constraints.py)."""

from __future__ import annotations

import unittest

from constraints import solve_task_assignment
from perception import perceive_agent
from reasoner import forward_chain
from scenario import build_default_world


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.world = build_default_world()
        for agent in self.world.agents.values():
            perceive_agent(agent, self.world)
            forward_chain(agent.knowledge)

    def test_assignments_are_feasible(self):
        result = solve_task_assignment(
            self.world, self.world.agents, max_tasks_per_agent=2
        )
        for task_id, agent_id in result.assignment.items():
            self.assertTrue(
                self.world.agents[agent_id].knowledge.contains(
                    ("can_assign", (agent_id, task_id))
                ),
                f"{agent_id} cannot actually take {task_id}",
            )

    def test_load_cap_enforced(self):
        result = solve_task_assignment(
            self.world, self.world.agents, max_tasks_per_agent=1
        )
        counts: dict = {}
        for agent_id in result.assignment.values():
            counts[agent_id] = counts.get(agent_id, 0) + 1
        for count in counts.values():
            self.assertLessEqual(count, 1)

    def test_forbidden_pruning_blocks_everything(self):
        forbidden = {
            tid: set(self.world.agents.keys())
            for tid in self.world.tasks
        }
        result = solve_task_assignment(
            self.world,
            self.world.agents,
            max_tasks_per_agent=2,
            forbidden=forbidden,
        )
        self.assertEqual(len(result.assignment), 0)
        self.assertEqual(len(result.unassigned), len(self.world.tasks))

    def test_partial_assignment_when_constrained(self):
        result = solve_task_assignment(
            self.world, self.world.agents, max_tasks_per_agent=1
        )
        # 3 agents, each limited to 1 task -> at most 3 assigned.
        self.assertLessEqual(len(result.assignment), 3)
        self.assertTrue(len(result.unassigned) > 0)

    def test_every_task_is_accounted_for(self):
        result = solve_task_assignment(
            self.world, self.world.agents, max_tasks_per_agent=2
        )
        assigned = set(result.assignment.keys())
        unassigned = set(result.unassigned)
        self.assertEqual(
            assigned | unassigned,
            set(self.world.tasks.keys()),
        )
        self.assertFalse(assigned & unassigned)


if __name__ == "__main__":
    unittest.main()