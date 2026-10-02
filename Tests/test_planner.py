"""Unit tests for the A* state-space planner (planner.py)."""

from __future__ import annotations

import unittest

from actions import MOVE
from kb import Fact
from perception import perceive_agent
from planner import location_heuristic, plan, plan_cost
from reasoner import forward_chain
from scenario import build_default_world


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.world = build_default_world()
        self.agent = self.world.agents["Logistics-1"]
        perceive_agent(self.agent, self.world)
        forward_chain(self.agent.knowledge)

    def test_move_plan_found(self):
        goal = lambda kb: kb.contains(Fact("at", ("Logistics-1", "E")))
        h = location_heuristic("Logistics-1", "E")
        result = plan(
            self.agent.knowledge, goal, heuristic=h, schemas=(MOVE,)
        )
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        for step in result:
            self.assertEqual(step.action.name, "Move")

    def test_plan_cost_positive(self):
        goal = lambda kb: kb.contains(Fact("at", ("Logistics-1", "E")))
        h = location_heuristic("Logistics-1", "E")
        result = plan(
            self.agent.knowledge, goal, heuristic=h, schemas=(MOVE,)
        )
        self.assertIsNotNone(result)
        self.assertGreater(plan_cost(result, self.agent.knowledge), 0.0)

    def test_unreachable_goal_returns_none(self):
        # Block every road that touches A, isolating Logistics-1.
        for road in list(self.world.roads):
            if "A" in (road.a, road.b):
                road.blocked = True
        perceive_agent(self.agent, self.world)
        forward_chain(self.agent.knowledge)

        goal = lambda kb: kb.contains(Fact("at", ("Logistics-1", "G")))
        h = location_heuristic("Logistics-1", "G")
        result = plan(
            self.agent.knowledge,
            goal,
            heuristic=h,
            schemas=(MOVE,),
            max_expansions=500,
        )
        self.assertIsNone(result)

    def test_goal_already_satisfied(self):
        goal = lambda kb: kb.contains(Fact("at", ("Logistics-1", "A")))
        h = location_heuristic("Logistics-1", "A")
        result = plan(
            self.agent.knowledge, goal, heuristic=h, schemas=(MOVE,)
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()