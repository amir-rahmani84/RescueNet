"""Unit tests for the forward-chaining reasoner (reasoner.py)."""

from __future__ import annotations

import unittest

from kb import Fact, KnowledgeBase
from reasoner import forward_chain
from scenario import build_default_world
from perception import perceive_agent


class TestReasoner(unittest.TestCase):
    def _base_kb(self) -> KnowledgeBase:
        kb = KnowledgeBase()
        # capabilities
        kb.add(Fact("has_capability", ("R1", "medical")))
        kb.add(Fact("has_capability", ("L1", "transport")))
        kb.add(Fact("has_capability", ("E1", "repair")))
        # agent locations
        kb.add(Fact("at", ("R1", "E")))
        kb.add(Fact("at", ("L1", "A")))
        kb.add(Fact("at", ("E1", "B")))
        # roads (open, both directions)
        for a, b in [
            ("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"),
            ("B", "E"), ("C", "F"), ("E", "F"), ("E", "G"),
        ]:
            kb.add(Fact("road_open", (a, b)))
            kb.add(Fact("road_open", (b, a)))
        # one pending task at B requiring transport
        kb.add(Fact("task_pending", ("t1",)))
        kb.add(Fact("task_type", ("t1", "deliver")))
        kb.add(Fact("task_at", ("t1", "B")))
        kb.add(Fact("requires_capability", ("t1", "transport")))
        return kb

    def test_can_reach_derived(self):
        kb = self._base_kb()
        forward_chain(kb)
        for loc in "ABCDEFG":
            self.assertTrue(
                kb.contains(Fact("can_reach", ("R1", loc))),
                f"R1 should be able to reach {loc}",
            )

    def test_capability_match(self):
        kb = self._base_kb()
        forward_chain(kb)
        self.assertTrue(kb.contains(Fact("can_perform", ("L1", "t1"))))

    def test_can_assign_derived(self):
        kb = self._base_kb()
        forward_chain(kb)
        # L1 is transport, can reach B, t1 is deliver at B
        self.assertTrue(kb.contains(Fact("can_assign", ("L1", "t1"))))
        self.assertFalse(kb.contains(Fact("can_assign", ("E1", "t1"))))

    def test_blocked_road_removes_reachability(self):
        kb = self._base_kb()
        # Cut G off from the rest of the graph.
        kb.remove(Fact("road_open", ("E", "G")))
        kb.remove(Fact("road_open", ("G", "E")))
        forward_chain(kb)
        self.assertFalse(kb.contains(Fact("can_reach", ("R1", "G"))))
        self.assertTrue(kb.contains(Fact("can_reach", ("R1", "E"))))

    def test_derivations_are_reported(self):
        kb = self._base_kb()
        derivations = forward_chain(kb)
        predicates = {d.fact.predicate for d in derivations}
        self.assertIn("can_reach", predicates)
        self.assertIn("can_assign", predicates)


class TestReasonerOnScenario(unittest.TestCase):
    def test_default_world_can_assign_after_perception(self):
        world = build_default_world()
        agent = world.agents["Rescue-1"]
        perceive_agent(agent, world)
        forward_chain(agent.knowledge)
        # Rescue-1 should be able to reach F (adjacent to E)
        self.assertTrue(
            agent.knowledge.contains(Fact("can_reach", ("Rescue-1", "F")))
        )


if __name__ == "__main__":
    unittest.main()