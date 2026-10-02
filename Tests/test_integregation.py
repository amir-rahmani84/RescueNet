"""Integration tests for the full RescueNet agent loop (Step 16)."""

from __future__ import annotations

import unittest

from actions import DELIVER, MOVE, PICK_UP, REFUEL
from communication import CommunicationHub, MessageType
from events import (
    civilian_found,
    fuel_low,
    road_blocked,
    road_risk_increased,
)
from enums import ResourceType, TaskStatus, TaskType
from kb import Fact, KnowledgeBase
from scenario import build_default_world
from simulation import Simulation


class TestIntegration(unittest.TestCase):
    def _make_world(self):
        world = build_default_world()
        world.schedule_event(road_blocked(6, "E", "G"))
        world.schedule_event(
            civilian_found(9, "civ3", "C", needs_medical=True)
        )
        world.schedule_event(road_risk_increased(13, "A", "B", 0.9))
        world.schedule_event(fuel_low(11, "Logistics-1", 6))
        return world

    def test_full_mission_runs(self):
        world = self._make_world()
        sim = Simulation(world, max_tasks_per_agent=2, max_ticks=80)
        sim.setup()
        result = sim.run()

        self.assertGreaterEqual(result.ticks, 0)
        accounted = (
            set(result.completed)
            | set(result.failed)
            | set(result.pending)
        )
        self.assertEqual(accounted, set(world.tasks.keys()))

    def test_communication_happens(self):
        world = self._make_world()
        sim = Simulation(
            world, hub=CommunicationHub(),
            max_tasks_per_agent=2, max_ticks=40,
        )
        sim.setup()
        sim.run()
        # Negotiation always emits at least OFFER/ACCEPT/REJECT messages.
        self.assertTrue(len(sim.hub.messages) > 0)

    def test_at_least_one_task_completes(self):
        world = build_default_world()
        sim = Simulation(world, max_tasks_per_agent=2, max_ticks=120)
        sim.setup()
        result = sim.run()
        self.assertTrue(
            len(result.completed) > 0,
            "expected at least one task to be completed in a clean run",
        )

    def test_no_task_left_in_progress_at_mission_end(self):
        world = self._make_world()
        sim = Simulation(world, max_tasks_per_agent=2, max_ticks=80)
        sim.setup()
        sim.run()
        for task in world.tasks.values():
            self.assertNotEqual(
                task.status, TaskStatus.IN_PROGRESS,
                f"{task.id} still marked IN_PROGRESS",
            )

    def test_events_were_fired(self):
        world = self._make_world()
        sim = Simulation(world, max_tasks_per_agent=2, max_ticks=80)
        sim.setup()
        sim.run()
        kinds = {e.kind for e in world.events}
        self.assertIn("road_blocked", kinds)
        self.assertIn("road_risk_increased", kinds)

    def test_blocked_road_creates_clear_task(self):
        world = build_default_world()
        world.schedule_event(road_blocked(1, "E", "G"))
        world.advance_time(1)

        task = world.tasks["unblock_road_E_G"]
        self.assertEqual(task.type, TaskType.CLEAR_ROAD)
        self.assertEqual(task.secondary_target, "G")
        self.assertEqual(task.description, "Unblock road E -> G")

    def test_engineer_can_plan_and_complete_clear_road(self):
        world = build_default_world()
        world.schedule_event(road_blocked(1, "E", "G"))
        world.advance_time(1)
        engineer = world.agents["Engineer-1"]

        from perception import perceive_agent
        from planner import plan
        from reasoner import forward_chain
        from replanning import make_goal_test

        perceive_agent(engineer, world)
        forward_chain(engineer.knowledge)
        task = world.tasks["unblock_road_E_G"]
        engineer.goal = task.id
        result = plan(
            engineer.knowledge,
            make_goal_test(world, engineer.id, task),
            max_expansions=500,
        )
        self.assertIsNotNone(result)
        self.assertTrue(any(step.action.name == "ClearRoad" for step in result))
        for action in result:
            world.apply_action(engineer.id, action)
        self.assertFalse(world.get_road("E", "G").blocked)
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_refuel_consumes_supply_and_adds_fuel(self):
        world = build_default_world()
        agent = world.agents["Logistics-1"]
        agent.fuel = 6

        from perception import perceive_agent
        from reasoner import forward_chain

        perceive_agent(agent, world)
        forward_chain(agent.knowledge)
        actions = REFUEL.instances(agent.knowledge)

        self.assertTrue(actions)
        fuel_before = world.locations["A"].resources[ResourceType.FUEL]
        world.apply_action(agent.id, actions[0])
        self.assertEqual(agent.fuel, 11)
        self.assertEqual(
            world.locations["A"].resources[ResourceType.FUEL], fuel_before - 1
        )

    def test_location_resources_replenish_periodically(self):
        world = build_default_world()
        location = world.locations["A"]
        self.assertEqual(location.resources[ResourceType.FUEL], 8)
        self.assertTrue(location.remove_resource(ResourceType.FUEL, 6))
        self.assertEqual(location.resources[ResourceType.FUEL], 2)

        world.advance_time(24)
        self.assertEqual(location.resources[ResourceType.FUEL], 2)
        world.advance_time(1)
        self.assertEqual(location.resources[ResourceType.FUEL], 8)

    def test_self_send_is_rejected(self):
        world = build_default_world()
        hub = CommunicationHub()
        self.assertIsNone(
            hub.send(
                world,
                "Rescue-1",
                "Rescue-1",
                MessageType.OFFER,
                "self-send",
            )
        )
        self.assertEqual(len(hub.messages), 0)

    def test_pickup_numeric_precondition_uses_runtime_kb_value(self):
        kb = KnowledgeBase([
            Fact("at", ("Rescue-1", "C")),
            Fact("resource_at", ("C", "medicine", 1)),
            Fact("capacity", ("Rescue-1", 2)),
            Fact("inventory", ("Rescue-1", "medicine", 0)),
        ])
        action = PICK_UP.bind(**{
            "?agent": "Rescue-1",
            "?loc": "C",
            "?resource": "medicine",
            "?available": 2,
        })
        self.assertTrue(action.preconditions_hold(kb))

    def test_deliver_requires_goal_target_location(self):
        kb = KnowledgeBase([
            Fact("at", ("Rescue-1", "C")),
            Fact("inventory", ("Rescue-1", "medicine", 1)),
            Fact("goal_target", ("Rescue-1", "D")),
        ])
        action = DELIVER.bind(**{
            "?agent": "Rescue-1",
            "?loc": "C",
            "?resource": "medicine",
        })
        self.assertFalse(action.preconditions_hold(kb))

    def test_detect_invalidation_checks_remaining_plan(self):
        world = build_default_world()
        agent = world.agents["Logistics-1"]
        agent.location = "D"
        agent.knowledge.clear()
        agent.knowledge.add(Fact("at", ("Logistics-1", "D")))
        agent.knowledge.add(Fact("fuel", ("Logistics-1", 10)))
        agent.knowledge.add(Fact("road_open", ("D", "E")))
        agent.knowledge.add(Fact("road_open", ("E", "D")))
        agent.knowledge.add(Fact("road_time", ("D", "E", 1)))
        agent.knowledge.add(Fact("road_time", ("E", "D", 1)))
        agent.knowledge.add(Fact("road_blocked", ("E", "G")))
        agent.knowledge.add(Fact("road_blocked", ("G", "E")))
        agent.plan = [
            MOVE.bind(**{"?agent": "Logistics-1", "?from": "D", "?to": "E"}),
            MOVE.bind(**{"?agent": "Logistics-1", "?from": "E", "?to": "G"}),
        ]
        agent.plan_index = 0
        from replanning import detect_invalidation
        invalid = detect_invalidation(agent, world)
        self.assertIsNotNone(invalid)
        self.assertEqual(invalid.index, 1)
        self.assertIn("step 1", str(invalid))


if __name__ == "__main__":
    unittest.main()