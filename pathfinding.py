"""Reusable pathfinding utilities for the world road network."""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

from world import World


def shortest_travel_time(world: World, start: str, goal: str) -> float:
    """Return the cheapest open-road travel cost between two locations."""
    if start == goal:
        return 0.0

    distances: Dict[str, float] = {start: 0.0}
    frontier: List[Tuple[float, str]] = [(0.0, start)]

    while frontier:
        distance, node = heapq.heappop(frontier)
        if distance > distances.get(node, float("inf")):
            continue
        if node == goal:
            return distance

        for neighbor, road in world.neighbors(node):
            if road.blocked:
                continue
            next_distance = distance + road.travel_time + road.risk
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                heapq.heappush(frontier, (next_distance, neighbor))

    return float("inf")
