"""Enumerations used across the RescueNet world model."""

from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ResourceType(Enum):
    MEDICAL_KIT = "medical_kit"
    FOOD = "food"
    MEDICINE = "medicine"
    EQUIPMENT = "equipment"
    FUEL = "fuel"


class TaskType(Enum):
    DELIVER = "deliver"
    RESCUE = "rescue"
    REPAIR = "repair"
    CLEAR_ROAD = "clear_road"
    EVACUATE = "evacuate"
    TRANSPORT = "transport"


class Capability(Enum):
    """What an agent is *able* to do."""

    MEDICAL = "medical"
    TRANSPORT = "transport"
    REPAIR = "repair"