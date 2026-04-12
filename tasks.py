from dataclasses import dataclass


@dataclass
class DeliveryTask:
    name: str
    num_nodes: int
    num_drivers: int
    orders_per_step: int
    max_steps: int
    traffic_level: float
    weather: str
    driver_failure_prob: float


TASKS = [
    DeliveryTask(
        name="easy",
        num_nodes=20,
        num_drivers=5,
        orders_per_step=3,
        max_steps=20,
        traffic_level=0.2,
        weather="clear",
        driver_failure_prob=0.02,
    ),
    DeliveryTask(
        name="medium",
        num_nodes=30,
        num_drivers=7,
        orders_per_step=5,
        max_steps=25,
        traffic_level=0.4,
        weather="clear",
        driver_failure_prob=0.05,
    ),
    DeliveryTask(
        name="hard",
        num_nodes=40,
        num_drivers=10,
        orders_per_step=7,
        max_steps=30,
        traffic_level=0.7,
        weather="rain",
        driver_failure_prob=0.1,
    ),
]