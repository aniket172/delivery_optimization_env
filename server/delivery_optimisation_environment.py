# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Delivery Optimisation Environment Implementation.
Perfect for testing HTTP server infrastructure.
"""


import random
import numpy as np
import networkx as nx
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        DeliveryOptimisationObservation,
        DeliveryOptimisationAction,
        DeliveryState,
        Driver,
        Order,
    )
    from ..tasks import TASKS
    from ..graders import compute_reward
except ImportError:
    from models import DeliveryOptimisationObservation, DeliveryOptimisationAction, DeliveryState, Driver, Order
    from tasks import TASKS
    from graders import compute_reward




class DeliveryOptimisationEnv(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.task_index = 0
        self._state = None

    def _build_city_graph(self):

        G = nx.random_geometric_graph(
            self.task.num_nodes,
            radius=0.35,
        )

        for u, v in G.edges():
            G[u][v]["travel_time"] = random.uniform(1, 5)

        return G

    def _generate_heatmap(self):

        heat = np.random.rand(self.task.num_nodes)
        heat /= heat.sum()

        return {i: float(heat[i]) for i in range(self.task.num_nodes)}

    def _generate_order(self):

        node = np.random.choice(
            list(self.graph.nodes),
            p=list(self.demand_heatmap.values()),
        )

        return Order(
            id=random.randint(1000, 9999),
            pickup=node,
            drop=random.choice(list(self.graph.nodes)),
            deadline=self.time + random.randint(6, 15),
            value=random.randint(100, 500),
            priority=random.random() < 0.2,
            restaurant_id=random.randint(1, 20),
        )

    def reset(self, seed=None, episode_id=None, **kwargs):

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self.task = TASKS[self.task_index]
        self.task_index = (self.task_index + 1) % len(TASKS)

        self.time = 0

        self.graph = self._build_city_graph()

        self.demand_heatmap = self._generate_heatmap()

        self.drivers = [
           Driver(
                id=i,
                node=random.choice(list(self.graph.nodes)),
                status="idle",
                fuel=1.0,
                busy_until=0,
                destination=None
            )
            for i in range(self.task.num_drivers)
        ]

        self.pending_orders = []

        self.completed = 0
        self.on_time = 0
        self.late = 0
        self.cancelled = 0
        self.batched = 0
        self.priority_success = 0
        self.fuel_used = 0

        self._state = DeliveryState(
            time=self.time,
            drivers=self.drivers,
            pending_orders=self.pending_orders,
        )

        return DeliveryOptimisationObservation(
            time=self.time,
            drivers=self.drivers,
            pending_orders=self.pending_orders,
            traffic_level=self.task.traffic_level,
            weather=self.task.weather,
            demand_heatmap=self.demand_heatmap,
        )

    def _travel_time(self, src, dst):

        try:
            path = nx.shortest_path(
                self.graph,
                src,
                dst,
                weight="travel_time",
            )

            dist = 0

            for i in range(len(path) - 1):
                dist += self.graph[path[i]][path[i + 1]]["travel_time"]

            return dist * (1 + self.task.traffic_level)

        except nx.NetworkXNoPath:
            return 10

    def step(self, action: DeliveryOptimisationAction, **kwargs):
        
        for driver in self.drivers:
            if driver.status == "delivering" and self.time >= driver.busy_until:
                driver.node = driver.destination
                driver.status = "idle"
                driver.destination = None

        for _ in range(self.task.orders_per_step):
            self.pending_orders.append(self._generate_order())

        if action.assignments:

            for a in action.assignments:

                driver = next(
                    (d for d in self.drivers if d.id == a["driver_id"]),
                    None,
                )

                order = next(
                    (o for o in self.pending_orders if o.id == a["order_id"]),
                    None,
                )

                if driver and order and driver.status == "idle":

                    travel = int(self._travel_time(driver.node, order.pickup))

                    driver.status = "delivering"
                    driver.destination = order.drop
                    driver.busy_until = self.time + travel

                    self.fuel_used += travel * 0.1

                    self.completed += 1

                    if self.time + travel <= order.deadline:

                        self.on_time += 1

                        if order.priority:
                            self.priority_success += 1
                    else:
                        self.late += 1

                    self.pending_orders.remove(order)

        if action.batch_assignments:

            for b in action.batch_assignments:

                driver = next(
                    (d for d in self.drivers if d.id == b["driver_id"]),
                    None,
                )

                if not driver:
                    continue

                orders = [
                    o for o in self.pending_orders if o.id in b["orders"]
                ]

                if len(orders) < 2:
                    continue

                if any(o.priority for o in orders):
                    continue

                self.batched += 1

                for order in orders:

                    travel = int(self._travel_time(driver.node, order.pickup))

                    driver.status = "delivering"
                    driver.destination = order.drop
                    driver.busy_until = self.time + travel
                    self.fuel_used += travel * 0.1
                    self.completed += 1

                    self.pending_orders.remove(order)

        if action.reposition:

            for r in action.reposition:

                driver = next(
                    (d for d in self.drivers if d.id == r["driver_id"]),
                    None,
                )

                if driver and driver.status == "idle":
                    dist = self._travel_time(driver.node, r["target_node"])
                    self.fuel_used += dist * 0.05
                    driver.node = r["target_node"]

        remaining = []

        for order in self.pending_orders:

            if self.time > order.deadline:
                self.cancelled += 1
            else:
                remaining.append(order)

        self.pending_orders = remaining

        for driver in self.drivers:

            if random.random() < self.task.driver_failure_prob:
                driver.status = "inactive"

        self.time += 1

        done = self.time >= self.task.max_steps

        reward = compute_reward(
            self.completed,
            self.on_time,
            self.late,
            self.cancelled,
            self.batched,
            self.priority_success,
            self.fuel_used,
        )

        self._state = DeliveryState(
            time=self.time,
            drivers=self.drivers,
            pending_orders=self.pending_orders,
        )

        return DeliveryOptimisationObservation(
            time=self.time,
            drivers=self.drivers,
            pending_orders=self.pending_orders,
            traffic_level=self.task.traffic_level,
            weather=self.task.weather,
            demand_heatmap=self.demand_heatmap,
            reward=reward,
            done=done,
        )

    @property
    def state(self):
        return self._state
