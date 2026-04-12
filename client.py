# Delivery Dispatch Environment Client

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    DeliveryOptimisationAction,
    DeliveryOptimisationObservation,
    DeliveryState,
    Driver,
    Order,
)


class DeliveryOptimisationEnv(
    EnvClient[DeliveryOptimisationAction, DeliveryOptimisationObservation, DeliveryState]
):
    """
    Client for the Delivery Dispatch Environment.

    Example:
        >>> with DeliveryEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.drivers)
        ...
        ...     action = DeliveryOptimisationAction(assignments=[])
        ...     result = client.step(action)
        ...     print(result.observation.reward)
    """

    # ───────────────────────── Payload Builder ─────────────────────────

    def _step_payload(self, action: DeliveryOptimisationAction) -> Dict:

        return {
            "assignments": action.assignments or [],
            "batch_assignments": action.batch_assignments or [],
            "reposition": action.reposition or [],
        }

    # ───────────────────────── Result Parser ─────────────────────────

    def _parse_result(self, payload: Dict) -> StepResult[DeliveryOptimisationObservation]:

        obs_data = payload.get("observation", {})

        drivers = [
            Driver(**d)
            for d in obs_data.get("drivers", [])
        ]

        orders = [
            Order(**o)
            for o in obs_data.get("pending_orders", [])
        ]

        observation = DeliveryOptimisationObservation(
            time=obs_data.get("time", 0),
            drivers=drivers,
            pending_orders=orders,
            traffic_level=obs_data.get("traffic_level", 0.0),
            weather=obs_data.get("weather", "clear"),
            demand_heatmap=obs_data.get("demand_heatmap", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ───────────────────────── State Parser ─────────────────────────

    def _parse_state(self, payload: Dict) -> DeliveryState:

        drivers = [
            Driver(**d)
            for d in payload.get("drivers", [])
        ]

        orders = [
            Order(**o)
            for o in payload.get("pending_orders", [])
        ]

        return DeliveryState(
            time=payload.get("time", 0),
            drivers=drivers,
            pending_orders=orders,
        )