from pydantic import BaseModel
from typing import List, Optional, Dict


class Driver(BaseModel):
    id: int
    node: int
    status: str
    fuel: float
    busy_until: int = 0
    destination: int | None = None

class Order(BaseModel):
    id: int
    pickup: int
    drop: int
    deadline: int
    value: float
    priority: bool
    restaurant_id: int
    

class DeliveryOptimisationObservation(BaseModel):
    time: int
    drivers: List[Driver]
    pending_orders: List[Order]
    traffic_level: float
    weather: str
    demand_heatmap: Dict[int, float]
    reward: Optional[float] = None
    done: bool = False


class DeliveryOptimisationAction(BaseModel):
    assignments: Optional[List[Dict]] = []
    batch_assignments: Optional[List[Dict]] = []
    reposition: Optional[List[Dict]] = []


class DeliveryState(BaseModel):
    time: int
    drivers: List[Driver]
    pending_orders: List[Order]