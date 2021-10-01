from collections import deque
from model.Node import OriginNode, DestinationNode
from model.TimeWindow import TimeWindow
from model.Route import Route
import math


class Vehicle:

    count = 0

    def __init__(
        self,
        origin_id,
        capacity,
        destination_id=None,
        origin_point=None,
        destination_point=None,
        origin_earliest_time=0,
        origin_latest_time=math.inf,
        destination_earliest_time=0,
        destination_latest_time=math.inf,
        alias="",
    ):
        Vehicle.count += 1
        self.id = Vehicle.count
        self.alias = alias if alias else self.id
        self.origin = OriginNode(origin_id, self, point=origin_point)
        self.destination = (
            DestinationNode(destination_id, self, point=destination_point)
            if destination_id
            else None
        )
        self.capacity = capacity
        self.origin_tw = TimeWindow(origin_earliest_time, origin_latest_time)
        self.destination_tw = TimeWindow(destination_earliest_time, destination_latest_time)
        self.origin.arrival = origin_earliest_time
        self.route = Route(self.origin)
        self.alias = (alias if alias else "V"+str(self.id))
        self.passengers = deque(maxlen=self.capacity)
        self.requests = list()

    @property
    def load(self):
        return len(self.passengers)
        
    def visit_nodes(self, *nodes):
        self.route.nodes.extend(nodes)

    def __str__(self) -> str:
        return f"{self.alias}({self.load}/{self.capacity})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def assign(self, request):
        self.requests.append(request)
