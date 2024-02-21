from collections import deque
from ..model.node import OriginNode, DestinationNode
from ..model.TimeWindow import TimeWindow
from ..model.Route import Route
import math


class Vehicle:

    count = 0

    def __init__(
        self,
        origin_id,
        capacity,
        origin_point=None,
        origin_earliest_time=0,
        origin_latest_time=math.inf,
        open_trip=False,
        destination_id=None,
        destination_point=None,
        destination_earliest_time=0,
        destination_latest_time=math.inf,
        alias="",
    ):
        self.id = Vehicle.count
        self.alias = alias if alias else self.id
        self.origin_node = OriginNode(origin_id, self, point=origin_point)

        # In some contexts (Open VRP), a vehicle does not return to the
        # depot after servicing the last customer.
        self.open_trip = open_trip

        # If closed trip, destination node is either
        # - The origin node
        # - Another node destination node

        if not open_trip:
            self.destination_node = (
                DestinationNode(destination_id, self, point=destination_point)
                if destination_id is not None
                else DestinationNode(origin_id, self, point=origin_point)
            )
        else:
            self.destination_node = None

        self.capacity = capacity
        self.origin_tw = TimeWindow(origin_earliest_time, origin_latest_time)
        self.destination_tw = TimeWindow(
            destination_earliest_time, destination_latest_time
        )
        self.origin_node.arrival = origin_earliest_time
        self.route = Route(self.origin_node)
        self.alias = alias if alias else f"V{str(self.id)}"
        self.passengers = deque(maxlen=self.capacity)
        self.requests = list()

        Vehicle.count += 1

    @staticmethod
    def cleanup():
        Vehicle.count = 0
    
    def must_return_to_origin(self):
        return self.destination_node != None

    @property
    def load(self):
        return len(self.passengers)

    @property
    def pos(self):
        return self.origin_node.pos
    
    def visit_nodes(self, *nodes):
        self.route.nodes.extend(nodes)

    def __str__(self) -> str:
        return f"{self.alias}({self.load}/{self.capacity})"

    def __repr__(self) -> str:
        return self.__str__()

    def assign(self, request):
        self.requests.append(request)
