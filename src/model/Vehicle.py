from collections import deque
from ..model.node import OriginNode, DestinationNode, NodeInfo


def create_vehicle(
    problem_type,
    capacity,
    node_o=None,
    node_d=None,
    cost_per_min=0,
    cost_per_km=1,
    speed_km_h=60,
    revenue_per_load_unit=20,
    alias=None,
    open_trip=False):
    if problem_type == MDVRP:
        return VehicleMDVRP(capacity, cost_per_min, cost_per_km, speed_km_h, revenue_per_load_unit,  alias=alias)
    elif problem_type == CVRP:
        return VehicleCVRP(node_o, node_d, capacity, alias=alias, open_trip=open_trip)
    else:
        raise ValueError("Invalid vehicle type")


MDVRP = "MDVRP"
CVRP = "CVRP"

class Vehicle:
    count = 0

    def __init__(
            self,
        capacity: int,
        cost_per_min=1,
        cost_per_km=1,
        speed_km_h=60,
        revenue_per_load_unit=1,
        alias="",
    ):
        
        self.id = Vehicle.count
        self.capacity = capacity
        self.alias = alias if alias else f"V{str(self.id)}"
        self.passengers = deque(maxlen=self.capacity)
        self.requests = list()

        self.origin_node = None
        self.destination_node = None
        self.node_o = None
        self.node_d = None
        
        self.cost_per_min = cost_per_min
        self.cost_per_km =cost_per_km
        self.speed_km_h =speed_km_h
        self.revenue_per_load_unit =revenue_per_load_unit
        
        Vehicle.count += 1
        self.requests = list()

    @property
    def origin_tw(self):
        return self.node_o.tw

    @property
    def destination_tw(self):
        return self.node_d.tw

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

    # def visit_nodes(self, *nodes):
    #     self.route.nodes.extend(nodes)

    def __str__(self) -> str:
        return f"{self.alias}({self.load}/{self.capacity})"

    def __repr__(self) -> str:
        return self.__str__()

    def assign(self, request):
        self.requests.append(request)



class VehicleMDVRP(Vehicle):
    def __init__(
        self, capacity: int,
        cost_per_min,
        cost_per_km,
        speed_km_h,
        revenue_per_load_unit,
        alias=None,
    ):
        super().__init__(capacity, alias)
        self.cost_per_min=cost_per_min
        self.cost_per_km=cost_per_km
        self.speed_km_h=speed_km_h
        self.revenue_per_load_unit = revenue_per_load_unit
        self.passengers = list()
    
class VehicleCVRP(Vehicle):

    def __init__(
        self,
        node_o: NodeInfo,
        node_d: NodeInfo,
        capacity: int,
        alias=None,
        open_trip=False,
    ):
        super().__init__(capacity, alias=alias)

        self.origin_node = OriginNode(node_o, self)
        self.node_o = node_o
        self.node_d = node_d

        # In some contexts (Open VRP), a vehicle does not return to the
        # depot after servicing the last customer.
        self.open_trip = open_trip

        # If closed trip, destination node is either
        # - The origin node
        # - Another node destination node

        if not open_trip:
            self.destination_node = (
                DestinationNode(node_d, self)
                if node_d is not None
                else DestinationNode(node_o, self)
            )
        else:
            self.destination_node = None

        self.origin_node.arrival = node_o.earliest
        self.passengers = deque(maxlen=self.capacity)
        self.requests = list()