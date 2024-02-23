from collections import deque
from ..model.node import OriginNode, DestinationNode, NodeInfo
from ..model.Route import Route

# depot_o = parse_node_line(origin_line, node_type=NodeType.O_DEPOT)
#     depot_d = parse_node_line(destination_line, node_type=NodeType.D_DEPOT)
    
#     return Vehicle(
#         o_id,
#         vehicle_capacity,
#         origin_point=o_point,
#         destination_id=d_id,
#         destination_point=d_point,
#         origin_earliest_time=o_earliest,
#         origin_latest_time=o_latest,
#         destination_earliest_time=d_earliest,
#         destination_latest_time=d_latest,
#     )


        
class Vehicle:

    count = 0

    def __init__(
        self,
        node_o: NodeInfo,
        node_d: NodeInfo,
        capacity: int,
        alias="",
        open_trip=False,
    ):
        self.id = Vehicle.count
        self.alias = alias if alias else self.id
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

        self.capacity = capacity
        self.origin_node.arrival = node_o.earliest
        # TODO Logic for adding routes to vehicles
        # self.route = Route(self.origin_node)
        self.alias = alias if alias else f"V{str(self.id)}"
        self.passengers = deque(maxlen=self.capacity)
        self.requests = list()

        Vehicle.count += 1
        
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