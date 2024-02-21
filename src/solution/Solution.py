# E.g.:
# input = 0 D:	455.309 Q:	3 W:	7.1095 T:	53.0065 ...
# output = 0, 455.309, 3, 7.1095, 53.0065, ...
from typing import OrderedDict
from ..model.node import DropoffNode, PickupNode
from collections import OrderedDict

VEHICLE_ROUTE_PATTERN_PARRAGH = (
    "(\d*)[\t ]"
    "*D:[\t ]*([\d.]*)[\t ]"
    "*Q:[\t ]*([\d]*)[\t ]"
    "*W:[\t ]*([\d.]*)[\t ]*"
    "T:[\t ]*([\d.]*)[\t ]*"
    "(.*)"
)

# E.g.:
# input = 0 (w: 0; b: 130.294; t: 0; q: 0)
# output = 0, 130.294, 0, 0
NODE_PATTERN_PARRAGH = (
    "([\d]*)[\t ]*"
    "\( *w: ([\d.]*); "
    "*b: ([\d.]*); "
    "*t: ([\d.]*); "
    "*q: ([\d.]*)\)[\t ]*"
)


class Solution:
    def __init__(
        self,
        cost,
        total_duration,
        total_waiting,
        total_transit,
        avg_waiting=None,
        avg_transit=None,
        vehicle_solutions=None,
    ):
        self.cost = cost
        self.total_duration = total_duration
        self.total_waiting = total_waiting
        self.total_transit = total_transit
        self.vehicle_routes = vehicle_solutions
        self.avg_transit = avg_transit
        self.avg_waiting = avg_waiting

    def __repr__(self):
        avg_w = (
            f"avg_waiting={self.avg_waiting:10.4f}, "
            if self.avg_waiting
            else ""
        )
        avg_t = (
            f", avg_transit={self.avg_transit:10.4f}"
            if self.avg_transit
            else ""
        )

        return (
            "Solution("
            f"total_cost={self.cost:10.4f}, "
            f"total_duration={self.total_duration:10.4f}, "
            f"total_waiting={self.total_waiting:10.4f}, {avg_w}"
            f"total_transit={self.total_transit:10.4f} {avg_t}"
            ")"
        )


class VehicleSolution:
    def __init__(self, id_vehicle, D, Q, W, T, visits, instance=None):
        self.id = id_vehicle
        # Duration = Departure 1st node - Arrival last node
        self.D = D
        # Max. load vehicle
        self.Q = Q
        # Avg. waiting at pickup and delivery nodes (vehicle arrived earlier than earliest time)
        self.W = W
        # Avg. transit time
        self.T = T
        self.visits = visits
        # 0 for pickups and >= for dropoffs
        self.total_transit = sum([node.t for node in self.visits])
        self.total_waiting = sum([node.w for node in self.visits])
        self.instance = instance

    def is_arrival_within_tw(self, node_instance, node_solution):
        # Arrives within time window
        if (
            node_solution.b >= node_instance.tw.earliest
            and node_solution.b <= node_instance.tw.latest
        ):
            return True
    
    def get_total_cost(self, dist_matrix):
        od_pos_pairs = [
            (o.id, d.id)
            for o, d in zip(self.visits[:-1], self.visits[1:])
        ]
        cost = sum([dist_matrix[o][d] for o,d in od_pos_pairs])
        print("cost_r:", cost)
        # assert round(cost, 2) == round(self.T, 2)
        return cost
    
    def get_total_duration(self):
        duration = self.visits[-1].b - self.visits[0].b
        assert round(duration,2) == round(self.D, 2)
        return duration

    def get_total_transit(self, instance):
        io_node_dict = self.get_ordered_dict_id_io_node_pair(instance)
        pickups = [pair["solution"] for pair in io_node_dict.values() if type(pair["instance"]) is PickupNode]
        dropoffs = [pair["solution"] for pair in io_node_dict.values() if type(pair["instance"]) is DropoffNode]
        transit = sum([n.t for n in dropoffs])
        transit2 = sum([n.t for n in pickups])
        # print("transit_delivery_nodes (t):", transit)
        # print("transit_pickup_nodes (t):", transit2)
        return transit
        
        
    def is_route_feasible(self, instance):

        io_node_dict = self.get_ordered_dict_id_io_node_pair(instance)

        total_transit = 0
        total_waiting = 0
        total_cost = 0
        departure = 0
        pairs = list(io_node_dict.values())
        for io in pairs:
            node_instance, node_solution = io["instance"], io["solution"]

            # Arrives within time window
            if not self.is_arrival_within_tw(node_instance, node_solution):
                return False

            if type(node_instance) is DropoffNode:
                dropoff_node = node_instance
                dropoff_solution = node_solution
                request = dropoff_node.request
                pickup_node = request.pickup_node
                shortest_distance = instance.dist_matrix[pickup_node.pos][
                    dropoff_node.pos
                ]
                pickup_sol = io_node_dict[pickup_node.pos]["solution"]
                
                if not self.is_transit_within_limits(
                    pickup_node,
                    pickup_sol,
                    dropoff_node,
                    dropoff_solution,
                    request.max_ride_time,
                    shortest_distance,
                ):
                    return False
                
                total_transit += dropoff_solution.t
                
        return True

    def is_transit_within_limits(
        self,
        pickup_instance,
        pickup_solution,
        dropoff_instance,
        dropoff_solution,
        max_ride_time,
        shortest_distance,
    ):

        earliest_arrival_from_pickup_sol = (
            pickup_solution.b
            + pickup_instance.service_delay
            + shortest_distance
        )

        earliest_arrival_from_pickup = (
            pickup_instance.tw.earliest
            + pickup_instance.service_delay
            + shortest_distance
        )
        # print(
        #     pickup_instance,
        #     dropoff_instance,
        #     "earliest(tw):",
        #     dropoff_instance.tw.earliest,
        #     "earliest(pk):",
        #     round(earliest_arrival_from_pickup),
        #     "earliest(pk_sol):",
        #     round(earliest_arrival_from_pickup_sol),
        #     round(shortest_distance),
        # )

        
        # Although vehicle can arrive earlier from pickup, the
        # earliest time at the destination must always be
        # greater than the best earliest time.
        # assert dropoff_node.tw.earliest >= earliest_arrival_from_pickup

        transit = dropoff_solution.b - earliest_arrival_from_pickup_sol
        print("transit:", transit)
        print("transit2:", dropoff_solution.t)
        if round(transit) >= 0 and transit <= max_ride_time:
            return True

    def get_ordered_dict_id_io_node_pair(self, instance):
        """Associate instance nodes to corresponding solution nodes
        and return a dictionary indexed by node ids.
        """
        io_node_dict = OrderedDict()

        for node_solution in self.visits:

            node_instance = instance.node_id_dict[node_solution.id]

            # Check if node IDs for input and output match
            assert (
                node_solution.id == node_instance.pos
            ), f"Mismatch node IO ids: {node_solution.id} {node_instance.pos}"

            # Associate intance and solution to ids
            io_node_dict[node_solution.id] = {
                "instance": node_instance,
                "solution": node_solution,
            }

        return io_node_dict

    def __repr__(self):
        visits = " ".join(map(str, self.visits))
        return (
            f"{self.id} "
            f"D:   {self.D:10.4f} "
            f"Q: {self.Q:>2} "
            f"W:    {self.W:10.4f} "
            f"T:    {self.T:10.4f} "
            f"{visits}"
        )


class NodeSolution:
    def __init__(self, id_node, w, b, t, q):
        self.id = id_node
        self.w = w
        self.b = b
        self.t = t
        self.q = q

    def __repr__(self):
        return (
            f"{self.id} ("
            f"w: {self.w:6.2f}; "
            f"b: {self.b:6.2f}; "
            f"t: {self.t:6.2f}; "
            f"q: {self.q:6.2f})"
        )
