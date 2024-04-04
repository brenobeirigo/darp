# E.g.:
# input = 0 D:	455.309 Q:	3 W:	7.1095 T:	53.0065 ...
# output = 0, 455.309, 3, 7.1095, 53.0065, ...
from typing import OrderedDict
from ..model.node import DropoffNode, PickupNode
from collections import OrderedDict
from dataclasses import dataclass
from ..data.instance import Instance
import pandas as pd

# import pandas as pd

VEHICLE_ROUTE_PATTERN_PARRAGH = (
    r"(\d*)[\t ]"
    r"*D:[\t ]*([\d.]*)[\t ]"
    r"*Q:[\t ]*([\d]*)[\t ]"
    r"*W:[\t ]*([\d.]*)[\t ]*"
    r"T:[\t ]*([\d.]*)[\t ]*"
    r"(.*)"
)

# E.g.:
# input = 0 (w: 0; b: 130.294; t: 0; q: 0)
# output = 0, 130.294, 0, 0
NODE_PATTERN_PARRAGH = (
    r"([\d]*)[\t ]*"
    r"\( *w: ([\d.]*); "
    r"*b: ([\d.]*); "
    r"*t: ([\d.]*); "
    r"*q: ([\d.]*)\)[\t ]*"
)


@dataclass
class SolutionNode:
    id: int
    w: float
    b: float
    t: float
    q: float

    def __repr__(self) -> str:
        return (
            f"{self.id:>3} ("
            f"w: {self.w:6.2f}; "
            f"b: {self.b:6.2f}; "
            f"t: {self.t:6.2f}; "
            f"q: {self.q:6.2f})"
        )


@dataclass
class SolutionVehicle:
    id: int
    D: float  # Duration = Departure 1st node - Arrival last node
    Q: float  # Max. load vehicle
    W: float
    W_avg: float  # Avg. waiting at PUDO nodes (vehicle arrived earlier than earliest time)
    T: float
    T_avg: float  # Avg. transit time
    route: list[SolutionNode]
    
    def node_route(self):
        return tuple([n.id for n in self.route])

    def summary(self):
        return f"cost={self.D:6.2f}"

    def __repr__(self):
        visits = " ".join(map(str, self.route))
        return (
            f"{self.id} "
            f"D: {self.D:10.4f} "
            f"Q: {int(self.Q):>2} "
            f"W: {self.W:10.4f} "
            f"T: {self.T:10.4f} "
            f"{visits}"
        )

    def __post_init__(self):
        # 0 for pickups and >= for dropoffs
        self.total_transit = sum([node.t for node in self.route])
        self.total_waiting = sum([node.w for node in self.route])

    def is_arrival_within_tw(self, node_instance, node_solution):
        # Arrives within time window
        if (
            node_solution.b >= node_instance.tw.earliest
            and node_solution.b <= node_instance.tw.latest
        ):
            return True

    def get_total_cost(self, dist_matrix):
        od_pos_pairs = [
            (o.id, d.id) for o, d in zip(self.route[:-1], self.route[1:])
        ]
        cost = sum([dist_matrix[o][d] for o, d in od_pos_pairs])
        # assert round(cost, 2) == round(self.T, 2)
        return cost

    def get_total_duration(self):
        duration = self.route[-1].b - self.route[0].b
        assert round(duration, 2) == round(self.D, 2)
        return duration

    def get_total_transit(self, instance):
        io_node_dict = self.get_ordered_dict_id_io_node_pair(instance)
        pickups = [
            pair["solution"]
            for pair in io_node_dict.values()
            if type(pair["instance"]) is PickupNode
        ]
        dropoffs = [
            pair["solution"]
            for pair in io_node_dict.values()
            if type(pair["instance"]) is DropoffNode
        ]
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

        for node_solution in self.route:
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


@dataclass
class SolutionSummary:
    cost: float
    total_duration: float
    total_waiting: float
    avg_waiting: float
    total_transit: float
    avg_transit: float


@dataclass
class SolutionSolver:
    sol_objvalue: float
    sol_cputime: float
    graph_numedges: int
    graph_numnodes: int
    solver_numconstrs: int
    solver_numvars: int
    solver_numiterations: int
    solver_numnodes: int
    solver_gap: float


@dataclass
class SolutionFleet:
    K: dict[int, SolutionVehicle]
    summary: SolutionSummary
    solver: SolutionSolver


@dataclass
class Solution:
    instance: Instance
    summary: SolutionSummary
    solver_stats: SolutionSolver
    vehicle_routes: dict[SolutionVehicle]
    
    def to_df(self):
        # Instance information
        config_info = {
            'Number of Vehicles': self.instance.config.n_vehicles,
            'Number of Customers': self.instance.config.n_customers,
            'Time Horizon (min)': self.instance.config.time_horizon_min,
            'Vehicle Capacity': self.instance.config.vehicle_capacity,
            'Maximum Ride Time (min)': self.instance.config.maximum_ride_time_min,
            'Maximum Driving Time (min)': self.instance.config.maximum_driving_time_min,
            'Number of Depots': self.instance.config.n_depots,
        }
        config_df = pd.DataFrame([config_info])
        
        summary_df = pd.DataFrame([{
            'Cost': self.summary.cost,
            'Total Duration': self.summary.total_duration,
            'Total Waiting': self.summary.total_waiting,
            'Average Waiting': self.summary.avg_waiting,
            'Total Transit': self.summary.total_transit,
            'Average Transit': self.summary.avg_transit
        }])
        
        # Create a DataFrame for the Solver Statistics
        solver_df = pd.DataFrame([{
            'Objective Value': self.solver_stats.sol_objvalue,
            'CPU Time': self.solver_stats.sol_cputime,
            'Number of Edges': self.solver_stats.graph_numedges,
            'Number of Nodes': self.solver_stats.graph_numnodes,
            'Number of Constraints': self.solver_stats.solver_numconstrs,
            'Number of Variables': self.solver_stats.solver_numvars,
            'Number of Iterations': self.solver_stats.solver_numiterations,
            'Number of Solver Nodes': self.solver_stats.solver_numnodes,
            'Solver Gap': self.solver_stats.solver_gap
        }])
        
        # Combine all DataFrames into one, with a hierarchical column structure
        combined_df = pd.concat([
           config_df,
            summary_df,
            solver_df,
        ], axis=1)
        
        return combined_df

    def __repr__(self):
        return repr(self.summary)
    
    def routes_dict(self):
        return { k:k_route.node_route() for k, k_route in self.vehicle_routes.items()}
    
    def route_df(self, fn_dist: callable):
        # Create initial DataFrame
        df = pd.DataFrame(
            [
                (
                    v.id,
                    n.id,
                    n.w,
                    n.b,
                    n.t,
                    n.q,
                    fn_dist(prev_n.id, n.id) if prev_n else 0,
                )
                for v in self.vehicle_routes.values()
                for prev_n, n in zip([None] + v.route[:-1], v.route)
            ],
            columns=[
                "vehicle_id",
                "id",
                "waiting",
                "arrival",
                "ride_time_delay",
                "vehicle_load",
                "distance_previous",
            ],
        )

        # Round and adjust data types
        df = df.round(2)
        df["vehicle_load"] = df["vehicle_load"].astype("int32")

        # Merge with nodeset_df and calculate departure
        df_m = pd.merge(
            df, self.instance.nodeset_df, on="id", how="left"
        ).round(2)
        df_m["departure"] = df_m["arrival"] + df_m["service_duration"]

        # Reorder columns
        df_m = df_m[
            [
                "vehicle_id",
                "vehicle_load",
                "id",
                "alias",
                "node_type",
                "x",
                "y",
                "distance_previous",
                "waiting",
                "earliest",
                "arrival",
                "latest",
                "service_duration",
                "departure",
            ]
        ]

        return df_m
