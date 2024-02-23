from collections import defaultdict, OrderedDict
from ortools.linear_solver import pywraplp
import logging

logger = logging.getLogger("__main__" + "." + __name__)


from ..data.instance import Instance
from ..solution.Solution import (
    SolutionNode,
    SolutionFleet,
    SolutionSolver,
    SolutionSummary,
    SolutionVehicle,
    Solution,
)


def get_node_set(
    P: list[str], D: list[str], origin_depot: str, destination_depot: str
):
    return [origin_depot] + P + D + [destination_depot]


def get_arc_set(
    P: list[str], D: list[str], origin_depot: str, destination_depot: str
):
    N = get_node_set(P, D, origin_depot, destination_depot)
    n = len(P)  # Number of requests

    origin_pickup_arcs = {(N[0], j) for j in P}

    pd_dp_arcs = [
        (i, j)
        for i in P + D
        for j in P + D
        # No loops
        if i != j
        # Delivery node of a request is visited after pickup node
        and (N.index(i) != n + N.index(j))
    ]

    delivery_terminal_arcs = [(i, N[2 * n + 1]) for i in D]

    # Vehicles travel directly to end depot when they are not assign to
    # users (equivalent to not leaving the depot)
    loop_depot_arcs = [(N[0], N[2 * n + 1])]

    return origin_pickup_arcs.union(
        set(pd_dp_arcs), set(delivery_terminal_arcs), set(loop_depot_arcs)
    )


class Darp:
    def __init__(self, i: Instance):
        self.instance = i
        self.init(**i.get_data())

    def init(
        self,
        origin_depot,
        K,
        Q,
        P,
        D,
        L,
        el,
        d,
        q,
        dist_matrix,
        total_horizon,
        destination_depot=None,
        open_trip=False,
    ):
        self.origin_depot = origin_depot

        self.total_horizon = total_horizon

        # Vehicle data
        self.K = K  # Set of vehicles
        self.Q = Q  # Capacity of a vehicle

        # Request data
        self.L = L  # Max. ride time of a request
        self.d = d  # Service duration at node i
        self.q = q  # Amount loaded onto vehicle at node i (q_i = q_{n+i})
        self.el = el  # Earliest and latest times to reach nodes

        # If destination depot is not set, create an artificial
        if destination_depot is None:
            self.destination_depot = f"{str(origin_depot)}*"
            self.q[self.destination_depot] = self.q[self.origin_depot]
            self.el[self.destination_depot] = self.el[self.origin_depot]
        else:
            self.destination_depot = destination_depot

        # Graph data
        self.P = P  # Pickup locations
        self.n = len(P)  # Number of requests
        self.D = D  # Delivery locations
        # Node set
        self.N = get_node_set(
            self.P, self.D, self.origin_depot, self.destination_depot
        )
        # Arc set
        self.A = get_arc_set(
            self.P, self.D, self.origin_depot, self.destination_depot
        )

        # Dictionary of node earliest times
        self.e = {node_id: el[node_id][0] for node_id in self.N}

        # Dictionary of node latest times
        self.l = {node_id: el[node_id][1] for node_id in self.N}

        self.N_inbound = defaultdict(set)
        self.N_outbound = defaultdict(set)
        self.K_N_valid = defaultdict(set)

        for i, j in self.A:
            self.N_outbound[i].add(j)
            self.N_inbound[j].add(i)

        for k in self.K:
            for i in self.N:
                if self.Q[k] >= abs(self.q[i]):
                    self.K_N_valid[k].add(i)

        def wrapper_dist_matrix(i, j):
            # Trips to dummy aux. depot have no cost
            is_trip_to_aux_depot = j == self.destination_depot and open_trip
            return 0 if is_trip_to_aux_depot else dist_matrix[i][j]

        self.dist = wrapper_dist_matrix

        # Create the mip solver with the SCIP backend
        self.solver = pywraplp.Solver.CreateSolver("SCIP")

        ###### VARIABLES ###############################################

        # 1 if the kth vehicles goes straight from node i to node j
        self.var_x = {}

        # When vehicle k starts visiting node i
        self.var_B = {}

        # The load of vehicle k after visiting node i
        self.var_Q = {}

        # The ride time of request i on vehicle k
        self.var_L = {}

        self.solution_ = None

    def __str__(self):
        input_data_str = ", ".join(
            f"\n\t{k}={v}" for k, v in vars(self).items()
        )
        return f"{type(self).__name__}({input_data_str})"

    def build(self):
        self.declare_variables()
        self.set_constraints()
        self.set_objfunc_min_distance_traveled()

    def build_solve(self):
        self.build()
        return self.solve()

    def set_constraints(self):
        # Routing constraints
        self.constr_every_request_is_served_exactly_once()
        self.constr_same_vehicle_services_pickup_and_delivery()
        self.constr_every_vehicle_leaves_the_start_terminal()
        self.constr_the_same_vehicle_that_enters_a_node_leaves_the_node()
        self.constr_every_vehicle_enters_the_end_terminal()

        # Time constraints
        self.constr_ensure_feasible_visit_times()
        self.constr_visit_times_within_requests_tw()
        self.constr_ride_times_are_lower_than_request_thresholds()
        self.constr_ensure_feasible_ride_times()

        # Load constraints
        self.constr_ensure_feasible_vehicle_loads()
        self.constr_vehicle_loads_are_lower_than_vehicles_max_capacities()
        self.constr_vehicle_starts_empty()
        self.constr_vehicle_ends_empty()
        # self.constr_vehicle_only_visits_valid_nodes()

    ####################################################################
    ####################################################################
    ### VARIABLES ######################################################
    ####################################################################
    ####################################################################

    def declare_variables(self):
        self.declare_decision_vars()
        self.declare_arrival_vars()
        self.declare_load_vars()
        self.declare_ridetime_vars()

    def declare_decision_vars(self):
        self.var_x = dict()
        for k in self.K:
            self.var_x[k] = dict()
            for i, j in self.A:
                if i not in self.var_x[k]:
                    self.var_x[k][i] = dict()
                label_var_x = f"x[{k},{i},{j}]"
                self.var_x[k][i][j] = self.solver.IntVar(0, 1, label_var_x)

    def declare_arrival_vars(self):
        for k in self.K:
            self.var_B[k] = {}
            for i in self.N:
                label_var_B = f"B[{k},{i}]"
                self.var_B[k][i] = self.solver.NumVar(
                    0, self.total_horizon, label_var_B
                )

    def declare_load_vars(self):
        for k in self.K:
            self.var_Q[k] = {}
            for i in self.N:
                label_var_Q = f"Q[{k},{i}]"
                self.var_Q[k][i] = self.solver.IntVar(
                    0, self.Q[k], label_var_Q
                )

    def declare_ridetime_vars(self):
        for k in self.K:
            self.var_L[k] = {}
            for i in self.P:
                label_var_L = f"L[{k},{i}]"
                self.var_L[k][i] = self.solver.NumVar(
                    0, self.L[i], label_var_L
                )

    ####################################################################
    ####################################################################
    ### OBJECTIVE FUNCTION #############################################
    ####################################################################
    ####################################################################

    def set_objfunc_min_total_waiting(self):
        obj_expr = [
            self.var_L[k][i] + (self.var_B[k][i] - self.e[i])
            for k in self.K
            for i in self.P
        ]

        self.solver.Minimize(self.solver.Sum(obj_expr))

        logger.debug("objective_function_min_total_waiting")

    def set_objfunc_min_distance_traveled(self):
        obj_expr = [
            self.dist(i, j) * self.var_x[k][i][j]
            for k in self.K
            for i, j in self.A
        ]

        self.solver.Minimize(self.solver.Sum(obj_expr))

        logger.debug("objective_function_min_distance_traveled")

    ####################################################################
    ####################################################################
    ### CONSTRAINTS ####################################################
    ####################################################################
    ####################################################################

    def constr_every_request_is_served_exactly_once(self):
        for i in self.P:
            constr_label = f"request_{i}_is_served_exactly_once"

            self.solver.Add(
                sum(
                    self.var_x[k][i][j]
                    for k in self.K
                    for j in self.N_outbound[i]
                )
                == 1,
                constr_label,
            )

            logger.debug(constr_label)

    def constr_same_vehicle_services_pickup_and_delivery(self):
        for k in self.K:
            for idx_i, i in enumerate(self.P):
                dest_i = self.N[self.n + idx_i + 1]

                constr_label = (
                    f"vehicle_{k}_"
                    f"services_pickup={i}_"
                    f"and_delivery={dest_i}"
                )

                self.solver.Add(
                    sum(self.var_x[k][i][j] for j in self.N_outbound[i])
                    - sum(
                        self.var_x[k][dest_i][j]
                        for j in self.N_outbound[dest_i]
                    )
                    == 0,
                    constr_label,
                )

                logger.debug(constr_label)

    def constr_every_vehicle_leaves_the_start_terminal(self):
        for k in self.K:
            constr_label = (
                f"vehicle_{k}_" f"leaves_start_terminal_{self.origin_depot}"
            )

            self.solver.Add(
                sum(
                    self.var_x[k][self.origin_depot][j]
                    for j in self.N_outbound[self.origin_depot]
                )
                == 1,
                constr_label,
            )

            logger.debug(constr_label)

    def constr_the_same_vehicle_that_enters_a_node_leaves_the_node(self):
        for k in self.K:
            for i in self.P + self.D:
                constr_label = f"vehicle_{k}_" f"enters_and_leaves_{i}"

                self.solver.Add(
                    sum(self.var_x[k][j][i] for j in self.N_inbound[i])
                    - sum(self.var_x[k][i][j] for j in self.N_outbound[i])
                    == 0,
                    constr_label,
                )

                logger.debug(constr_label)

    def constr_every_vehicle_enters_the_end_terminal(self):
        for k in self.K:
            constr_label = (
                f"vehicle_{k}_"
                f"enters_the_end_terminal_{self.destination_depot}"
            )

            self.solver.Add(
                sum(
                    self.var_x[k][j][self.destination_depot]
                    for j in self.N_inbound[self.destination_depot]
                )
                == 1,
                constr_label,
            )

            logger.debug(constr_label)

    def constr_vehicle_only_visits_valid_nodes(self):
        for k in self.K:
            for i, j in self.A:
                if self.Q[k] < abs(self.q[i]) or self.Q[k] < abs(self.q[j]):
                    constr_label = (
                        f"vehicle_{k}_cannot_travel_"
                        f"edge_{i}({self.q[i]})_{j}({self.q[j]})"
                    )

                    self.solver.Add(self.var_x[k][i][j] == 0, constr_label)

                    logger.debug(constr_label)

    def constr_ensure_feasible_visit_times(self):
        for k in self.K:
            for i, j in self.A:
                BIGM_ijk = max(
                    [0, self.l[i] + self.dist(i, j) + self.d[i] - self.e[j]]
                )

                constr_label = (
                    f"vehicle_{k}_arrives_at_{j}"
                    f"_after_arrival_at_{i}_plus_"
                    f"service={self.d[i]}_and_"
                    f"t={round(self.dist(i,j),1)}_"
                    f"BIGM_{round(BIGM_ijk,1)}"
                )

                self.solver.Add(
                    self.var_B[k][j]
                    >= self.var_B[k][i]
                    + self.d[i]
                    + self.dist(i, j)
                    - BIGM_ijk * (1 - self.var_x[k][i][j]),
                    constr_label,
                )

                logger.debug(constr_label)

    def constr_visit_times_within_requests_tw(self):
        for k in self.K:
            for i in self.N:
                constr_label_earliest = (
                    f"vehicle_{k}_arrives_at_{i}_"
                    f"after_earliest={self.e[i]}"
                )

                logger.debug(constr_label_earliest)

                self.solver.Add(
                    self.var_B[k][i] >= self.e[i], constr_label_earliest
                )

                constr_label_latest = (
                    f"vehicle_{k}_arrives_at_{i}_" f"before_latest={self.l[i]}"
                )

                logger.debug(constr_label_latest)

                self.solver.Add(
                    self.var_B[k][i] <= self.l[i], constr_label_latest
                )

    def constr_ensure_feasible_ride_times(self):
        for k in self.K:
            for idx, i in enumerate(self.P):
                dest_i = self.N[idx + self.n + 1]

                constr_label = (
                    f"set_ride_time_of_{i}(service={self.d[i]})_"
                    f"in_vehicle_{k}_"
                    f"to_reach_{dest_i}"
                )

                self.solver.Add(
                    self.var_L[k][i]
                    == self.var_B[k][dest_i] - (self.var_B[k][i] + self.d[i]),
                    constr_label,
                )

                logger.debug(constr_label)

    def constr_ride_times_are_lower_than_request_thresholds(self):
        for k in self.K:
            for idx, i in enumerate(self.P):
                dest_i = self.N[self.n + idx + 1]

                constr_label_lower = (
                    f"trip_from_{i}_to_{dest_i}_inside_vehicle_{k}_"
                    f"lasts_at_least_{round(self.dist(i,dest_i),1)}"
                )

                self.solver.Add(
                    self.var_L[k][i] >= self.dist(i, dest_i),
                    constr_label_lower,
                )

                logger.debug(constr_label_lower)

                constr_label_upper = (
                    f"{i}_travels_at_most_{self.L[i]}_" f"inside_vehicle_{k}"
                )

                self.solver.Add(
                    self.var_L[k][i] <= self.L[i], constr_label_upper
                )

                logger.debug(constr_label_upper)

    def constr_vehicle_starts_empty(self):
        for k in self.K:
            constr_label = f"{k}_starts_empty_from_{self.origin_depot}"

            self.solver.Add(
                self.var_Q[k][self.origin_depot] == 0,
                constr_label,
            )

            logger.debug(constr_label)

    def constr_vehicle_ends_empty(self):
        for k in self.K:
            constr_label = f"{k}_ends_empty_at_{self.destination_depot}"

            self.solver.Add(
                self.var_Q[k][self.destination_depot] == 0,
                constr_label,
            )

            logger.debug(constr_label)

    def constr_ensure_feasible_vehicle_loads(self):
        for k in self.K:
            for i, j in self.A:
                BIGW_ijk = min([2 * self.Q[k], 2 * self.Q[k] + self.q[j]])

                increase_decrease_str = (
                    f"increases_by_{abs(self.q[j])}"
                    if self.q[j] > 0
                    else f"decreases_by_{abs(self.q[j])}"
                )

                constr_label = (
                    f"load_of_vehicle_{k}_"
                    "traveling_from_"
                    f"{i}_to_{j}_{increase_decrease_str}_"
                    f"BIGW_{round(BIGW_ijk,1)}"
                )

                logger.debug(constr_label)

                self.solver.Add(
                    self.var_Q[k][j]
                    >= self.var_Q[k][i]
                    + self.q[j]
                    - BIGW_ijk * (1 - self.var_x[k][i][j]),
                    constr_label,
                )

    def constr_vehicle_loads_are_lower_than_vehicles_max_capacities(self):
        for k in self.K:
            for i in self.K_N_valid[k]:
                lower_capacity = max(0, self.q[i])
                upper_capacity = min(self.Q[k], self.Q[k] + self.q[i])

                constr_label_lower = (
                    f"load_of_vehicle_{k}_at_{i}_"
                    f"is_greater_than_{lower_capacity}_"
                    f"max(0_or_{self.q[i]})"
                )

                logger.debug(constr_label_lower)

                self.solver.Add(
                    self.var_Q[k][i] >= lower_capacity, constr_label_lower
                )

                constr_label_upper = (
                    f"load_of_vehicle_{k}_at_{i}_"
                    f"is_lower_than_{upper_capacity}_"
                    f"min({self.Q[k]}_or_{self.Q[k]}_plus_{self.q[i]})"
                )

                logger.debug(constr_label_upper)

                self.solver.Add(
                    self.var_Q[k][i] <= upper_capacity,
                    constr_label_upper,
                )

    ####################################################################
    ####################################################################
    ### SOLUTION #######################################################
    ####################################################################
    ####################################################################

    @property
    def solver_numvars_(self):
        return self.solver.NumVariables()

    @property
    def solver_numconstrs_(self):
        return self.solver.NumConstraints()

    @property
    def sol_objvalue_(self):
        return self.solver.Objective().Value()

    @property
    def sol_cputime_(self):
        return self.solver.wall_time()

    @property
    def graph_numnodes_(self):
        return len(self.N)

    @property
    def graph_numedges_(self):
        return len(self.A)

    @property
    def solver_numiterations_(self):
        return self.solver.iterations()

    @property
    def solver_numnodes_(self):
        return self.solver.nodes()

    @property
    def sol_(self) -> SolutionSolver:
        return SolutionSolver(
            self.sol_objvalue_,
            self.sol_cputime_,
            self.graph_numedges_,
            self.graph_numnodes_,
            self.solver_numconstrs_,
            self.solver_numvars_,
            self.solver_numiterations_,
            self.solver_numnodes_,
        )

    def var_B_sol(self, k, i):
        return self.var_B[k][i].solution_value()

    def var_L_sol(self, k, i):
        return self.var_L[k][i].solution_value()

    def var_Q_sol(self, k, i):
        return self.var_Q[k][i].solution_value()

    def var_x_sol(self, k, i, j):
        return self.var_x[k][i][j].solution_value()

    def get_dict_route_vehicle(self, edges):
        dict_vehicle_arcs = defaultdict(lambda: defaultdict(str))
        for k, i, j in edges:
            dict_vehicle_arcs[k][i] = j

        dict_route_vehicle = dict()
        for k, from_to in dict_vehicle_arcs.items():
            node_id = self.origin_depot
            ordered_list = list()

            while True:
                ordered_list.append(node_id)
                next_id = from_to[node_id]
                node_id = next_id
                if node_id not in from_to.keys():
                    ordered_list.append(node_id)
                    break

            dict_route_vehicle[k] = ordered_list

        return dict_route_vehicle

    def get_flow_edges(self):
        edges = []
        for k in self.K:
            for i, j in self.A:
                if self.var_x_sol(k, i, j) > 0.9:
                    edges.append((k, i, j))

        return edges

    def get_vehicle_routes_dict(self):
        flow_edges = self.get_flow_edges()
        dict_vehicle_routes = self.get_dict_route_vehicle(flow_edges)

        return dict_vehicle_routes

    def get_sol_str_classic(self, sol):
        for k, k_data in sol["K"].items():
            node_data_list = []
            for node_id, node_data in k_data["routes"].items():
                node_data_str = (
                    f"{node_id} "
                    f"(w: {node_data['w']:7.3f}; "
                    f"b: {node_data['b']:7.3f}; "
                    f"t: {node_data['t']:7.3f}; "
                    f"q: {round(node_data['q'])})"
                )
                node_data_list.append(node_data_str)

            # vehicle_line = (
            #     f"{k} "
            #     f"D:\t{k_data["D"]:7.3f} "
            #     f"Q:\t{k_data["Q"]:>7} "
            #     f"W: {k_data["W_avg"]:7.3f} "
            #     f"T:\t{k_data["T_avg"]:7.3f}")

        # print(
        #     f"cost: {sol["total_cost"]:7.2f}\n"
        #     f"total duration: {sol["total_duration"]:7.2f}\n"
        #     f"total waiting time: {sol["total_waiting"]:7.2f} average: {sol["avg_waiting"]:7.2f}\n"
        #     f"total transit time: {sol["total_transit"]:7.2f} average: {sol["avg_transit"]:7.2f}\n"
        # )

    def get_solution(self):
        result = dict()
        total_cost = 0
        total_transit = 0
        total_waiting = 0
        total_duration = 0

        fleet_solution = dict()
        vehicle_routes_dict = self.get_vehicle_routes_dict()
        for k in self.K:
            k_route_node_ids = vehicle_routes_dict[k]
            logger.debug("Route:", k, k_route_node_ids)

            k_route_node_data = OrderedDict()

            k_total_cost = 0
            k_max_load = 0
            k_edges = list(zip(k_route_node_ids[:-1], k_route_node_ids[1:]))
            arrival = 0
            for i, j in k_edges:
                k_total_cost += self.dist(i, j)
                ride_delay = 0
                if i in self.D:
                    pickup_i = self.N[self.N.index(i) - self.n]
                    # arrival_at_pickup = self.var_B_sol(k,pickup_i)
                    # departure_at_pickup = arrival_at_pickup + self.d[pickup_i]
                    # ride_delay = self.var_B_sol(k,i) - departure_at_pickup
                    ride_delay = self.var_L_sol(k, pickup_i)
                    # assert ride_delay == self.var_L_sol(k, pickup_i)
                    # print("index:", self.N.index(i), self.n, "from:", pickup_i,  "   to:", i,  "   shortest_delay:", shortest_delay,  "   transit_time:", transit_time,  "   ride_delay:", ride_delay)

                departure_from_node_i = self.var_B_sol(k, i) + self.d[i]
                waiting_at_node_i = self.var_B_sol(k, i) - arrival

                # Update maximum load at vehicle k
                if self.var_Q_sol(k, i) > k_max_load:
                    k_max_load = self.var_Q_sol(k, i)

                k_route_node_data[i] = SolutionNode(
                    id=i,
                    w=waiting_at_node_i,
                    b=self.var_B_sol(k, i),
                    t=ride_delay,
                    q=self.var_Q_sol(k, i),
                )

                # print(f"arrival({i:>6}) = {arrival:8.3f} / {self.var_B_sol(k,i):7.3f} (w:{vehicle_waiting_node:>8.3f}) \t\t departure({i:>6}) = {departure:8.3f} \t\t dist({i:>6},{j:>6}) = {self.dist(i,j):7.3f} / ({total_distance:7.3f}) + d[{i:>6}] = {self.d[i]:7.3f} \t\t ({self.e[i]:7},{self.l[i]:7}) \t\t node w: {node_waiting:7.3f}")
                arrival = departure_from_node_i + self.dist(i, j)

            # Arrival at final depot
            k_route_node_data[k_route_node_ids[-1]] = SolutionNode(
                id=k_route_node_ids[-1],
                w=0,
                b=self.var_B_sol(k, k_route_node_ids[-1]),
                t=0,
                q=self.var_Q_sol(k, k_route_node_ids[-1]),
            )

            k_total_transit = 0
            k_total_waiting = 0
            arrival_end_depot = k_route_node_data[k_route_node_ids[-1]].b
            arrival_start_depot = k_route_node_data[k_route_node_ids[0]].b
            k_total_duration = arrival_end_depot - arrival_start_depot

            for node_id in k_route_node_ids:
                node_data = k_route_node_data[node_id]
                k_total_transit += node_data.t
                k_total_waiting += node_data.w

            requests = set(k_route_node_ids).intersection(self.P)
            k_avg_waiting = (
                k_total_waiting / (2 * len(requests)) if requests else None
            )
            k_avg_transit = (
                k_total_transit / len(requests) if requests else None
            )

            total_transit += k_total_transit
            total_waiting += k_total_waiting
            total_duration += k_total_duration
            total_cost += k_total_cost

            fleet_solution[k] = SolutionVehicle(
                id=k,
                D=k_total_cost,
                Q=k_max_load,
                W=k_total_waiting,
                W_avg=k_avg_waiting,
                T=k_total_transit,
                T_avg=k_avg_transit,
                route=list(k_route_node_data.values()),
            )

        summary = SolutionSummary(
            cost=total_cost,
            total_duration=total_duration,
            total_waiting=total_waiting,
            avg_waiting=total_waiting / len(self.K),
            total_transit=total_transit,
            avg_transit=total_transit / len(self.K),
        )

        return Solution(
            instance=self.instance,
            summary=summary,
            solver_stats=self.sol_,
            vehicle_routes=fleet_solution,
        )

    def solve(self) -> Solution:
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            self.solution_ = self.get_solution()

            logger.debug("# Flow variables:")
            flow_edges = self.get_flow_edges()
            for k, i, j in flow_edges:
                logger.debug(
                    f"{self.var_x[k][i][j].name():>20} = "
                    f"{self.var_x_sol(k, i, j):<7}"
                )

            logger.debug("# Routes:")
            dict_vehicle_routes = self.get_dict_route_vehicle(flow_edges)
            logger.debug(dict_vehicle_routes)

            logger.debug("# Arrivals")
            for k in self.K:
                for i in self.N:
                    logger.debug(
                        f"{self.var_B[k][i].name():>20} = "
                        f"{self.var_B_sol(k,i):>7.2f}"
                    )

            logger.debug("# Loads")
            for k in self.K:
                for i in self.N:
                    logger.debug(
                        f"{self.var_Q[k][i].name():>20} = "
                        f"{self.var_Q_sol(k, i):>7.0f}"
                    )

            logger.debug(" # Ride times:")
            for k in self.K:
                for i in self.P:
                    logger.debug(
                        f"{self.var_L[k][i].name():>20} = "
                        f"{self.var_L[k][i].solution_value():>7.2f}"
                    )

            # 0 D:	455.309 Q:	3 W:	7.1095 T:	53.0065	0 (w: 0; b: 85.2408; t: 0; q: 0) 44 (w: 0; b: 89; t: 0; q: 1) 20 (w: 29.7905; b: 131.026; t: 0; q: 2) 92 (w: 0; b: 142.205; t: 43.2055; q: 1) 38 (w: 0; b: 153.316; t: 0; q: 2) 27 (w: 0; b: 167; t: 0; q: 3) 68 (w: 15.4097; b: 195; t: 53.9744; q: 2) 86 (w: 0; b: 209.645; t: 46.3291; q: 1) 30 (w: 21.2879; b: 248; t: 0; q: 2) 75 (w: 0; b: 261.402; t: 84.4021; q: 1) 15 (w: 58.657; b: 334.401; t: 0; q: 2) 78 (w: 0; b: 348; t: 90; q: 1) 1 (w: 0; b: 359.092; t: 0; q: 2) 25 (w: 18.666; b: 390; t: 0; q: 3) 63 (w: 0; b: 405.759; t: 61.3586; q: 2) 4 (w: 0; b: 418.866; t: 0; q: 3) 52 (w: 0; b: 431.388; t: 2.52195; q: 2) 49 (w: 12.5978; b: 459; t: 89.9084; q: 1) 26 (w: 0; b: 473.387; t: 0; q: 2) 73 (w: 0; b: 488.611; t: 88.6115; q: 1) 74 (w: 0; b: 500.733; t: 17.3457; q: 0) 37 (w: 0; b: 512.889; t: 0; q: 1) 85 (w: 0; b: 528.304; t: 5.41463; q: 0) 0 (w: 0; b: 540.55; t: 0; q: 0)
            # 1 D:	373.95 Q:	3 W:	4.99494 T:	26.9578	0 (w: 0; b: 130.294; t: 0; q: 0) 32 (w: 0; b: 131.129; t: 0; q: 1) 3 (w: 0; b: 147.789; t: 0; q: 2) 51 (w: 0; b: 160.728; t: 2.93888; q: 1) 48 (w: 0; b: 173; t: 0; q: 2) 80 (w: 0; b: 184.506; t: 43.3773; q: 1) 18 (w: 0.02555; b: 195.577; t: 0; q: 2) 14 (w: 0; b: 205.873; t: 0; q: 3) 96 (w: 0; b: 216.958; t: 33.9584; q: 2) 66 (w: 0; b: 228; t: 22.4226; q: 1) 62 (w: 10.3861; b: 255; t: 39.1268; q: 0) 16 (w: 40.219; b: 307.726; t: 0; q: 1) 64 (w: 0; b: 318; t: 0.27413; q: 0) 5 (w: 49.2683; b: 382.143; t: 0; q: 1) 7 (w: 0; b: 393.097; t: 0; q: 2) 53 (w: 0; b: 414.597; t: 22.4541; q: 1) 33 (w: 0; b: 430; t: 0; q: 2) 55 (w: 0; b: 445.518; t: 42.4212; q: 1) 39 (w: 0; b: 462.033; t: 0; q: 2) 81 (w: 0; b: 481.006; t: 41.0057; q: 1) 87 (w: 0; b: 493.632; t: 21.5985; q: 0) 0 (w: 0; b: 504.244; t: 0; q: 0)
            # 2 D:	381.499 Q:	3 W:	24.5507 T:	37.0241	0 (w: 0; b: 54.962; t: 0; q: 0) 12 (w: 0; b: 60.639; t: 0; q: 1) 40 (w: 0; b: 78.7991; t: 0; q: 2) 34 (w: 0; b: 92; t: 0; q: 3) 82 (w: 0; b: 106.008; t: 4.00812; q: 2) 21 (w: 16.6023; b: 134.571; t: 0; q: 3) 88 (w: 0; b: 148.378; t: 59.5793; q: 2) 60 (w: 0; b: 160.639; t: 90; q: 1) 69 (w: 0; b: 172.609; t: 28.0388; q: 0) 17 (w: 228.905; b: 412.506; t: 0; q: 1) 65 (w: 0; b: 426; t: 3.49444; q: 0) 0 (w: 0; b: 436.461; t: 0; q: 0)
            # 3 D:	408.096 Q:	4 W:	7.90828 T:	39.385	0 (w: 0; b: 79.9294; t: 0; q: 0) 42 (w: 0; b: 81.7573; t: 0; q: 1) 36 (w: 0; b: 94.0047; t: 0; q: 2) 29 (w: 0; b: 108.805; t: 0; q: 3) 84 (w: 0; b: 121.356; t: 17.3516; q: 2) 43 (w: 0; b: 133.702; t: 0; q: 3) 77 (w: 0; b: 146.43; t: 27.6245; q: 2) 90 (w: 0; b: 156.712; t: 64.9542; q: 1) 31 (w: 0; b: 168; t: 0; q: 2) 79 (w: 0; b: 179.4; t: 1.39989; q: 1) 91 (w: 0; b: 190.812; t: 47.1093; q: 0) 35 (w: 134.262; b: 337; t: 0; q: 1) 28 (w: 23.9033; b: 373; t: 0; q: 2) 46 (w: 0; b: 384.849; t: 0; q: 3) 23 (w: 0; b: 396.296; t: 0; q: 4) 83 (w: 0; b: 407.798; t: 60.7979; q: 3) 71 (w: 0; b: 420.354; t: 14.0575; q: 2) 2 (w: 0; b: 433.872; t: 0; q: 3) 50 (w: 0; b: 447.352; t: 3.48052; q: 2) 76 (w: 0; b: 458.948; t: 75.948; q: 1) 94 (w: 0; b: 475.976; t: 81.1264; q: 0) 0 (w: 0; b: 488.025; t: 0; q: 0)

            logger.debug("# Problem solved in:")
            logger.debug(f"\t- {self.sol_cputime_:.1f} milliseconds")
            logger.debug(f"\t- {self.solver_numiterations_} iterations")
            logger.debug(f"\t- {self.solver_numnodes_} branch-and-bound nodes")
            logger.debug(f"# Objective value = {self.sol_objvalue_:.2f}")

            return self.solution_

        else:
            logger.debug("The problem does not have an optimal solution.")
            return None
