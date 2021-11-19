from collections import defaultdict
from ortools.linear_solver import pywraplp
from pprint import pprint

import logging
logging.basicConfig(level=logging.WARNING) # filename='loop2.log', 
logger = logging.getLogger("darp")

# from pyomo.environ import ConcreteModel, Var, PositiveReals, Objective, Constraint, maximize, SolverFactory

TOTAL_DISTANCE_TRAVELED = "total_dist"
TOTAL_WAITING = "total_waiting"


def get_node_set(P, D, origin_depot, destination_depot):
    return [origin_depot] + P + D + [destination_depot]


def get_arc_set(P, D, origin_depot, destination_depot):

    N = get_node_set(P, D, origin_depot, destination_depot)
    n = len(P)  # Number of requests

    origin_pickup_arcs = set([(N[0], j) for j in P])

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
        set(pd_dp_arcs),
        set(delivery_terminal_arcs),
        set(loop_depot_arcs))


class Darp:
    def __init__(
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

        if destination_depot is None:
            self.destination_depot = str(origin_depot) + "*"
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
            self.P,
            self.D,
            self.origin_depot,
            self.destination_depot)
        # Arc set
        self.A = get_arc_set(
            self.P,
            self.D,
            self.origin_depot,
            self.destination_depot)
        
        # Dictionary of node earliest times
        self.e = {node_id: el[node_id][0] for node_id in self.N}

        # Dictionary of node latest times
        self.l = {node_id: el[node_id][1] for node_id in self.N}

        self.N_inbound = defaultdict(set)
        self.N_outbound = defaultdict(set)
        self.K_N_valid = defaultdict(set)

        for i in self.N:
            for j in self.N:
                if (i,j) in self.A:
                    self.N_outbound[i].add(j)
                    self.N_inbound[j].add(i)

        for k in self.K:
            for i in self.N:
                if self.Q[k] >= abs(self.q[i]):
                    self.K_N_valid[k].add(i)

        def wrapper_dist_matrix(i,j):
            if j == self.destination_depot:
                return 0
            return dist_matrix[i][j]
            
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

    def __str__(self):
        input_data_str = ", ".join(f"\n\t{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({input_data_str})"

    def build(self):
        self.declare_variables()
        self.set_constraints()
        self.set_objective_function()
        
    def build_solve(self):
        self.build()
        self.solve()

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

    def declare_variables(self):
        self.declare_decision_vars()
        self.declare_arrival_vars()
        self.declare_load_vars()
        self.declare_ridetime_vars()

    def declare_decision_vars(self):
        self.var_x = dict()
        for k in self.K:
            self.var_x[k] = dict()
            for i in self.N:
                for j in self.N:
                    if (i,j) in self.A:
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
                self.var_Q[k][i] = self.solver.IntVar(0, self.Q[k], label_var_Q)

    def declare_ridetime_vars(self):
        for k in self.K:
            self.var_L[k] = {}
            for i in self.P:
                label_var_L = f"L[{k},{i}]"
                self.var_L[k][i] = self.solver.NumVar(0, self.L[i], label_var_L)

    def plot(self):
        pass

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
    def summary_sol(self):
        return dict(sol_objvalue=self.sol_objvalue_,
                    sol_cputime=self.sol_cputime_,
                    graph_numedges=self.graph_numedges_,
                    graph_numnodes=self.graph_numnodes_,
                    solver_numconstrs=self.solver_numconstrs_,
                    solver_numvars=self.solver_numvars_,
                    solver_numiterations=self.solver_numiterations_,
                    solver_numnodes=self.solver_numnodes_)
    
    def stats(self):
        print(f"  Number of variables = {self.solver_numvars_}")
        print(f"Number of constraints = {self.solver_numconstrs_}")
        print(   f"   Objective value = {self.sol_objvalue_:.2f}")
        # print(f"Constraints = {list(map(str, self.solver.constraints()))}")
        # print(f"Variables = {self.solver.variables()}")

    def solve(self):

        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:

            logger.info("# Arrivals")
            for k in self.K:
                for i in self.N:
                    logger.info(
                        f"{self.var_B[k][i].name():>20} = "
                        f"{self.var_B[k][i].solution_value():>7.2f}"
                    )

            logger.info("# Loads")
            for k in self.K:
                for i in self.N:

                    logger.info(
                        f"{self.var_Q[k][i].name():>20} = "
                        f"{self.var_Q[k][i].solution_value():>7.0f}"
                    )

            logger.info(" # Ride times:")
            for k in self.K:
                for i in self.P:
                    logger.info(
                        f"{self.var_L[k][i].name():>20} = "
                        f"{self.var_L[k][i].solution_value():>7.2f}"
                    )

            logger.info("# Flow variables:")
            flow_edges = self.get_flow_edges()
            for k,i,j in flow_edges:
                logger.info(
                    f"{self.var_x[k][i][j].name():>20} = "
                    f"{self.var_x[k][i][j].solution_value():<7}"
                )


            logger.info("# Routes:")
            dict_vehicle_routes = self.get_dict_route_vehicle(flow_edges)
            logger.info(dict_vehicle_routes)
            
            logger.info("# Problem solved in:")
            logger.info(f"\t- {self.sol_cputime_:.1f} milliseconds")
            logger.info(f"\t- {self.solver_numiterations_} iterations")
            logger.info(f"\t- {self.solver_numnodes_} branch-and-bound nodes")

            logger.info(f"# Objective value = {self.sol_objvalue_:.2f}")

        else:
            logger.info("The problem does not have an optimal solution.")

    def get_dict_route_vehicle(self, edges):
        
        dict_vehicle_arcs = defaultdict(lambda: defaultdict(str))
        for k,i,j in edges:
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
            for i in self.N:
                for j in self.N_outbound[i]:
                    if self.var_x[k][i][j].solution_value() > 0.9:
                        edges.append((k,i,j))
        return edges

    def set_objective_function(self, obj=TOTAL_DISTANCE_TRAVELED):

        if obj == TOTAL_DISTANCE_TRAVELED:

            obj_expr = [
                self.dist(i,j) * self.var_x[k][i][j]
                for k in self.K
                for i in self.N
                for j in self.N_outbound[i]
            ]

        else:
            obj_expr = [
                self.var_L[k][i] + (self.var_B[k][i] - self.e[i])
                for k in self.K
                for i in self.P
            ]

        self.solver.Minimize(self.solver.Sum(obj_expr))

    def constr_every_request_is_served_exactly_once(self):
        for i in self.P:
            
            constr_label = f"request_{i}_is_served_exactly_once"
            
            self.solver.Add(
                sum(self.var_x[k][i][j]
                    for k in self.K
                    for j in self.N_outbound[i]) == 1,
                constr_label,
            )
            
            logger.info(constr_label)

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
                    - sum(self.var_x[k][dest_i][j] for j in self.N_outbound[dest_i])
                    == 0,
                    constr_label,
                )
                
                logger.info(constr_label)

    def constr_every_vehicle_leaves_the_start_terminal(self):
        
        for k in self.K:
            
            constr_label = (
                f"vehicle_{k}_"
                f"leaves_start_terminal_{self.origin_depot}"
            )
            
            self.solver.Add(
                sum(
                    self.var_x[k][self.origin_depot][j]
                    for j in self.N_outbound[self.origin_depot]
                )
                == 1,
                constr_label,
            )
            
            logger.info(constr_label)

    def constr_the_same_vehicle_that_enters_a_node_leaves_the_node(self):
        
        for k in self.K:
            for i in self.P + self.D:
        
                constr_label = (
                    f"vehicle_{k}_"
                    f"enters_and_leaves_{i}"
                )        

                self.solver.Add(
                    sum(self.var_x[k][j][i] for j in self.N_inbound[i])
                    - sum(self.var_x[k][i][j] for j in self.N_outbound[i])
                    == 0,
                    constr_label,
                )
        
                logger.info(constr_label)


    def constr_every_vehicle_enters_the_end_terminal(self):
        
        for k in self.K:
            
            constr_label = (
                f"vehicle_{k}_"
                f"enters_the_end_terminal_{self.destination_depot}"
            )
            
            self.solver.Add(
                sum(self.var_x[k][j][self.destination_depot]
                    for j in self.N_inbound[self.destination_depot]) == 1,
                constr_label,
            )
            
            logger.info(constr_label)

    def constr_vehicle_only_visits_valid_nodes(self):
        for k in self.K:
            for i in self.N:
                for j in self.N_outbound[i]:
                    if self.Q[k] < abs(self.q[i]) or self.Q[k] < abs(self.q[j]):
                        constr_label = f"vehicle_{k}_cannot_travel_edge_{i}({self.q[i]})_{j}({self.q[j]})"
                        self.solver.Add(
                            self.var_x[k][i][j] == 0,
                            constr_label,
                        )

                        logger.info(constr_label)

    def constr_ensure_feasible_visit_times(self):
        for k in self.K:
            for i,j in self.A:

                    BIGM_ijk = max([0,
                                    self.l[i]
                                    + self.dist(i,j)
                                    + self.d[i]
                                    - self.e[j]])

                    constr_label = (
                        f"vehicle_{k}_arrives_at_{j}"
                        f"_after_arrival_at_{i}_plus_"
                        f"service={self.d[i]}_and_t={round(self.dist(i,j),1)}_"
                        f"BIGM_{round(BIGM_ijk,1)}"
                    )
                    
                    self.solver.Add(
                        self.var_B[k][j]
                        >= self.var_B[k][i]
                        + self.d[i]
                        + self.dist(i,j)
                        - BIGM_ijk * (1 - self.var_x[k][i][j]),
                        constr_label,
                    )
                    
                    logger.info(constr_label)

    def constr_visit_times_within_requests_tw(self):
        for k in self.K:
            for i in self.N:

                constr_label_earliest = (
                    f"vehicle_{k}_arrives_at_{i}_" f"after_earliest={self.e[i]}"
                )

                logger.info(constr_label_earliest)

                self.solver.Add(
                    self.var_B[k][i] >= self.e[i],
                    constr_label_earliest)

                constr_label_latest = (
                    f"vehicle_{k}_arrives_at_{i}_" f"before_latest={self.l[i]}"
                )

                logger.info(constr_label_latest)

                self.solver.Add(
                    self.var_B[k][i] <= self.l[i],
                    constr_label_latest)

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
                
                logger.info(constr_label)

    def constr_ride_times_are_lower_than_request_thresholds(self):
        for k in self.K:
            for idx, i in enumerate(self.P):

                dest_i = self.N[self.n + idx + 1]

                constr_label_lower = (
                    f"trip_from_{i}_to_{dest_i}_inside_vehicle_{k}_"
                    f"lasts_at_least_{round(self.dist(i,dest_i),1)}"
                )

                self.solver.Add(
                    self.var_L[k][i] >= self.dist(i,dest_i),
                    constr_label_lower)

                logger.info(constr_label_lower)
                
                constr_label_upper = (
                    f"{i}_travels_at_most_{self.L[i]}_"
                    f"inside_vehicle_{k}"
                )
                
                self.solver.Add(
                    self.var_L[k][i] <= self.L[i],
                    constr_label_upper)

                logger.info(constr_label_upper)


    def constr_vehicle_starts_empty(self):

        for k in self.K:
            constr_label = f"{k}_starts_empty_from_{self.origin_depot}"

            self.solver.Add(
                self.var_Q[k][self.origin_depot] == 0,
                constr_label,
            )
            
            logger.info(constr_label)

    def constr_vehicle_ends_empty(self):

        for k in self.K:
            constr_label = f"{k}_ends_empty_at_{self.destination_depot}"

            self.solver.Add(
                self.var_Q[k][self.destination_depot] == 0,
                constr_label,
            )

            logger.info(constr_label)

    def constr_ensure_feasible_vehicle_loads(self):
        for k in self.K:
            for i, j in self.A:

                    BIGW_ijk = min([2 * self.Q[k],
                                    2 * self.Q[k] + self.q[j]])

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
                    
                    logger.info(constr_label)

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

                logger.info(constr_label_lower)

                self.solver.Add(
                    self.var_Q[k][i] >= lower_capacity,
                    constr_label_lower
                )

                constr_label_upper = (
                    f"load_of_vehicle_{k}_at_{i}_"
                    f"is_lower_than_{upper_capacity}_"
                    f"min({self.Q[k]}_or_{self.Q[k]}_plus_{self.q[i]})"
                )

                logger.info(constr_label_upper)

                self.solver.Add(
                    self.var_Q[k][i] <= upper_capacity,
                    constr_label_upper,
                )
