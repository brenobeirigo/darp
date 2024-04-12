from collections import defaultdict, OrderedDict
import logging
from pprint import pprint
from gurobipy import GRB, quicksum, Model, Env
import os
import pathlib

logger = logging.getLogger("__main__" + "." + __name__)

OBJ_MIN_TRAVEL_DISTANCE = "Min. Travel Distance"
OBJ_MIN_DRIVING_TIME = "Min. Driving Time"
OBJ_MAX_PROFIT = "Max. Profit"
OBJ_MIN_PROFIT = "Min. Profit"
OBJ_MIN_TRAVEL_COST = "Min. Travel Cost"
OBJ_MAX_TRAVEL_COST = "Max. Travel Cost"
OBJ_MIN_TOTAL_LATENCY = "Min. Total Latency"
OBJ_MIN_FINAL_MAKESPAN = "Min. Final Makespan"
CONSTR_FLEXIBLE_DEPOT = "constr_flexible_depot"
CONSTR_FIXED_DEPOT = "constr_fixed_depot"
MO_MIN_PROFIT_MIN_TRAVEL_COST = "mo_min_profit_min_travel_cost"

from ..data.instance import Instance
from ..data.scenario import ScenarioConfig
from ..solution.Solution import (
    SolutionNode,
    SolutionFleet,
    SolutionSolver,
    SolutionSummary,
    SolutionVehicle,
    Solution,
)


def get_node_set(
    P: list[str], D: list[str], origin_depot: list[str], destination_depot: list[str]
):
    return origin_depot + P + D + destination_depot


def get_arc_set(
    P: list[str],
    D: list[str],
    origin_depot: list[str],
    destination_depot: list[str]
):
    N = get_node_set(P, D, origin_depot, destination_depot)
    n = len(P)  # Number of requests

    origin_pickup_arcs = {(o, j) for j in P for o in origin_depot}

    pd_dp_arcs = [
        (i, j)
        for i in P + D
        for j in P + D
        # No loops
        if i != j
        # Delivery node of a request is visited after pickup node
        and (N.index(i) != n + N.index(j))
    ]

    delivery_terminal_arcs = [
        (i, d)
        for i in D
        for d in destination_depot]

    # Vehicles travel directly to end depot when they are not assign to
    # users (equivalent to not leaving the depot)
    loop_depot_arcs = [
        (o, d)
        for o in origin_depot
        for d in destination_depot]

    return origin_pickup_arcs.union(
        set(pd_dp_arcs),
        set(delivery_terminal_arcs),
        set(loop_depot_arcs)
    )


class Darp:
    def __init__(self, i: Instance, s: ScenarioConfig = None):
        self.instance = i
        dict_data = i.get_data()
        
        # Scenario is defined, overwrite parameters according to 
        # scenario specs
        if s:
            dict_data["K_params"] = {
                k:dict(
                    cost_per_min=s.cost_per_min,
                    cost_per_km=s.cost_per_km,
                    speed_km_h=s.speed_km_h,
                    revenue_per_load_unit=s.revenue_per_load_unit) 
                for k, v in dict_data["K_params"].items()}
            
        self.init(**dict_data)
        self.init_solver()


        # self. = wrapper_time_matrix
    def travel_time_min(self, k, i, j):
        # If speed is not defined, km_min = 1. Then, distance
        # and travel time are equivalent.
        km_min = self.K_params[k]["speed_km_h"]/60
        travel_time_min = self.dist(i,j)/km_min
        return travel_time_min
    
    def init(
        self,
        origin_depot,
        K,
        K_params,
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
        self.destination_depot = destination_depot
        self.dist_matrix = dist_matrix,
        self.total_horizon = total_horizon
        self.K_params = K_params

        # Vehicle data
        self.K = K  # Set of vehicles
        self.Q = Q  # Capacity of a vehicle

        # Request data
        self.L = L  # Max. ride time of a request
        self.d = d  # Service duration at node i
        self.q = q  # Amount loaded onto vehicle at node i (q_i = q_{n+i})
        self.el = el  # Earliest and latest times to reach nodes
        # If destination depot is not set, create an artificial
        # if destination_depot is None:
        #     self.destination_depot = f"{str(origin_depot)}*"
        #     self.q[self.destination_depot] = self.q[self.origin_depot]
        #     self.el[self.destination_depot] = self.el[self.origin_depot]
        # else:
        #     self.destination_depot = destination_depot

        # Graph data
        self.P = P  # Pickup locations
        self.n = len(P)  # Number of requests
        self.n_depots = len(self.origin_depot)
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
            return dist_matrix[i][j]

        self.dist = wrapper_dist_matrix

        # 50km = 60 min
        # dist = x

    def set_time_limit_min(self, limit_in_min):
        self.solver.setParam('TimeLimit', limit_in_min*60)
        return self

    def init_solver(self):
        
        # Create the mip solver with the Gurobi backend
        self.solver = Model("PDPTW")
        env = Env(empty=True)
        env.setParam("OutputFlag",0)
        self.solver.params.OutputFlag = True
        self.solver.params.LogToConsole = False
        self.set_time_limit_min(10)
        self.solution_ = None

    def __str__(self):
        input_data_str = ", ".join(
            f"\n\t{k}={v}" for k, v in vars(self).items()
        )
        return f"{type(self).__name__}({input_data_str})"

    def set_obj(self, obj):
        if obj == OBJ_MIN_TRAVEL_DISTANCE:
            self.set_objfunc_min_distance_traveled()
        elif obj == OBJ_MAX_PROFIT:
            self.set_objfunc_profit(obj=GRB.MAXIMIZE)
        elif obj == OBJ_MIN_PROFIT:
            self.set_objfunc_profit(obj=GRB.MINIMIZE)
        elif obj == OBJ_MIN_TRAVEL_COST:
            self.set_objfunc_travel_cost(obj=GRB.MINIMIZE)
        elif obj == OBJ_MAX_TRAVEL_COST:
            self.set_objfunc_travel_cost(obj=GRB.MAXIMIZE)
        elif obj == OBJ_MIN_TOTAL_LATENCY:
            self.set_objfunc_min_total_latency()
        elif obj == OBJ_MIN_FINAL_MAKESPAN:
            self.set_objfunc_min_total_makespan()
        elif obj == OBJ_MIN_DRIVING_TIME:
            self.set_objfunc_min_driving_time()
        else:
            raise ValueError("Invalid objective function specified.")
            
        return self

    def set_flex_depot(self, flex):
        if not flex:
            self.constr_vehicle_start_and_end_terminal_are_equal()
        return self
        
    def build(self, allow_rejection=False, max_driving_time_h=None):
        self.declare_variables()
        self.set_constraints(allow_rejection=allow_rejection, max_driving_time_h=max_driving_time_h)

    def build_solve(self):
        self.build()
        return self.solve()

    def set_constraints(self, allow_rejection=False, max_driving_time_h=None):
        # Routing constraints
        
        if allow_rejection:
            self.constr_every_request_is_served_exactly_once_or_rejected()
        else:
            self.constr_every_request_is_served_exactly_once()
        
        self.constr_same_vehicle_services_pickup_and_delivery()
        self.constr_the_same_vehicle_that_enters_a_node_leaves_the_node()
        self.constr_every_vehicle_leaves_the_start_terminal()
        self.constr_every_vehicle_enters_the_end_terminal()

        # Time constraints
        self.constr_ensure_feasible_visit_times()
        self.constr_visit_times_within_requests_tw()
        self.constr_ride_times_are_lower_than_request_thresholds()
        self.constr_ensure_feasible_ride_times()
        self.constr_driving_time_between_depots()
        # self.constr_arrival_time_at_depot_only_if_visit()
        if max_driving_time_h:
            self.constr_max_driving_time(max_driving_time_h)

        # Load constraints
        self.constr_ensure_feasible_vehicle_loads()
        self.constr_vehicle_loads_are_lower_than_vehicles_max_capacities()
        self.constr_vehicle_starts_empty()
        self.constr_vehicle_ends_empty()
        self.constr_vehicle_only_visits_valid_nodes()
        
        self.constr_vehicle_only_arrival_departure_depot_times()
        
        self.constr_makespan()

    ####################################################################
    ####################################################################
    ### VARIABLES ######################################################
    ####################################################################
    ####################################################################

    def declare_variables(self):

        ###### VARIABLES ###############################################

        # 1 if the kth vehicles goes straight from node i to node j
        self.var_x = {}

        # When vehicle k starts visiting node i
        self.var_B = {}

        # The load of vehicle k after visiting node i
        self.var_Q = {}

        # The ride time of request i on vehicle k
        self.var_L = {}
        
        self.var_driving_time = {}
        
        self.var_end_working_time_vehicle = {}
        self.var_start_working_time_vehicle = {}

        self.declare_decision_vars()
        self.declare_arrival_vars()
        self.declare_load_vars()
        self.declare_ridetime_vars()
        self.declare_driving_time_vars()
        self.declare_end_working_time_vars()
        self.declare_start_working_time_vars()
        
        # The max arrival time at the depot across the fleet
        self.var_makespan = self.solver.addVar(0, self.total_horizon, vtype=GRB.CONTINUOUS, name="var_makespan")


    def declare_decision_vars(self):
        self.var_x = dict()
        for k in self.K:
            self.var_x[k] = dict()
            for i, j in self.A:
                if i not in self.var_x[k]:
                    self.var_x[k][i] = dict()
                label_var_x = f"x[{k},{i},{j}]"
                self.var_x[k][i][j] = self.solver.addVar(vtype=GRB.BINARY, name=label_var_x)

    def declare_arrival_vars(self):
        for k in self.K:
            self.var_B[k] = {}
            for i in self.N:
                label_var_B = f"B[{k},{i}]"
                self.var_B[k][i] = self.solver.addVar(
                    lb=self.e[i], ub=self.l[i], vtype=GRB.CONTINUOUS, name=label_var_B
                )

    def declare_load_vars(self):
        for k in self.K:
            self.var_Q[k] = {}
            for i in self.N:
                label_var_Q = f"Q[{k},{i}]"
                self.var_Q[k][i] = self.solver.addVar(
                    lb=0, ub=self.Q[k], vtype=GRB.CONTINUOUS, name=label_var_Q
                )

    def declare_ridetime_vars(self):
        for k in self.K:
            self.var_L[k] = {}
            for i in self.P:
                label_var_L = f"L[{k},{i}]"
                # Max ride time if present (i.e., not None)
                max_ride_time = self.L[i] if self.L[i] is not None else float('inf')
                self.var_L[k][i] = self.solver.addVar(
                    lb=0, ub=max_ride_time, vtype=GRB.CONTINUOUS, name=label_var_L
                )
    
    def declare_driving_time_vars(self):
        for k in self.K:
            self.var_driving_time[k] = self.solver.addVar(
                    lb=0, ub=self.total_horizon, vtype=GRB.CONTINUOUS, name=f"driving_t_{k}"
                )
    
    def declare_end_working_time_vars(self):
        for k in self.K:
            self.var_end_working_time_vehicle[k] = self.solver.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name=f"end_working_time_vehicle_{k}"
                )
    
    def declare_start_working_time_vars(self):
        for k in self.K:
            self.var_start_working_time_vehicle[k] = self.solver.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name=f"start_working_time_vehicle_{k}"
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

        self.solver.setObjective(
            quicksum(obj_expr),
            GRB.MINIMIZE
        )

        logger.debug("objective_function_min_total_waiting")
    
    def set_objfunc_min_total_latency(self):
        obj_expr = [
            self.var_B[k][i]
            for k in self.K
            for i in self.P + self.D
        ]

        self.solver.setObjective(
            quicksum(obj_expr),
            GRB.MINIMIZE
        )

        logger.debug("objective_function_min_total_latency")
    
    def set_objfunc_min_total_makespan(self):
        
        self.solver.setObjective(
            self.var_makespan,
            GRB.MINIMIZE
        )
        
        logger.debug("objective_function_min_total_makespan")

    def set_objfunc_min_distance_traveled(self):
        obj_expr = [
            self.dist(i, j) * self.var_x[k][i][j]
            for k in self.K
            for i, j in self.A
        ]

        self.solver.setObjective(
            quicksum(obj_expr),
            GRB.MINIMIZE
        )

        logger.debug("objective_function_min_distance_traveled")
        

    def set_objfunc_travel_cost(self, obj=GRB.MINIMIZE):

        obj_exp = []
        for k in self.K:
            for i, j in self.A:
                total_travel_cost_per_km = (
                    self.K_params[k]["cost_per_km"]
                    * self.dist(i, j)
                    * self.var_x[k][i][j])
                
                obj_exp.append(total_travel_cost_per_km)
                
                logger.debug(
                    f"obj_{k}_travels_{i}-{j}_"
                    f"cost_per_km={self.K_params[k]['cost_per_km']:06.2f}_times_"
                    f"dist={self.dist(i, j):06.2f}_is_"
                    f"{self.dist(i, j) * self.K_params[k]['cost_per_km']:06.2f}"
                )

        self.solver.setObjective(quicksum(obj_exp), obj)

        logger.debug("objective_function_min_cost_per_km")
    
    def set_objfunc_min_driving_time(self):
        
        obj_expr = []
        for k in self.K:
            obj_expr.append(self.var_driving_time[k])
        self.solver.setObjective(quicksum(obj_expr), GRB.MINIMIZE)

        logger.debug("objective_function_min_driving_time")

        
    def set_objfunc_profit(self, obj=GRB.MAXIMIZE):
        
        obj_expr = []
        for k in self.K:
            total_cost_per_min = (self.K_params[k]["cost_per_min"] * self.var_driving_time[k])
            logger.debug(f"obj_driving_{k}_cost_per_min={self.K_params[k]['cost_per_min']:06.2f}")
            obj_expr.append(-total_cost_per_min)

            for i, j in self.A:
                total_travel_cost_per_km = (
                    self.K_params[k]["cost_per_km"]
                    * self.dist(i, j)
                    * self.var_x[k][i][j])
                logger.debug(
                    f"obj_{k}_travels_{i}-{j}_"
                    f"cost_per_km={self.K_params[k]['cost_per_km']:06.2f}_times_"
                    f"dist={self.dist(i, j):06.2f}_is_"
                    f"{self.dist(i, j) * self.K_params[k]['cost_per_km']:06.2f}"
                )
                obj_expr.append(-total_travel_cost_per_km)
                
                
                

                
                revenue_load = 0
                if i in self.P:
                    revenue_load = (
                        self.K_params[k]["revenue_per_load_unit"]
                        * self.var_x[k][i][j]
                        * self.q[i]
                    )
                    logger.debug(f"obj_{k}_picks_{i}_revenue_{self.K_params[k]['revenue_per_load_unit']:06.2f}_load_{self.q[i]}")

                    obj_expr.append(revenue_load)

        self.solver.setObjective(quicksum(obj_expr), obj)

        logger.debug("objective_function_min_distance_traveled")

    ####################################################################
    ####################################################################
    ### CONSTRAINTS ####################################################
    ####################################################################
    ####################################################################

    def constr_driving_time_between_depots(self):
        
        for k in self.K:
            constr_label = f"driving_time_between_depots_vehicle_{k}"
            self.solver.addConstr(
                self.var_driving_time[k]  == self.var_end_working_time_vehicle[k] - self.var_start_working_time_vehicle[k],
                name=constr_label
            )
            
            logger.debug(constr_label)
            
    def constr_max_driving_time(self, max_driving_time_h):
        
        for k in self.K:
            constr_label = f"max_driving_time_vehicle_{k}_lt_{max_driving_time_h*60}min"
            self.solver.addConstr(
                self.var_driving_time[k] <= max_driving_time_h*60,
                name=constr_label
            )
            
            logger.debug(constr_label)
    
    def constr_makespan(self):
        for k in self.K:
            for d in self.destination_depot:
                constr_label = f"makespan_{k}_depot_{d}"
                self.solver.addConstr(
                    self.var_makespan >= self.var_B[k][d],
                    name=constr_label)
                logger.debug(constr_label)
            
    def constr_every_request_is_served_exactly_once(self):
        for i in self.P:
            constr_label = f"request_{i}_is_served_exactly_once"

            self.solver.addConstr(
            quicksum(
                self.var_x[k][i][j]
                for k in self.K
                for j in self.N_outbound[i]
            )
            == 1,
            name=constr_label,
            )

            logger.debug(constr_label)
    
    def constr_every_request_is_served_exactly_once_or_rejected(self):
        for i in self.P:
            constr_label = f"request_{i}_is_served_exactly_once_or_rejected"

            self.solver.addConstr(
            quicksum(
                self.var_x[k][i][j]
                for k in self.K
                for j in self.N_outbound[i]
            )
            <= 1,
            name=constr_label,
            )

            logger.debug(constr_label)

    def constr_same_vehicle_services_pickup_and_delivery(self):
        for k in self.K:
            for idx_i, i in enumerate(self.P):
                dest_i = self.N[self.n + idx_i + self.n_depots]

                constr_label = (
                    f"vehicle_{k}_"
                    f"services_pickup={i}_"
                    f"and_delivery={dest_i}"
                )

                self.solver.addConstr(
                    quicksum(
                        self.var_x[k][i][j]
                        for j in self.N_outbound[i]
                    )
                    - quicksum(
                        self.var_x[k][dest_i][j]
                        for j in self.N_outbound[dest_i]
                    )
                    == 0,
                    name=constr_label,
                )

                logger.debug(constr_label)

    def constr_arrival_time_at_depot_only_if_visit(self):
        for k in self.K:
            for o in self.origin_depot:
                rhs = self.l[o] * quicksum(
                    self.var_x[k][o][i] 
                    for i in self.N_outbound[o]
                )
                
                constr_label_o_visit = (
                    f"vehicle_{k}_arrives_at_o={o}_only_if_visits"
                )
                self.solver.addConstr(
                    self.var_B[k][o] <= rhs, 
                    name=constr_label_o_visit
                )
                
                logger.debug(constr_label_o_visit)

            for d in self.destination_depot:
                
                rhs = self.l[d] * quicksum(
                    self.var_x[k][i][d] 
                    for i in self.N_inbound[d]
                )
                
                constr_label_d_visit = (
                    f"vehicle_{k}_arrives_at_d={d}_only_if_Visits"
                )
                
                self.solver.addConstr(
                    self.var_B[k][d] <= rhs, 
                    name=constr_label_d_visit
                )
                logger.debug(constr_label_d_visit)
    
    def constr_every_vehicle_leaves_the_start_terminal(self):
        for k in self.K:

            constr_label = f"vehicle_{k}_leaves_a_start_terminal"
            self.solver.addConstr(
                quicksum(
                    self.var_x[k][o][j] 
                    for o in self.origin_depot 
                    for j in self.N_outbound[o]
                )
                == 1,
                name = constr_label,
            )

            logger.debug(constr_label)



    def constr_the_same_vehicle_that_enters_a_node_leaves_the_node(self):
        for k in self.K:
            for i in self.P + self.D:
                inbound_sum = quicksum(
                    self.var_x[k][j][i] for j in self.N_inbound[i]
                )
                outbound_sum = quicksum(
                    self.var_x[k][i][j] for j in self.N_outbound[i]
                )
                balance_constraint = inbound_sum - outbound_sum

                constr_label = f"vehicle_{k}_enters_and_leaves_{i}"

                self.solver.addConstr(
                    balance_constraint == 0,
                    name=constr_label,
                )

                logger.debug(constr_label)


    def constr_vehicle_start_and_end_terminal_are_equal(self):

        for o in self.origin_depot:
            for k in self.K:
                d = o + self.n_depots + 2*self.n

                constr_label = (
                    f"vehicle_{k}_"
                    f"departs_from_{o}_and_"
                    f"finishes_at_{d}"
                )

                lhs = quicksum(
                    self.var_x[k][o][j]
                    for j in self.N_outbound[o]
                )

                rhs = quicksum(
                    self.var_x[k][j][d]
                    for j in self.N_inbound[d]
                )
                self.solver.addConstr(lhs == rhs, name=constr_label)

            logger.debug(constr_label)

    def constr_every_vehicle_enters_the_end_terminal(self):
        for k in self.K:
            constr_label = (
                f"vehicle_{k}_"
                f"enters_an_end_terminal"
            )

            inbound_sum = quicksum(
                self.var_x[k][j][d]
                for d in self.destination_depot 
                for j in self.N_inbound[d]
            )
                
                

            self.solver.addConstr(
                inbound_sum == 1, 
                name=constr_label
            )

            logger.debug(constr_label)


    def constr_vehicle_only_arrival_departure_depot_times(self):
        for k in self.K:
            for i, j in self.A:
                if j in self.destination_depot:
                    
                    constr_label = (
                        f"vehicle_{k}_end_working_time_lt_{i}-{j}_arrival"
                    )
                    
                    self.solver.addConstr(
                         self.var_end_working_time_vehicle[k] <= self.var_B[k][j] + self.total_horizon *(1 - self.var_x[k][i][j]),
                        name=constr_label
                    )
                    
                    constr_label = (
                        f"vehicle_{k}_end_working_time_gt_{i}-{j}_arrival"
                    )
                    
                    self.solver.addConstr(
                         self.var_end_working_time_vehicle[k] >= self.var_B[k][j] - self.total_horizon *(1 - self.var_x[k][i][j]),
                        name=constr_label
                    )
                    
                    
                if i in self.origin_depot:
                    
                    constr_label = (
                        f"vehicle_{k}_start_working_time_lt_{i}-{j}_arrival"
                    )
                    
                    self.solver.addConstr(
                         self.var_start_working_time_vehicle[k] <= self.var_B[k][i] + self.total_horizon *(1 - self.var_x[k][i][j]),
                        name=constr_label
                    )
                    
                    constr_label = (
                        f"vehicle_{k}_start_working_time_gt_{i}-{j}_arrival"
                    )
                    
                    self.solver.addConstr(
                         self.var_start_working_time_vehicle[k] >= self.var_B[k][i] - self.total_horizon *(1 - self.var_x[k][i][j]),
                        name=constr_label
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

                    self.solver.addConstr(
                        self.var_x[k][i][j] == 0, 
                        name=constr_label
                    )

                    logger.debug(constr_label)


    def constr_ensure_feasible_visit_times(self):
        for k in self.K:
            for i, j in self.A:
                BIGM_ijk = max(
                    [0, self.l[i] + self.travel_time_min(k, i, j) + self.d[i] - self.e[j]]
                )

                constr_label = (
                    f"vehicle_{k}_arrives_at_{j}"
                    f"_after_arrival_at_{i}_plus_"
                    f"service={self.d[i]:02d}_and_"
                    f"t={self.travel_time_min(k,i,j):06.2f}_"
                    f"BIGM_{BIGM_ijk:06.2f}"
                )

                self.solver.addConstr(
                    self.var_B[k][j]
                    >= self.var_B[k][i]
                    + self.d[i]
                    + self.travel_time_min(k, i, j)
                    - BIGM_ijk * (1 - self.var_x[k][i][j]),
                    name=constr_label,
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

                self.solver.addConstr(
                    self.var_B[k][i] >= self.e[i],
                    name=constr_label_earliest
                )

                constr_label_latest = (
                    f"vehicle_{k}_arrives_at_{i}_" f"before_latest={self.l[i]}"
                )

                logger.debug(constr_label_latest)

                self.solver.addConstr(
                    self.var_B[k][i] <= self.l[i],
                    name=constr_label_latest
                )

    def constr_ensure_feasible_ride_times(self):
        for k in self.K:
            for idx, i in enumerate(self.P):
                dest_i = self.N[idx + self.n + self.n_depots]

                constr_label = (
                    f"set_ride_time_of_{i}(service={self.d[i]})_"
                    f"in_vehicle_{k}_"
                    f"to_reach_{dest_i}"
                )

                self.solver.addConstr(
                    self.var_L[k][i]
                    == self.var_B[k][dest_i] - (self.var_B[k][i] + self.d[i]),
                    name=constr_label,
                )

                logger.debug(constr_label)

    def constr_ride_times_are_lower_than_request_thresholds(self):
        for k in self.K:
            for idx, i in enumerate(self.P):
                dest_i = self.N[self.n + idx + self.n_depots]

                constr_label_lower = (
                    f"trip_from_{i}_to_{dest_i}_inside_vehicle_{k}_"
                    f"lasts_at_least_{round(self.travel_time_min(k, i,dest_i),10)}"
                )

                self.solver.addConstr(
                    self.var_L[k][i] >= self.travel_time_min(k, i, dest_i),
                    name=constr_label_lower,
                )

                logger.debug(constr_label_lower)
                
                
                # Create ride time constraints if maximum
                # ride time is setup (i.e., not None)
                if self.L[i] is not None:

                    constr_label_upper = (
                        f"{i}_travels_at_most_{self.L[i]}_" f"inside_vehicle_{k}"
                    )

                    self.solver.addConstr(
                        self.var_L[k][i] <= self.L[i], name=constr_label_upper
                    )

                    logger.debug(constr_label_upper)

    def constr_vehicle_starts_empty(self):
        for k in self.K:
            for o in self.origin_depot:
                constr_label = f"{k}_starts_empty_from_{o}"

                self.solver.addConstr(
                    self.var_Q[k][o] == 0,
                    name=constr_label,
                )

                logger.debug(constr_label)

    def constr_vehicle_ends_empty(self):
        for d in self.destination_depot:
            for k in self.K:
                constr_label = f"{k}_ends_empty_at_{d}"

                self.solver.addConstr(
                    self.var_Q[k][d] == 0,
                    name=constr_label,
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

                self.solver.addConstr(
                    self.var_Q[k][j]
                    >= self.var_Q[k][i]
                    + self.q[j]
                    - BIGW_ijk * (1 - self.var_x[k][i][j]),
                    name=constr_label,
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

                self.solver.addConstr(
                    self.var_Q[k][i] >= lower_capacity,
                    name=constr_label_lower
                )

                constr_label_upper = (
                    f"load_of_vehicle_{k}_at_{i}_"
                    f"is_lower_than_{upper_capacity}_"
                    f"min({self.Q[k]}_or_{self.Q[k]}_plus_{self.q[i]})"
                )

                logger.debug(constr_label_upper)

                self.solver.addConstr(
                    self.var_Q[k][i] <= upper_capacity,
                    name=constr_label_upper,
                )

    @property
    def solver_gap_(self):
        """
        MIPGap provides the current relative gap between the best objective
        bound and the objective of the best feasible solution.
        """
        try:
            return self.solver.MIPGap
        except:
            return None

    @property
    def solver_numvars_(self):
        """
        NumVars returns the number of variables in the model.
        """
        try:
            return self.solver.NumVars
        except:
            return None

    @property
    def solver_numconstrs_(self):
        """
        NumConstrs returns the number of constraints in the model.
        """
        try:
            return self.solver.NumConstrs
        except:
            return None

    @property
    def sol_objvalue_(self):
        """
        ObjVal returns the objective value of the current solution.
        """
        try:
            return self.solver.ObjVal
        except:
            return None

    @property
    def sol_cputime_(self):
        """
        Runtime returns the optimization time in seconds.
        """
        try:
            return self.solver.Runtime
        except:
            return None

    @property
    def graph_numnodes_(self):
        """
        The number of nodes in the graph is derived from the length of the
        N list.
        """
        try:
            return len(self.N)
        except:
            return None

    @property
    def graph_numedges_(self):
        """
        The number of edges in the graph is derived from the length of the
        A list.
        """
        try:
            return len(self.A)
        except:
            return None

    @property
    def solver_numiterations_(self):
        """
        IterCount returns the number of simplex iterations performed during
        the optimization.
        """
        try:
            return self.solver.IterCount
        except:
            return None

    @property
    def solver_numnodes_(self):
        """
        NodeCount returns the number of MIP nodes explored during the
        optimization.
        """
        try:
            return self.solver.NodeCount
        except:
            return None

    @property
    def solver_objbound_(self):
        """
        Best available objective bound (lower bound for minimization, upper
        bound for maximization)
        """
        try:
            return self.solver.ObjBound
        except:
            return None

    @property
    def solver_work_(self):
        """
        Work spent on the most recent optimization. In contrast to Runtime,
        work is deterministic, meaning that you will get exactly the same
        result every time provided you solve the same model on the same
        hardware with the same parameter and attribute settings.
        The units on this metric are arbitrary. One work unit corresponds
        very roughly to one second on a single thread, but this greatly
        depends on the hardware on which Gurobi is running and the model
        that is being solved.
        """
        try:
            return self.solver.Work
        except:
            return None


    @property
    def sol_(self) -> SolutionSolver:
        # This constructs and returns a SolutionSolver object with the current solution's details.
        return SolutionSolver(
            self.sol_objvalue_,
            self.sol_cputime_,
            self.graph_numedges_,
            self.graph_numnodes_,
            self.solver_numconstrs_,
            self.solver_numvars_,
            self.solver_numiterations_,
            self.solver_numnodes_,
            self.solver_gap_,
            self.solver_work_,
            self.solver_objbound_,
        )


    def var_B_sol(self, k, i):
        # Accessing the solution value of variable B for vehicle k at node i
        return self.var_B[k][i].X
    
    def var_driving_time_sol(self, k):
        # Accessing the solution value of variable driving_time of vehicle k
        return self.var_driving_time[k].X
    
    def var_arrival_depot_sol(self, k):
        return self.var_end_working_time_vehicle[k].X
    
    def var_departure_depot_sol(self, k):
        return self.var_start_working_time_vehicle[k].X

    def var_L_sol(self, k, i):
        # Accessing the solution value of variable L for vehicle k at pickup node i
        return self.var_L[k][i].X

    def var_Q_sol(self, k, i):
        # Accessing the solution value of variable Q for vehicle k after visiting node i
        return self.var_Q[k][i].X

    def var_x_sol(self, k, i, j):
        # Accessing the solution value of variable x indicating if vehicle k travels from node i to j
        return self.var_x[k][i][j].X

    def get_dict_route_vehicle(self, edges)->dict[int, list[int]]:
        dict_vehicle_arcs = defaultdict(lambda: defaultdict(str))
        dict_vehicle_origin = dict()
        for k, i, j in edges:
            dict_vehicle_arcs[k][i] = j
            if i in self.origin_depot:
                dict_vehicle_origin[k] = i

        dict_route_vehicle = dict()
        for k, from_to in dict_vehicle_arcs.items():
            node_id = dict_vehicle_origin[k]
            ordered_list = []

            while node_id in from_to:
                ordered_list.append(node_id)
                node_id = from_to[node_id]

            ordered_list.append(node_id)  # Append the last node
            dict_route_vehicle[k] = ordered_list

        return dict_route_vehicle

    def get_flow_edges(self):
        edges = []
        for k in self.K:
            for i, j in self.A:
                if self.var_x_sol(k, i, j) > 0.9:
                    edges.append((k, i, j))

        return edges

    def get_vehicle_routes_dict(self) -> dict[int, list[int]]:
        flow_edges = self.get_flow_edges()
        return self.get_dict_route_vehicle(flow_edges)

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

        total_cost = 0
        total_transit = 0
        total_waiting = 0
        total_duration = 0

        fleet_solution:dict[int,SolutionVehicle] = dict()
        vehicle_routes_dict = self.get_vehicle_routes_dict()
        serviced_set = set()
        for k in self.K:
            k_route_node_ids = vehicle_routes_dict[k]
            serviced_vehicle = set(k_route_node_ids).intersection(set(n.id for n in self.instance.pickup_nodes))
            serviced_set.update(serviced_vehicle)
            logger.debug(f"Route: {k} - {k_route_node_ids}")

            k_route_node_data = OrderedDict()

            k_total_cost = 0
            k_total_distance_traveled = 0
            k_max_load = 0
            k_edges = list(zip(k_route_node_ids[:-1], k_route_node_ids[1:]))
            arrival = 0
            for i, j in k_edges:
                k_total_cost += self.dist(i, j)
                k_total_distance_traveled += self.dist(i, j)
                
                if i in self.D:
                    pickup_i_index = self.N.index(i) - self.n
                    pickup_i = self.N[pickup_i_index]
                    ride_delay = self.var_L[k][pickup_i].X
                else:
                    ride_delay = 0

                waiting_at_node_i = self.var_B[k][i].X - arrival
                arrival_at_next_node = self.var_B[k][i].X + self.d[i] + self.travel_time_min(k, i, j)

                if self.var_Q[k][i].X > k_max_load:
                    k_max_load = self.var_Q[k][i].X

                k_route_node_data[i] = SolutionNode(
                    id=i,
                    w=waiting_at_node_i,
                    b=self.var_B[k][i].X,
                    t=ride_delay,
                    q=self.var_Q[k][i].X,
                )

                arrival = arrival_at_next_node

            # Arrival at final depot
            final_depot_id = k_route_node_ids[-1]
            k_route_node_data[final_depot_id] = SolutionNode(
                id=final_depot_id,
                w=0,
                b=self.var_B[k][final_depot_id].X,
                t=0,
                q=self.var_Q[k][final_depot_id].X,
            )

            k_total_transit = sum(node.t for node in k_route_node_data.values())
            k_total_waiting = sum(node.w for node in k_route_node_data.values())
            arrival_end_depot = k_route_node_data[final_depot_id].b
            arrival_start_depot = k_route_node_data[k_route_node_ids[0]].b
            k_total_duration = arrival_end_depot - arrival_start_depot

            requests = set(k_route_node_ids).intersection(self.P)
            k_avg_waiting = k_total_waiting / (2 * len(requests)) if requests else 0
            k_avg_transit = k_total_transit / len(requests) if requests else 0

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
                arrival_depot=self.var_end_working_time_vehicle[k].X,
                departure_depot=self.var_start_working_time_vehicle[k].X,
            )
            
        total_latency = sum([
            n.b
            for sol_vehicle in fleet_solution.values()
            for n in sol_vehicle.route
            if n.id in self.D or n.id in self.P
        ])

        summary = SolutionSummary(
            total_distance=total_cost,
            total_duration=total_duration,
            total_waiting=total_waiting,
            avg_waiting=total_waiting / len(self.K),
            total_transit=total_transit,
            avg_transit=total_transit / len(self.K),
            total_latency=total_latency,
            n_serviced=len(serviced_set),
            final_makespan=max(self.var_end_working_time_vehicle[k].X for k in self.K)
        )

        return Solution(
            instance=self.instance,
            summary=summary,
            solver_stats=self.sol_,
            vehicle_routes=fleet_solution,
        )

    def save_log(self, path):
        
        path2 = pathlib.Path(path)
        logger.debug(f"Saving .log at '{path2}'")

        if path2.suffix != ".log":
            raise Exception(f"{path} is not a log file.")
        
        os.makedirs(path2.parent, exist_ok=True)
        self.solver.params.LogFile = path
    
    def save_lp(self, path):
        path2 = pathlib.Path(path)

        logger.debug(f"Saving .lp at '{path2}'")

        if path2.suffix != ".lp":
            raise Exception(f"{path} is not a lp file.")
        
        os.makedirs(path2.parent, exist_ok=True)
        
        self.solver.params.ResultFile = path

    def solve(self) -> Solution:
        
        self.solver.optimize()
        
        print("STATUS: ", self.solver.Status, self.instance.config.label)

        if self.solver.Status in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]:
            
            if  self.solver.SolCount > 0:
                
                self.solution_ = self.get_solution()
                
                logger.debug("\n# Flow variables:")
                flow_edges = self.get_flow_edges()
                for k, i, j in flow_edges:
                    logger.debug(
                        f"{self.var_x[k][i][j].VarName:>40} = "
                        f"{self.var_x[k][i][j].X:<7.2f}"
                    )

                logger.debug("# Routes:")
                dict_vehicle_routes = self.get_dict_route_vehicle(flow_edges)
                logger.debug(dict_vehicle_routes)


                logger.debug("# Driving times:")
                for k in self.K:
                    logger.debug(
                        f"{self.var_driving_time[k].VarName:>40} = "
                        f"{self.var_driving_time[k].X:>7.2f}"
                    )
                
                logger.debug("# Arrivals")
                for k in self.K:
                    for i in self.N:
                        logger.debug(
                            f"{self.var_B[k][i].VarName:>40} = "
                            f"{self.var_B[k][i].X:>7.2f}"
                        )
                
                logger.debug("# Vehicles (departure/arrival)")
                for k in self.K:
                    logger.debug(
                        f"{self.var_start_working_time_vehicle[k].VarName:>40} = "
                        f"{self.var_start_working_time_vehicle[k].X:>7.2f}"
                    )
                    logger.debug(
                        f"{self.var_end_working_time_vehicle[k].VarName:>40} = "
                        f"{self.var_end_working_time_vehicle[k].X:>7.2f}"
                    )

                logger.debug("# Loads")
                for k in self.K:
                    for i in self.N:
                        logger.debug(
                            f"{self.var_Q[k][i].VarName:>40} = "
                            f"{self.var_Q[k][i].X:>7.0f}"
                        )

                logger.debug("# Ride times:")
                for k in self.K:
                    for i in self.P:
                        logger.debug(
                            f"{self.var_L[k][i].VarName:>40} = "
                            f"{self.var_L[k][i].X:>7.2f}"
                        )

                logger.debug("# Problem solved in:")
                logger.debug(f"\t- {self.solver.Runtime:.1f} seconds")
                logger.debug(f"\t- {self.solver.IterCount} iterations")
                logger.debug(f"\t- {self.solver.NodeCount} branch-and-bound nodes")
                logger.debug(f"# Objective value = {self.solver.ObjVal:.2f}")

            
            else:
                self.solution_ = Solution(
                    instance=self.instance,
                    summary=None,
                    solver_stats=self.sol_,
                    vehicle_routes=None,
                )

            
            return self.solution_
            

        else:
            logger.debug("The problem does not have an optimal solution.")
            
            self.solution_ = Solution(
                    instance=self.instance,
                    summary=None,
                    solver_stats=self.sol_,
                    vehicle_routes=None,
                )
            return self.solution_