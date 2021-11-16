from collections import defaultdict
from ortools.linear_solver import pywraplp
import math
import matplotlib.pyplot as plt
from pprint import pprint

# from pyomo.environ import ConcreteModel, Var, PositiveReals, Objective, Constraint, maximize, SolverFactory

BIGM = 1000
BIGMQ = 1000
TOTAL_DISTANCE_TRAVELED = "total_dist"
TOTAL_WAITING = "total_waiting"
class Darp:
    def __init__(self, N, P, D, K, Q, L, el, d, q, dist_matrix):
        # data = create_data_model()
        self.N = N # Set of all nodes in graph
        self.P = P # Pickup locations
        self.n = len(P) # Number of requests
        self.D = D # Delivery locations
        self.K = K # Set of vehicles
        self.Q = Q # Capacity of a vehicle
        self.L = L # Max. ride time of a request
        self.el = el # node i time window (earliest, latest)
        self.d = d # service duration at node i
        self.q = q # amount loaded onto vehicle at node i (q_i = q_{n+i})
        self.dist_matrix = dist_matrix
        
        self.N_inbound = defaultdict(set)
        self.N_outbound = defaultdict(set)
        for i in self.N:
            for j in self.N:
                if dist_matrix[i][j] >=0:
                    self.N_outbound[i].add(j)
                    self.N_inbound[j].add(i)

        pprint(self.N_outbound)
        pprint(self.N_inbound)
        
        # Variables
        
        # Create the mip solver with the SCIP backend.
        self.solver = pywraplp.Solver.CreateSolver("SCIP")
        infinity = self.solver.infinity()
        print("inf:", infinity)
        
        # 1 if the kth vehicles goes straight from node i to node j
        self.var_x = {}
        
        # When vehicle k starts visiting node i
        self.var_B = {}
        
        # The load of vehicle k after visiting node i
        self.var_Q = {}
        
        # The ride time of request i on vehicle k
        self.var_L = {}
        
        # Variable declaration
        self.declare_decision_vars()
        self.declare_arrival_vars(infinity)
        self.declare_load_vars()
        self.declare_ridetime_vars()

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
        self.constr_vehicle_starts_empty()
        self.constr_vehicle_ends_empty()
        self.constr_ensure_feasible_vehicle_loads()
        self.constr_vehicle_loads_are_lower_than_vehicles_max_capacities()
        self.set_objective_function()
        self.solve()

    def declare_decision_vars(self):
        for k in self.K:
            self.var_x[k] = {}
            for i in self.N:
                self.var_x[k][i] = {}
                for j in self.N_outbound[i]:
                    self.var_x[k][i][j] = self.solver.IntVar(
                        0, 1, f"x[{k},{i},{j}]"
                    )

    def declare_arrival_vars(self, infinity):
        for k in self.K:
            self.var_B[k] = {}
            for i in self.N:
                print( f"B[{k},{i}]")
                self.var_B[k][i] = self.solver.NumVar(
                    0, infinity, f"B[{k},{i}]"
                )

    def declare_load_vars(self):
        for k in self.K:
            self.var_Q[k] = {}
            for i in self.N:
                self.var_Q[k][i] = self.solver.IntVar(
                    0, self.Q[k], f"Q[{k},{i}]"
                )

    def declare_ridetime_vars(self):
        for k in self.K:
            self.var_L[k] = {}
            for i in self.P:
                self.var_L[k][i] = self.solver.NumVar(
                    0, self.L[i], f"L[{k},{i}]"
                )

    def plot(self):
        pass

    def stats(self):
        print("Number of variables =", self.solver.NumVariables())
        print("Variables = ", self.solver.variables())
        print("Number of constraints = ", self.solver.NumConstraints())
        # print("Constraints = ", list(map(str, self.solver.constraints())))

    def solve(self):
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print("Objective value =", round(self.solver.Objective().Value()))

            print("# Arrivals")
            for k in self.K:
                for i in self.N:
                    
                    print(
                        self.var_B[k][i].name(),
                        " = ",
                        self.var_B[k][i].solution_value(),
                    )
            
            print("# Loads")
            for k in self.K:
                for i in self.N:
                    
                    print(
                        self.var_Q[k][i].name(),
                        " = ",
                        self.var_Q[k][i].solution_value(),
                    )
            
            print("Ride times:")
            for k in self.K:
                for i in self.P:
                    print(
                        self.var_L[k][i].name(),
                        " = ",
                        self.var_L[k][i].solution_value(),
                    )
            
            
            print("Flow:")    
            for k in self.K:
                for i in self.N:
                    for j in self.N_outbound[i]:
                        if self.var_x[k][i][j].solution_value() > 0:
                            print(
                                self.var_x[k][i][j].name(),
                                " = ",
                                self.var_x[k][i][j].solution_value(),
                            )

            print(
                "Problem solved in %f milliseconds" % self.solver.wall_time()
            )
            print("Problem solved in %d iterations" % self.solver.iterations())
            print(
                "Problem solved in %d branch-and-bound nodes"
                % self.solver.nodes()
            )
        else:
            print("The problem does not have an optimal solution.")

    def set_objective_function(self, obj=TOTAL_DISTANCE_TRAVELED):
        
        if obj == TOTAL_DISTANCE_TRAVELED:
            
            obj_expr = [
                self.dist_matrix[i][j] * self.var_x[k][i][j]
                for k in self.K
                for i in self.N
                for j in self.N_outbound[i]
            ]
        
        else:
            obj_expr = [
                self.var_L[k][i] + (self.var_B[k][i] - self.el[i][0])
                for k in self.K
                for i in self.P
            ]

        self.solver.Minimize(self.solver.Sum(obj_expr))

    def constr_every_request_is_served_exactly_once(self):
        for i in self.P:
            constr_label = f"request_{i}_is_served_exactly_once"
            print(constr_label)
            self.solver.Add(
                sum(self.var_x[k][i][j] for k in self.K for j in self.N_outbound[i]) == 1,
                constr_label,
            )

    def constr_same_vehicle_services_pickup_and_delivery(self):
        for k in self.K:
            for idx_i, i in enumerate(self.P):
                dest_i = self.N[self.n + idx_i + 1]
                constr_label = f"vehicle_{k}_services_pickup={i}_and_delivery={dest_i}"
                print(constr_label)
                self.solver.Add(
                    sum(self.var_x[k][i][j] for j in self.N_outbound[i])
                    - sum(self.var_x[k][dest_i][j] for j in self.N_outbound[dest_i])
                    == 0,
                    constr_label,
                )

    def constr_every_vehicle_leaves_the_start_terminal(self):
        start_terminal = self.N[0]
        for k in self.K:
            constr_label = f"vehicle_{k}_leaves_start_terminal_{start_terminal}"
            self.solver.Add(
                sum(self.var_x[k][start_terminal][j] for j in self.N_outbound[start_terminal]) == 1,
                constr_label,
            )

    def constr_the_same_vehicle_that_enters_a_node_leaves_the_node(self):
        for k in self.K:
            for i in self.P + self.D:
                constr_label = f"vehicle_{k}_enters_and_leaves_{i}"
                print(constr_label)
                
                self.solver.Add(
                    sum(self.var_x[k][j][i] for j in self.N_inbound[i])
                    - sum(self.var_x[k][i][j] for j in self.N_outbound[i])
                    == 0,
                    constr_label,
                )

    def constr_every_vehicle_enters_the_end_terminal(self):
        end_terminal = self.N[-1]
        for k in self.K:
            constr_label = f"vehicle_{k}_enters_the_end_terminal_{end_terminal}"
            print(constr_label)
            self.solver.Add(
                sum(self.var_x[k][j][end_terminal] for j in self.N_inbound[end_terminal]) == 1,
                constr_label,
            )

    def constr_ensure_feasible_visit_times(self):
        for k in self.K:
            for i in self.N:
                for j in self.N_outbound[i]:
                    
                    constr_label = (
                        f"vehicle_{k}_arrives_at_{j}"
                        f"_after_arrival_at_{i}_plus_"
                        f"service={self.d[i]}_and_t={self.dist_matrix[i][j]})"
                    )
                    print(constr_label)
                    self.solver.Add(
                        self.var_B[k][j]
                        >= self.var_B[k][i]
                        + self.d[i]
                        + self.dist_matrix[i][j]
                        - BIGM * (1 - self.var_x[k][i][j]),
                        constr_label,
                    )

    def constr_visit_times_within_requests_tw(self):
        for k in self.K:
            for i in self.N:
                
                earliest_arrival, latest_arrival = self.el[i]
                
                constr_label_earliest = (
                    f"vehicle_{k}_arrives_at_{i}_"
                    f"after_earliest={earliest_arrival}")
                
                
                print(constr_label_earliest)
                
                self.solver.Add(
                    self.var_B[k][i] >= earliest_arrival,
                    constr_label_earliest
                )
                
                constr_label_latest = (
                    f"vehicle_{k}_arrives_at_{i}_"
                    f"before_latest={latest_arrival}")
                
                print(constr_label_latest)
                
                self.solver.Add(
                    self.var_B[k][i] <= latest_arrival,
                    constr_label_latest   
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
                
                print(constr_label)
                
                self.solver.Add(
                    self.var_L[k][i]
                    ==
                    self.var_B[k][dest_i]
                    - (self.var_B[k][i] + self.d[i]),
                    constr_label    
                )

    def constr_ride_times_are_lower_than_request_thresholds(self):
        for k in self.K:
            for i in self.P:
                
                constr_label = (
                    f"{i}_travels_at_most_{[self.L[i]]}_"
                    f"inside_vehicle_{k}"
                )
                
                print(constr_label)
                
                self.solver.Add(
                    self.var_L[k][i] <= self.L[i],
                    constr_label    
                )

    # TODO debug BIGMQ or test uncapacitated version
    def constr_vehicle_starts_empty(self):
        start_terminal = self.N[0]
        for k in self.K:
            constr_label = f"{k}_starts_empty_from_{start_terminal}"
            print(constr_label)
            
            self.solver.Add(
                self.var_Q[k][start_terminal]
                == 0,
                constr_label,
            )
    def constr_vehicle_ends_empty(self):
        end_terminal = self.N[-1]    
        for k in self.K:
            constr_label = f"{k}_ends_empty_at_{end_terminal}"
            print(constr_label)
            
            self.solver.Add(
                self.var_Q[k][end_terminal]
                == 0,
                constr_label,
            )
        
    def constr_ensure_feasible_vehicle_loads(self):
        for k in self.K:
            for i in self.N:
                for j in self.N_outbound[i]:

                    a = (f"increases_by_{abs(self.q[j])}" if self.q[j] > 0 else f"decreases_by_{abs(self.q[j])}")
                    constr_label = (
                        f"load_of_{k}_traveling_from_"
                        f"{i}_to_{j}_{a}"
                    )
                    print(constr_label)
                    
                    wijk = min([2*self.Q[k], 2*self.Q[k]+ self.q[j]])
                    print(wijk, self.q[j])
                    self.solver.Add(
                        self.var_Q[k][j]
                        >= self.var_Q[k][i]
                        + self.q[j]
                        - wijk * (1 - self.var_x[k][i][j]),
                        constr_label,
                    )

    def constr_vehicle_loads_are_lower_than_vehicles_max_capacities(self):
        for k in self.K:
            for i in self.N:
                
                constr_label = (
                    f"load_of_vehicle_{k}_at_{i}_"
                    f"is_lower_than_max_capacity_{self.Q[k]}"
                )
                
                print(constr_label)
                
                self.solver.Add(
                    self.var_Q[k][i] <= self.Q[k],
                    constr_label
                )

    def constr_variables_are_binary(self):
        pass