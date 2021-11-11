from ortools.linear_solver import pywraplp
import math
import matplotlib.pyplot as plt


BIGM = 1000
BIGMQ = 5

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["constraint_coeffs"] = [
        [5, 7, 9, 2, 1],
        [18, 4, -9, 10, 12],
        [4, 7, 3, 8, 5],
        [5, 13, 16, 3, -7],
    ]
    data["bounds"] = [250, 285, 211, 315]
    data["obj_coeffs"] = [7, 8, 2, 9, 6]
    data["num_vars"] = 5
    data["num_constraints"] = 4
    return data


class Darp:
    def __init__(self, N, P, D, K, Q, L, el, d, q, dist_matrix):
        # data = create_data_model()
        self.N = N
        self.P = P
        self.n = len(P)
        self.D = D
        self.K = K
        self.Q = Q
        self.L = L
        self.el = el
        self.d = d
        self.q = q
        self.dist_matrix = dist_matrix

        # Variables
        
        # 1 if the kth vehicles goes straight from node i to node j
        self.var_x = {}
        
        # When vehicle k starts visiting node i
        self.var_B = {}
        
        # The load of vehicle k after visiting node i
        self.var_Q = {}
        
        # The ride time of request i on vehicle k
        self.var_L = {}

        # Create the mip solver with the SCIP backend.
        self.solver = pywraplp.Solver.CreateSolver("SCIP")
        infinity = self.solver.infinity()
        print("inf:", infinity)
        self.var_x = {}

        for k in self.K:
            self.var_x[k] = {}
            for i in self.N:
                self.var_x[k][i] = {}
                for j in self.N:
                    self.var_x[k][i][j] = self.solver.IntVar(
                        0, 1, f"x[{k},{i},{j}]"
                    )

        for k in self.K:
            self.var_B[k] = {}
            for i in self.N:
                print( f"B[{k},{i}]")
                self.var_B[k][i] = self.solver.NumVar(
                    0, infinity, f"B[{k},{i}]"
                )

        for k in self.K:
            self.var_Q[k] = {}
            for i in self.N:
                self.var_Q[k][i] = self.solver.IntVar(
                    0, self.Q[k], f"Q[{k},{i}]"
                )

        for k in self.K:
            self.var_L[k] = {}
            for i in self.P:
                self.var_L[k][i] = self.solver.NumVar(
                    0, self.L[i], f"L[{k},{i}]"
                )

        self.constr_every_request_is_served_exactly_once()
        self.constr_same_vehicle_services_pickup_and_delivery()
        self.constr_every_vehicle_leaves_the_start_terminal()
        self.constr_the_same_vehicle_that_enters_a_node_leaves_the_node()
        self.constr_every_vehicle_enters_the_end_terminal()
        self.constr_ensure_feasible_visit_times()
        self.constr_visit_times_within_requests_tw()
        self.constr_ride_times_are_lower_than_request_thresholds()
        self.constr_ensure_feasible_ride_times()
        self.constr_ensure_feasible_vehicle_loads()
        self.constr_vehicle_loads_are_lower_than_vehicles_max_capacities()
        self.set_objective_function()
        self.solve()
        
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
            print("Objective value =", self.solver.Objective().Value())

            for k in self.K:
                for i in self.N:
                    for j in self.N:
                        if self.var_x[k][i][j].solution_value() > 0:
                        print(
                            self.var_x[k][i][j].name(),
                            " = ",
                            self.var_x[k][i][j].solution_value(),
                        )
            print()
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

    def set_objective_function(self):
        obj_expr = [
            self.dist_matrix[i][j] * self.var_x[k][i][j]
            for k in self.K
            for i in self.N
            for j in self.N
        ]

        self.solver.Minimize(self.solver.Sum(obj_expr))

    def constr_every_request_is_served_exactly_once(self):
        for i in self.P:
            constr_label = f"request_{i}_is_served_exactly_once"
            self.solver.Add(
                sum(self.var_x[k][i][j] for k in self.K for j in self.N) == 1,
                constr_label,
            )

    def constr_same_vehicle_services_pickup_and_delivery(self):
        for k in self.K:
            for idx_i, i in enumerate(self.P):
                dest_i = self.N[self.n + idx_i + 1]
                constr_label = f"vehicle_{k}_services_{i}_and_{dest_i}"
                print(constr_label)
                self.solver.Add(
                    sum(self.var_x[k][i][j] for j in self.N)
                    - sum(self.var_x[k][dest_i][j] for j in self.N)
                    == 0,
                    constr_label,
                )

    def constr_every_vehicle_leaves_the_start_terminal(self):
        start_terminal = self.N[0]
        for k in self.K:
            self.solver.Add(
                sum(self.var_x[k][start_terminal][j] for j in self.N) == 1,
                f"vehicle_{k}_leaves_start_terminal_{start_terminal}",
            )

    def constr_the_same_vehicle_that_enters_a_node_leaves_the_node(self):
        for k in self.K:
            for i in self.P + self.D:
                constr_name = f"vehicle_{k}_enters_and_leaves_{i}"
                print(constr_name)
                self.solver.Add(
                    sum(self.var_x[k][j][i] for j in self.N)
                    - sum(self.var_x[k][i][j] for j in self.N)
                    == 0,
                    constr_name,
                )

    def constr_every_vehicle_enters_the_end_terminal(self):
        end_terminal = self.N[-1]
        for k in self.K:
            self.solver.Add(
                sum(self.var_x[k][j][end_terminal] for j in self.N) == 1,
                f"vehicle_{k}_enters_the_end_terminal_{end_terminal}",
            )

    def constr_ensure_feasible_visit_times(self):
        for k in self.K:
            for i in self.N:
                for j in self.N:
                    constr_label = (
                        f"vehicle_{k}_arrives_at_{j}"
                        f"_after_arrival_at_{i}_plus_"
                        f"t_{self.d[i]}_and_t_{self.dist_matrix[i][j]})"
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
                    f"vehicle_{k}_arrives_at_{i}"
                    f"after_{earliest_arrival}")
                
                self.solver.Add(
                    self.var_B[k][i] >= earliest_arrival,
                    constr_label_earliest
                )
                
                constr_label_latest = (
                    f"vehicle_{k}_arrives_at_{i}"
                    f"before_{latest_arrival}")
                
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
                    f"to_reach{dest_i}"
                )
                
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

                self.solver.Add(
                    self.var_L[k][i] <= self.L[i],
                    constr_label    
                )

    # TODO debug BIGMQ or test uncapacitated version
    def constr_ensure_feasible_vehicle_loads(self):
        for k in self.K:
            for i in self.N:
                for j in self.N:

                    constr_label = (
                        f"load_of_{k}_traveling_from_"
                        f"{i}_to_{j}_"
                        f"is_higher_or_lower_at_{j}_by_{self.q[j]}"
                    )
                    print(constr_label)

                    self.solver.Add(
                        self.var_Q[k][j]
                        >= self.var_Q[k][i]
                        + self.q[j]
                        - BIGMQ * (1 - self.var_x[k][i][j]),
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


def main_darp():

    """
        route = [1, 2 , 1', 2']
    route ids   = [1, 2 , 3 , 4 ]

        arr.:              150         175          475          500
    route: [O]---150--->[A]---25--->[B]---400--->[A']---25--->[B']
        tw:           [0  ,180)   [20 ,200)    [300,600)    [320,620)
    e. arr.:              150         100          450          400

    """
    
    BIG = 5000
    dist_matrix = {
        "O":  {"O": 0,   "A": 150, "B": 100, "A'": BIG, "B'": BIG, "O'": BIG},
        "A":  {"O": BIG, "A": BIG, "B": 25,  "A'": 150, "B'": 100, "O'": BIG},
        "B":  {"O": BIG, "A": 150, "B": BIG, "A'": 400, "B'": 300, "O'": BIG},
        "A'": {"O": BIG, "A": 300, "B": 100, "A'": BIG, "B'": 25,  "O'": 0},
        "B'": {"O": BIG, "A": 150, "B": 100, "A'": 150, "B'": BIG, "O'": 0},
        "O'": {"O": BIG, "A": BIG, "B": BIG, "A'": BIG, "B'": BIG, "O'": 0},
        # "A": {"A": 0, "A'": 300, "B": 25},
        # "B": {"B": 0, "A'": 400, "B'": 300},
        # "B'": {"B'": 0, "A'": 300, "B'": 300},
        # "A'": {"B": 300, "B'": 25},
    }
    
    model = Darp(
        # N=["O", "A", "B", "A'", "B'", "O'"],
        N=["O", "A", "A'"],
        # P=["A", "B"],
        # D=["A'", "B'"],
        P=["A"],
        D=["A'"],
        K=["V1"],
        Q={"V1": 6},
        L={"A": 600, "B": 600},
        d={
            "O": 0,
            "A": 0,
            "B": 0,
            "A'": 0,
            "B'": 0,
            "O'": 0,
        },
        q={
            "O": 0,
            "A": 1,
            "B": 1,
            "A'": -1,
            "B'": -1,
            "O'": 0,
        },
        el={
            "O": (0, BIG),
            "A": (0, 180),
            "B": (20, 200),
            "A'": (300, 600),
            "B'": (320, 620),
            "O'": (0, BIG),
        },
        dist_matrix=dist_matrix,
    )

    model.stats()
    # model = Darp(N=[0,1,2,3,4], P=[1,2], D=[3,4], K=1, Q=1, L=600, el=[(0, math.inf), (0, 180), (20, 200), (300, 600), (320, 620)], dist_matrix=)


def main():
    data = create_data_model()
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    infinity = solver.infinity()
    x = {}
    for j in range(data["num_vars"]):
        x[j] = solver.IntVar(0, infinity, "x[%i]" % j)
    print("Number of variables =", solver.NumVariables())

    for i in range(data["num_constraints"]):
        constraint = solver.RowConstraint(0, data["bounds"][i], "")
        for j in range(data["num_vars"]):
            constraint.SetCoefficient(x[j], data["constraint_coeffs"][i][j])
    print("Number of constraints =", solver.NumConstraints())
    # In Python, you can also set the constraints as follows.
    # for i in range(data['num_constraints']):
    #  constraint_expr = \
    # [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
    #  solver.Add(sum(constraint_expr) <= data['bounds'][i])

    objective = solver.Objective()
    for j in range(data["num_vars"]):
        objective.SetCoefficient(x[j], data["obj_coeffs"][j])
    objective.SetMaximization()
    # In Python, you can also set the objective as follows.
    # obj_expr = [data['obj_coeffs'][j] * x[j] for j in range(data['num_vars'])]
    # solver.Maximize(solver.Sum(obj_expr))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Objective value =", solver.Objective().Value())
        for j in range(data["num_vars"]):
            print(x[j].name(), " = ", x[j].solution_value())
        print()
        print("Problem solved in %f milliseconds" % solver.wall_time())
        print("Problem solved in %d iterations" % solver.iterations())
        print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main_darp()
