from ortools.linear_solver import pywraplp
import math
import matplotlib.pyplot as plt

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
    def __init__(self, N, P, D, K, Q, L, el, d, dist_matrix):
        data = create_data_model()
        self.N = N
        self.P = P
        self.D = D
        self.K = K
        self.Q = Q
        self.L = L
        self.el = el
        self.d = d
        self.dist_matrix = dist_matrix

        # Variables
        self.var_x = {}
        self.var_B = {}
        self.var_Q = {}
        self.var_L = {}

        # Create the mip solver with the SCIP backend.
        self.solver = pywraplp.Solver.CreateSolver("SCIP")
        infinity = self.solver.infinity()
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

        self.every_request_is_served_exactly_once()
        self.of()
        self.solve()
        
    def plot(self):
        pass
        

    def stats(self):
        print("Number of variables =", self.solver.NumVariables())
        print("Variables = ", self.solver.variables())
        print("Number of constraints = ", self.solver.NumConstraints())
        print("Constraints = ", list(map(str, self.solver.constraints())))

    def solve(self):
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print("Objective value =", self.solver.Objective().Value())

            for k in self.K:
                for i in self.N:
                    for j in self.N:
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

    def of(self):
        obj_expr = [
            self.dist_matrix[i][j] * self.var_x[k][i][j]
            for k in self.K
            for i in self.N
            for j in self.N
        ]

        self.solver.Minimize(self.solver.Sum(obj_expr))

    def every_request_is_served_exactly_once(self):
        for i in self.P:
            self.solver.Add(
                sum(self.var_x[k][i][j] for k in self.K for j in self.N) == 1,
                f"request_{i}_is_served_exactly_once",
            )

    def same_vehicle_services_pickup_and_delivery(self):
        pass

    def every_vehicle_leaves_the_start_terminal(self):
        pass

    def the_same_vehicle_that_enters_a_node_leaves_the_node(self):
        pass

    def every_vehicle_enters_the_end_terminal(self):
        pass

    def set_visit_times(self):
        ## Linearized
        pass

    def visit_times_within_requests_tw(self):
        pass

    def set_ride_times(self):
        pass

    def ride_times_are_lower_than_request_thresholds(self):
        pass

    def set_vehicle_loads(self):
        pass

    def vehicle_loads_are_lower_than_vehicles_max_capacities(self):
        pass

    def variables_are_binary(self):
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
    BIG = 500
    dist_matrix = {
        "O": {"O": 0, "A": 150, "B": 100, "A'": BIG, "B'": BIG},
        "A": {"O": BIG, "A": 0, "B": 100, "A'": 150, "B'": 100},
        "B": {"O": BIG, "A": 150, "B": 0, "A'": 150, "B'": 300},
        "A'": {"O": BIG, "A": 150, "B": 100, "A'": 0, "B'": 100},
        "B'": {"O": BIG, "A": 150, "B": 100, "A'": 150, "B'": 0},
        # "A": {"A": 0, "A'": 300, "B": 25},
        # "B": {"B": 0, "A'": 400, "B'": 300},
        # "B'": {"B'": 0, "A'": 300, "B'": 300},
        # "A'": {"B": 300, "B'": 25},
    }
    model = Darp(
        N=["O", "A", "B", "A'", "B'"],
        P=["A", "B"],
        D=["A'", "B'"],
        K=["V1"],
        Q={"V1": 1},
        L={"A": 600, "B": 600},
        d={
            "O": 0,
            "A": 0,
            "B": 0,
            "A'": 0,
            "B'": 0,
        },
        el={
            "O": (0, math.inf),
            "A": (0, 180),
            "B": (20, 200),
            "A'": (300, 600),
            "B'": (320, 620),
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
