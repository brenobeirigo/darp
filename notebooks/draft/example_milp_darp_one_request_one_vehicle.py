from ..src.solver.darp import Darp
from pprint import pprint

dist_matrix = {
    "depot": {"1": 150, "2": 100, "depot": 0},
    "1": {"2": 25, "1*": 150, "2*": 100},
    "2": {"1": 150, "1*": 400, "2*": 300},
    "1*": {"1": 300, "2": 100, "2*": 25, "depot": 0},
    "2*": {"1": 150, "2": 100, "1*": 150, "depot": 0},
}

TOTAL_HORIZON = 1000

data = dict(
    origin_depot="depot",
    K=["V1"],
    Q={"V1": 6},
    P=["1"],
    D=["1*"],
    L={"1": 600},
    el={"depot": (0, TOTAL_HORIZON), "1": (0, 180), "1*": (150, 600)},
    d={"depot": 0, "1": 0, "1*": 0},
    q={"depot": 0, "1": 4, "1*": -4},
    dist_matrix=dist_matrix,
    total_horizon=TOTAL_HORIZON,
)

model = Darp(**data)
model.build()
print(model)
result = model.solve()
pprint(result)
