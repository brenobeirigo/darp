# %% [markdown]
# # MILP model

# %%
# BIG = -1

# dist_matrix = {
#     "depot": {"depot": BIG, "1": 150, "2": 100, "1*": BIG, "2*": BIG, "depot*": 0},
#     "1": {"depot": BIG, "1": BIG, "2": 25, "1*": 150, "2*": 100, "depot*": BIG},
#     "2": {"depot": BIG, "1": 150, "2": BIG, "1*": 400, "2*": 300, "depot*": BIG},
#     "1*": {"depot": BIG, "1": 300, "2": 100, "1*": BIG, "2*": 25, "depot*": 0},
#     "2*": {"depot": BIG, "1": 150, "2": 100, "1*": 150, "2*": BIG, "depot*": 0},
#     "depot*": {"depot": BIG, "1": BIG, "2": BIG, "1*": BIG, "2*": BIG, "depot*": BIG},
# }

# %%

import os
import sys
import matplotlib.pyplot as plt
from pprint import pprint

import networkx as nx

dist_matrix = {
    "depot": {"1": 150, "2": 100}, #,  "depot*": 0},
    "1": {"2": 100, "1*": 150, "2*": 100},
    "2": {"1": 150, "1*": 400, "2*": 300},
     "1*": {"2": 100, "2*": 25, "depot": 100},
     "2*": {"1": 150, "1*": 150, "depot": 100},
    #"depot*": {},
}

print(dist_matrix)


#     route = [1, 2 , 1*, 2*]
#     route ids   = [1, 2 , 3 , 4 ]
#        arr.:              150         175          475          500
#       route: [0]---150--->[1]---25--->[2]---400--->[1*]---25--->[2*]
#          tw:           [0  ,180)   [20 ,200)    [300,600)    [320,620)
#     e. arr.:              150         100          450          400



G = nx.DiGraph()

options = {
    "font_size": 10,
    "node_size": 300,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}

G.add_weighted_edges_from(
    (o, d, dist_matrix[o].get(d))
    for o in dist_matrix
    for d in dist_matrix[o]
    # if dist_matrix[o][d] != BIG
)


fig, ax = plt.subplots()
nx.draw_networkx(G, arrows=True, ax=ax, **options)

plt.show()

# %% [markdown]
# ### Example 1 - one request and one vehicle

# %%
TOTAL_HORIZON = 1000

data = dict(
    origin_depot="depot",
    # destination_depot="depot",
    destination_depot="depot*",
    K=["V1"],
    Q={"V1": 6},
    P=["1"],
    D=["1*"],
    L={"1": 600},
    el={
        "depot": (0, TOTAL_HORIZON),
        "depot*": (0, TOTAL_HORIZON),
        "1": (0, 180),
        "1*": (150, 600),
    },
    d={
        "depot": 0,
        "depot*": 0,
        "1": 0,
        "1*": 0},
    q={
        "depot": 0,
        "depot*": 0,
        "1": 4,
        "1*": -4},
    dist_matrix=dist_matrix,
    total_horizon=TOTAL_HORIZON
)

# %%
sys.path.append(os.path.abspath("../"))
print(sys.path)
from src.solver.darp import Darp
model = Darp(**data)
print(model)

# %%
model.build()

# %%

solution = model.solve()
pprint(solution)
# %% [markdown]
# With a new request (2), vehicle picks up two requests:

# %%
data["P"] = ["1", "2"]
data["D"] = ["1*", "2*"]
data["L"]["2"] = 600
data["el"]["2"] = (20, 200)
data["el"]["2*"] = (320, 620)

data["d"]["2"] = 0
data["d"]["2*"] = 0

data["q"]["2"] = 2
data["q"]["2*"] = -2

model = Darp(**data)
model.build()
solution = model.solve()
pprint(solution)

# %% [markdown]
# Adding vehicles `V2` with capacity `6` and vehicle `V3` with capacity `2`:

# %%
data["K"].append("V2")
data["Q"]["V2"] = 4

data["K"].append("V3")
data["Q"]["V3"] = 2

model = Darp(**data)
model.build()
print(model)
solution = model.solve()
pprint(solution)

# %% [markdown]
# With smaller vehicles (capacity 4), the requests cannot be combined:

# %%
data["Q"]["V1"] = 4
data["Q"]["V2"] = 4
print(data)
model = Darp(**data)
solution = model.build_solve()
pprint(solution)
