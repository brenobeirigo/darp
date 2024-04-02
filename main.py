# %%

from src import parse_instance_from_filepath, Instance, Solution

import src.solver.darp as darp
from time import time
from pprint import pprint
import logging
logging.basicConfig(level=logging.DEBUG)

# logging.basicConfig(level=logging.DEBUG)

# i : Instance = parse_instance_from_filepath("data/raw/darp_instances/darp_cordeau_2006/a2-16")
# i.nodeset_df
# # %%

# model = Darp(**i.get_data())
# print(model.build_solve())

if __name__ == "__main__":
    t_start = time()
    i: Instance = parse_instance_from_filepath(
        "data/raw/darp_instances/darp_cordeau_2006/a4-8"
    )
    # i : Instance = parse_instance_from_filepath("data/raw/darp_instances/darp_cordeau_2003/pr01")
    print("Time to load instance:", time() - t_start)

    t_start = time()
    # model = Darp(**i.get_data())
    model = darp.Darp(i)

    print("Time to initialize the model:", time() - t_start)

    t_start = time()
    model.build()
    print("Time to build the model:", time() - t_start)

    t_start = time()
    sol: Solution = model.solve()
    print("Time to solve the model:", time() - t_start)

    print("\n### Solution summary:")
    pprint(sol)

    print("\n### Solver output:")
    pprint(sol.solver_stats)

    # Create a Pandas DataFrame to exhibit the routes
    df = sol.route_df(fn_dist=model.dist)

    
    for k, data in sol.vehicle_routes.items():
        print(f"\n### Vehicle {k}")
        print(data)

    print(i.nodeset_df)


# %%

# import matplotlib.pyplot as plt
# from src.visualization.route import plot_vehicle_route


# fig, ax = plt.subplots(1)
# vehicle_id = 0
# v = sol.vehicle_routes[vehicle_id]
# print(v)
# plot_vehicle_route(ax,df)
#     ax,
#     sol.route_df,
#     show_arrows=True,
#     show_node_labels=True,
#     linestyle="-",
#     linewidth=0.5,
#     arrowstyle="-|>",
#     title=f"Route vehicle {vehicle_id} ({v.summary()})",
# )

# plt.savefig(f"../reports/figures/route_v{vehicle_id:02}.svg")


# %%
