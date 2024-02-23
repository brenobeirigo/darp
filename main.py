# %%

from src import parse_instance_from_filepath, Instance
import src.solver.darp as darp
from time import time
import logging

# logging.basicConfig(level=logging.DEBUG)

# i : Instance = parse_instance_from_filepath("data/raw/darp_instances/darp_cordeau_2006/a2-16")
# i.nodeset_df
# # %%

# model = Darp(**i.get_data())
# print(model.build_solve())

if __name__ == "__main__":
    t_start = time()
    i: Instance = parse_instance_from_filepath(
        "data/raw/darp_instances/darp_cordeau_2006/a2-8"
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
    sol = model.solve()
    print("Time to solve the model:", time() - t_start)

    # print("Calculated output:")
    # print(sol)

    # print("Solver output:")
    # print(sol.solver_solution)

    # Create a Pandas DataFrame to exhibit the routes
    df = sol.route_df(fn_dist=model.dist)
    print(df)

    print("Vehicle Routes:")
    print(sol.vehicle_solutions)
    for k, data in sol.vehicle_solutions.items():
        print(data)

    print(i.nodeset_df)


# %%

import matplotlib.pyplot as plt
from src.plot.route import plot_vehicle_route

fig, ax = plt.subplots(1)
vehicle_id = 0
v = sol.vehicle_solutions[vehicle_id]
plot_vehicle_route(
    ax,
    v.route,
    i.nodes,
    show_arrows=True,
    show_node_labels=True,
    route_color="green",
    linestyle="-",
    linewidth=0.5,
    arrowstyle="-|>",
    title=f"Route vehicle {vehicle_id} ({v.summary()})",
)

# %%
