
# import logging
# logging.basicConfig(level=logging.DEBUG)
# run using python -m src.visualization.plot
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .route import plot_vehicle_route

plots = os.listdir(f"reports/tables/routes/")
folder_fig = "reports/figures"
color = ["red", "green", "blue"]
for p in plots:
    test_case = p[:-4]
    df = pd.read_csv(f"reports/tables/routes/{p}")
    vehicles = df["vehicle_id"].unique()

    fig, axs = plt.subplots(1, len(vehicles), figsize=(5*len(vehicles), 5))  # Creates a 2x2 grid of Axes objects
    if len(vehicles) > 1:
        axs = axs.flatten()

    for i, vehicle_id in enumerate(vehicles):
        df_vehicle_solution = df[df["vehicle_id"] == vehicle_id].copy()
        
        depot1 = min(df_vehicle_solution["arrival"])
        depot2 = max(df_vehicle_solution["arrival"])
        dist = sum(df_vehicle_solution["distance_previous"])
    

        latency = sum(df_vehicle_solution[
            (df_vehicle_solution["node_type"] =="PU")
            | (df_vehicle_solution["node_type"] =="DO")
        ]["arrival"])

        box = np.array(
            [
            min(df_vehicle_solution["x"])-10,
            max(df_vehicle_solution["x"])+10,
            min(df_vehicle_solution["y"])-10,
            max(df_vehicle_solution["y"])+10])
        
        print(df_vehicle_solution)
        if len(vehicles) > 1:
            ax = axs[i]
        else:
            ax = axs
        title_plot = f"{test_case}\nRoute vehicle {vehicle_id}\ntw=[{depot1/60:3.2f},{depot2/60:3.2f}] / dist={dist:6.2f} / lat={latency/60:3.2f}"# ({vehicle_sol.summary()})"
        plot_vehicle_route(ax, df_vehicle_solution, title=title_plot, coord_box=tuple(box), route_color=color[vehicle_id])
        
    plt.savefig(f"{folder_fig}/{test_case}.png", bbox_inches='tight')
    
