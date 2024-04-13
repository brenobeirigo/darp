
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

def plot_routes(p):
    test_case = p[:-4]
    df = pd.read_csv(f"reports/tables/routes/{p}")
    vehicles = df["vehicle_id"].unique()

    fig, axs = plt.subplots(1, figsize=(10, 10))  # Creates a single Axes object with a square figure size
    axs.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
    #fig, axs = plt.subplots(1, len(vehicles), figsize=(5*len(vehicles), 5))  # Creates a 2x2 grid of Axes objects
    #if len(vehicles) > 1:
    #    axs = axs.flatten()

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
            -10,
            110,
            -10,
            110,
            ])
        
        print(df_vehicle_solution)
        
        #title_plot = f"Route vehicle {vehicle_id}\ntw=[{depot1/60:3.2f},{depot2/60:3.2f}] / dist={dist:6.2f} / lat={latency/60:3.2f}"# ({vehicle_sol.summary()})"
        plot_vehicle_route(axs, df_vehicle_solution, coord_box=tuple(box), route_color=color[vehicle_id])
        
    plt.savefig(f"{folder_fig}/{test_case}.svg", bbox_inches='tight')
    
for p in plots:
    plot_routes(p)
    