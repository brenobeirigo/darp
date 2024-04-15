
# import logging
# logging.basicConfig(level=logging.DEBUG)
# run using python -m src.visualization.plot

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .route import plot_vehicle_route, plot_nodes, plot_node_labels
import re
import math

def compute_transport_metrics(df, speed, revenue, cost_driving_h, cost_per_km):
    # Compute costs
    # cost_per_km = cost_emission * emission  # euro/km
    
    final_makespan = max(df["arrival"]) / 60
    total_hours_working = sum(
        [
            (max(df[df["vehicle_id"]==i]["arrival"]) 
             - min(df[df["vehicle_id"]==i]["arrival"])) 
            for i in df["vehicle_id"].unique()]) / 60

    latency = sum(df[(df["node_type"]=="PU") | (df["node_type"]=="DO")]["arrival"])/60
    serviced = len(df[(df["node_type"]=="PU")])
    
    # Compute total load
    total_load = 0
    for i, row in df.iterrows():
        if row["node_type"] == "PU" and i > 0:  # check for previous row existence
            total_load += row["vehicle_load"] - df.iloc[i - 1]["vehicle_load"]

    # Compute total distance
    dist = 0
    coordinates = df[["x", "y"]].values
    for o, d in zip(coordinates[:-1], coordinates[1:]):
        dist += math.sqrt((float(d[0]) - float(o[0])) ** 2 + (float(d[1]) - float(o[1])) ** 2)

    total_distance = sum(df["distance_previous"])
    # total_distance = dist
    total_hours_driving = total_distance / speed
    
    cost_km = cost_per_km * total_distance
    total_revenue = total_load * revenue
    cost_driving = cost_driving_h * total_hours_driving
    cost_working = cost_driving_h * total_hours_working

    # Create and return the dictionary with formatted values
    result = {
        "Distance": f"{total_distance:,.2f} km",
        "N. of Serviced" : f"{serviced}",
        "Load Transported": f"{total_load:,.2f} kg",
        "Hours Driving (distance/speed)": f"{total_hours_driving:,.2f} hours",
        "Hours Working (driving + waiting)": f"{total_hours_working:,.2f} hours",
        # "Euclidean Distance": f"{dist:,.2f} km",
        "Makespan": f"{final_makespan:,.2f} h",
        "Latency": f"{latency:,.2f} h",
        "Cost (driving)": f"{cost_driving:,.2f} euros",
        "Cost (working)": f"{cost_driving:,.2f} euros",
        "Cost (distance)": f"{cost_km:,.2f} euros",
        "Revenue": f"{total_revenue:,.2f} euros",
        # "Revenue - (Driving + Emission Costs)": f"{total_revenue - cost_driving - cost_km:,.2f} euros",
        "Profit (discount driving & emissions)": f"{total_revenue - cost_driving - cost_km:,.2f} euros",
        "Profit (discount working & emissions)": f"{total_revenue - cost_working - cost_km:,.2f} euros",
        # "Revenue - Driving Costs": f"{total_revenue - cost_driving:,.2f} euros"
    }

    return result

def get_instance_dict():
    
    from pathlib import Path
    from pprint import pprint

    # Parser to load and parse DARP instance data
    import src.data.parser as parser

    folder_path = Path("data/raw/mdvrppdtw/")

    instances_files = [
        "vrppd_13-3-5.txt",
        "vrppd_23-3-10.txt",
        "vrppd_33-3-15.txt",
    ]

    # The instances are saved into a dictionary with
    # keys corresponding to the instance name
    instances = {}

    for instance_file in instances_files:

        # Create file path
        instance_filepath = folder_path / instance_file
        print(f"\n## Processing instance at '{instance_filepath}'")

        # Create an Instance object. The parser, is a function
        # created to read files of this instance type. When working
        # with instances from different sources, you build a parser
        # for each source so instance files are read correctly.
        instance_obj = parser.parse_instance_from_filepath(
            instance_filepath,
            instance_parser=parser.PARSER_TYPE_MVRPPDTW
        )

        instances[instance_file[:-4]] = instance_obj
        
    return instances
        

instances = get_instance_dict()

## %%
def find_instance_id(scenario_id):

    pattern = r"vrppd_\d+-\d+-\d+"
    match = re.search(pattern, scenario_id)

    if match:
        print("Match found!")
        return match.group()
    else:
        print("Match not found.")
        return None

plots = os.listdir(f"reports/tables/routes/")
folder_fig = "reports/figures"
color = ["red", "green", "blue"]

def plot_routes(p):
    test_case = p[:-4]
    print(test_case)
    speed = re.search(r"speed_km_h=(\d+)", test_case).groups()[0]
    revenue = re.search(r"revenue_per_load_unit=(\d+)", test_case).groups()[0]
    cost_per_min = re.search(r"cost_per_min=(\d+)", test_case).groups()[0]
    cost_emission = re.search(r"cost_per_min=(\d+)", test_case).groups()[0]
    cost_per_km = float(re.search(r"cost_per_km=(\d+\.\d+)", test_case).groups()[0])
    
    match = re.search(r"scenario_id=(..)", test_case)
    print(match.groups())
    # if match.groups()[0] != "M6":
    #     return
    instance_id = find_instance_id(test_case)
    df_nodes = instances[instance_id].nodeset_df
    print(instance_id, df_nodes)
    
    df = pd.read_csv(f"reports/tables/routes/{p}")
    results = compute_transport_metrics(df, float(speed), float(revenue), 20, float(cost_per_km))
    print(results)
    
    vehicles = df["vehicle_id"].unique()
    print(f"Plotting {test_case}")
    print(df.head())
    print(df_nodes.columns)
    print(df.columns)
    df_filtered = df_nodes[~df_nodes.index.isin(df["id"])]
    print("Filtered")
    print(df_filtered)
    
    

    fig, axs = plt.subplots(1, figsize=(10, 10))  # Creates a single Axes object with a square figure size
    
    
    axs.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
    #fig, axs = plt.subplots(1, len(vehicles), figsize=(5*len(vehicles), 5))  # Creates a 2x2 grid of Axes objects
    #if len(vehicles) > 1:
    #    axs = axs.flatten()
    
    # Plot the results dictionary at the top right of the axs
    results_str = "\n".join([f"{key} = {value:<16}" for key, value in results.items()])
    axs.text(0.95, 0.95, results_str, transform=axs.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontfamily='monospace')
    # Nodes that do not belong to the route are plotted with transparency
    plot_nodes(axs, df_filtered, skip=True)
    plot_node_labels(axs, df_filtered)

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
    plt.close(fig)
    
    
    # print(instances[instance_id].to_df())

    
for p in plots:
    plot_routes(p)
    
from pprint import pprint
pprint(instances)
    