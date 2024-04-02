from src.data import parser
from pathlib import Path
import src.solver.darp as darp

# import logging

# logging.basicConfig(level=logging.DEBUG, 
#                     # Add the following line to set the logging level specifically for your project
#                     # filename='/c:/Users/AlvesBeirigoB/OneDrive/pkm/PARA/Area/dev/darp/example.py',
#                     # level=logging.DEBUG,
#                     )

if __name__ == "__main__":
        
    folder_path = Path("data/raw/mdvrppdtw/")

    instances_files = [
        "vrppd_13-3-5.txt",
        "vrppd_23-3-10.txt",
        "vrppd_33-3-15.txt",
    ]


    instances = {}
    for instance_file in instances_files:
        instance_filepath = folder_path / instance_file
        print(f"## Processing instance at '{instance_filepath}'...")

        # Create an Instance object
        instance_obj = parser.parse_instance_from_filepath(
            instance_filepath,
            instance_parser=parser.PARSER_TYPE_MVRPPDTW
        )
        print(instance_obj)
        instances[instance_file[:-4]] = instance_obj

    # # Displaying the instance data as a DataFrame for verification
    # df_instance = instances["vrppd_13-3-5"].nodeset_df
    # df_instance

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create subplots for each instance
    fig, axs = plt.subplots(
        1,
        len(instances),
        figsize=(5 * len(instances), 5))

    # Iterate over instances and plot scatter plots
    for i, (instance_file, instance_obj) in enumerate(instances.items()):
        ax = axs[i]
        subfig_title = f"{instance_file}\n({instance_obj.config.label})"
        ax.set_title(subfig_title)
        sns.scatterplot(
            data=instance_obj.nodeset_df,
            x="x",
            y="y",
            hue="node_type",
            ax=ax)


    from time import time
    import matplotlib.pyplot as plt
    from pprint import pprint

    # Parser to load and parse DARP instance data
    from src.data.parser import parse_instance_from_filepath

    # Darp class for building and solving the DARP model
    from src.solver.darp import Darp

    # Function for plotting vehicle routes
    from src.visualization.route import plot_vehicle_route


    # Initializing the DARP model
    t_start = time()
    # instance_obj = instances["vrppd_13-3-5"] 
    instance_obj = instances["vrppd_23-3-10"] 
    # instance_obj = instances["vrppd_33-3-15"] 
    model = Darp(instance_obj)
    print("Time to initialize the model:", time() - t_start)

    # Building the model with constraints, variables, and objective function
    t_start = time()
    model.build()
    model.save_lp("./reports/lps/model_example.lp")
    model.save_log("./reports/logs/model_example.log")
    model.set_time_limit_min(0.05)
    print("Time to build the model:", time() - t_start)

    # Solving the model to minimize costs
    t_start = time()
    model.set_obj(darp.OBJ_MIN_TRAVEL_DISTANCE)
    solution_obj = model.solve()
    print("Time to solve the model:", time() - t_start)

    
    pprint(solution_obj)
    
    # Detailed solver-specific information
    pprint(solution_obj.solver_stats)
    
    # Detailed solver-specific information

    df = solution_obj.route_df(fn_dist=model.dist)

    # Creating a 2x2 grid for plotting routes
    fig, axs = plt.subplots(1, figsize=(10, 10))


    # Iterating through vehicle routes for visualization
    for vehicle_id, vehicle_sol in solution_obj.vehicle_routes.items():
        df_vehicle_solution = df[df["vehicle_id"] == vehicle_id].copy()
        print(df_vehicle_solution)
        title_plot = f"Route vehicle {vehicle_id} ({vehicle_sol.summary()})"
        plot_vehicle_route(None, df_vehicle_solution, title=title_plot)

    plt.tight_layout()