import os
from pprint import pprint
from solver.darp import Darp
from instance import parser as instance_parser
from solution import parser as solution_parser
import plot.route as route_plot
import json
import logging

# Change logging level to DEBUG to see the construction of the model
format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger()

result_folder = "./results/darp_mip/"
os.makedirs(result_folder, exist_ok=True)

instance_folder = "./instance/data/darp_cordeau_2006/"
instance_filenames = os.listdir(instance_folder)

for filename in instance_filenames:
    logger.info(filename)
    filepath = os.path.join(instance_folder, filename)
    instance = instance_parser.parse_instance_from_filepath(
        filepath, instance_parser=instance_parser.PARSER_TYPE_CORDEAU
    )

    # print(instance.get_data())
    model = Darp(**instance.get_data())
    model.build()
    result = model.solve()
    
    # PRINT ROUTES
    # result["instance"] = filename
    # print(result["fleet"]["K"].items())
    # for k, k_nodes in result["fleet"]["K"].items():
    #     print(k, [n for n, _ in k_nodes["route"]])
    # #     print(k, [n for n, data in k_result["route"].items()])
    
    logger.info(result["solver"]["sol_objvalue"])

    solution_obj = solution_parser.parse_solution_dict(result)
    
    sol_filepath = f"{os.path.join(result_folder, filename)}.json"
    with open(sol_filepath, "w") as outfile:
        json.dump(result, outfile, indent=4)
    
    fig, ax = route_plot.plot_vehicle_routes(
        instance,
        solution_obj,
        jointly=True,
        figsize=(10, 10),
        show_arrows=False,
        show_node_labels=False,
    )

    fig_filepath = f"{os.path.join(result_folder, filename)}.pdf"
    fig.savefig(fig_filepath, bbox_inches="tight")
