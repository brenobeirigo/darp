# %%
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

result_folder = "./results/darp_heuristic/"
os.makedirs(result_folder, exist_ok=True)

instance_folder = "./instance/data/darp_cordeau_2006/"
filename = "a2-16"

filepath = os.path.join(instance_folder, filename)
instance = instance_parser.parse_instance_from_filepath(
    filepath, instance_parser=instance_parser.PARSER_TYPE_CORDEAU
)

# %%
data = instance.get_data()
customers = data["P"]
customers_tw = [(i, data["el"][i]) for i in customers]
customers_earliest = sorted(customers_tw, key=lambda x: x[1][0])
pprint(customers_earliest)

# %% [markdown]
# Consider now the case in which there are N customer demands for service
# and n available Dial-a-Ride vehicles.
# ADARTW begins by indexing customers in the order of their earliest pick-up times:
#
# %%

for node in instance.requests:
    print(node)

pprint(sorted(instance.requests, key=lambda r: r.pickup_node.el[0]))
# %%
# ”, EPTi (i = 1, . . . , N), i.e. according to the earliest time at which they are expected to be available for a pick-up. Section 4 shows how EPT, is computed.

# result["instance"] = filename

# solution_obj = solution_parser.parse_solution_dict(result)

# ol_filepath = f"{os.path.join(result_folder, filename)}.json"


# with open(sol_filepath, "w") as outfile:
#     json.dump(result, outfile, indent=4)

# fig, ax = route_plot.plot_vehicle_routes(
#     instance,
#     solution_obj,
#     jointly=True,
#     figsize=(10, 10),
#     show_arrows=False,
#     show_node_labels=False,
# )

# fig_filepath = f"{os.path.join(result_folder, filename)}.pdf"
# fig.savefig(fig_filepath, bbox_inches="tight")

# %%
