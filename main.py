
# %%

from src import parse_instance_from_filepath, Instance
from src import Darp
from time import time
from src.solution.parser import Solution, parse_solution_dict

# i : Instance = parse_instance_from_filepath("data/raw/darp_instances/darp_cordeau_2006/a2-16")
# i.nodeset_df
# # %%

# model = Darp(**i.get_data())
# print(model.build_solve())
    
if __name__ == "__main__":
    t_start = time()
    i : Instance = parse_instance_from_filepath("data/raw/darp_instances/darp_cordeau_2006/a2-16")
    print("Time to load instance:", time() - t_start)
    
    t_start = time()
    model = Darp(**i.get_data())
    print("Time to initialize the model:", time() - t_start)
    
    t_start = time()
    model.build()
    print("Time to build the model:", time() - t_start)
    
    t_start = time()
    sol_dict = model.solve()
    print("Time to solve the model:", time() - t_start)
    
    print("Calculated output:")
    print(sol_dict["fleet"]["summary"])
    
    print("Solver output:")
    print(sol_dict["solver"])


    print(i.nodeset_df)

    for k, data in sol_dict["fleet"]["K"].items():
        print(data)
        
    print("## Solution")
    s = parse_solution_dict(sol_dict)
    v0_sol = s.vehicle_solutions
    print(v0_sol)
    print(s)