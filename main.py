
from src import parse_instance_from_filepath, Instance
from src import Darp

if __name__ == "__main__":
    i : Instance = parse_instance_from_filepath("data/raw/darp_instances/darp_cordeau_2006/a2-16")
    model = Darp(**i.get_data())
    print(model.build_solve())
    
    