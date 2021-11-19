import os
from solver.darp import Darp
from instance import parser as pi


root = "/home/bbeirigo/study/metaheuristics/darp"
instance_folder = os.path.join(root, "instance/data/darp_cordeau_2006/")

instance_filenames = os.listdir(instance_folder)

for filename in instance_filenames:
    print(filename)
    instance = pi.parse_instance_from_filepath(
            os.path.join(instance_folder, filename),
            instance_parser=pi.PARSER_TYPE_CORDEAU)
    
    model = Darp(**instance.get_data())
    model.build()
    model.solve()
    print(model.summary_sol)
    # model.stats()

