
from src.data.scenario import ScenarioConfig
import src.data.parser as parser
import pandas as pd
import pathlib
import logging
from src.solver.darp import Darp
# import sys
# sys.path.append("../")

logger = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)

def build_run(
    config: ScenarioConfig,
    time_limit_min=10,
    lp_filepath=None,
    log_filepath=None,
    ):


    logger.debug(f"\n## Processing instance at '{config.instance_filepath}'")

    # Create an Instance object. The parser, is a function
    # created to read files of this instance type. When working
    # with instances from different sources, you build a parser
    # for each source so instance files are read correctly.
    instance_obj = parser.parse_instance_from_filepath(
        config.instance_filepath,
        instance_parser=parser.PARSER_TYPE_MVRPPDTW
    )

    logger.debug(instance_obj)

    test_label = config.get_test_label(time_limit_min)
    logger.debug(f"### Running test: {test_label}")
    logger.debug(config)

    logger.debug(f"### Instance:")
    logger.debug(instance_obj.config)
        
    ## Build model
    model = Darp(i=instance_obj, s=config)
    model.build(
        max_driving_time_h=config.max_driving_time_h,
        allow_rejection=config.allow_rejection
        )

    # Configure model
    model.set_flex_depot(config.is_flex_depot)
    model.set_obj(config.obj)

    # Configure logs
    if lp_filepath:
        model.save_lp(str(pathlib.Path(lp_filepath)/f"{test_label}.lp"))
    if log_filepath:
        model.save_log(str(pathlib.Path(log_filepath)/f"{test_label}.log"))

    model.set_time_limit_min(time_limit_min)

    # Run solver
    solution_obj = model.solve()

    df_config = pd.DataFrame([config])
    
    # Print solver statistics
    logger.debug(solution_obj.solver_stats)

    if solution_obj.vehicle_routes:

        # Save route
        df_routes = solution_obj.route_df(fn_dist=model.dist)
        df_routes.to_csv(f"../reports/tables/routes/{test_label}.csv")
        logger.debug(df_routes)
    
    else:
        logger.debug("Optimal solution was not found. Can't retrieve routes.")

    # Add results
    df_test = pd.concat(
        [
            df_config,
            solution_obj.to_df(),
            pd.DataFrame({"Max Runtime (min)":[time_limit_min]})], axis=1)


    df_test.insert(0, 'Date', pd.Timestamp.today())
    df_test.insert(0, 'Test ID', test_label)
    
    rename_dict = {
        'Test ID': 'Test ID',
        'Date': 'Date',
        'scenario_id': 'Scenario ID',
        'instance_filepath': 'Instance Filepath',
        'is_flex_depot': 'Allow Flexible Depot Return',
        'max_driving_time_h': 'Maximum Working Time (h)',
        'allow_rejection': 'Allow Customer Rejection',
        'obj': 'Objective',
        'cost_per_min': 'Cost Per Min',
        'cost_per_km': 'Cost Per Km',
        'speed_km_h': 'Speed (km/h)',
        'revenue_per_load_unit': 'Revenue Per Load Unit',
        'instance_label': 'Instance Label',
        'Number of Vehicles': 'Number Of Vehicles',
        'Number of Customers': 'Number Of Customers',
        'Time Horizon (min)': 'Time Horizon (min)',
        'Vehicle Capacity': 'Vehicle Capacity',
        'Maximum Ride Time (min)': 'Maximum Ride Time (min)',
        'Maximum Driving Time (min)': 'Maximum Driving Time (min)',
        'Number of Depots': 'Number Of Depots',
        'Total Distance (km)': 'Total Distance (km)',
        'Total Duration (min)': 'Total Duration (min)',
        'Total Waiting (min)': 'Total Waiting (min)',
        'Average Waiting (min)': 'Average Waiting (min)',
        'Total Transit (min)': 'Total Transit (min)',
        'Average Transit (min)': 'Average Transit (min)',
        'Objective Value': 'Objective Value',
        'CPU Time (sec)': 'Cpu Time (sec)',
        'Number of Edges': 'Number Of Edges',
        'Number of Nodes': 'Number Of Nodes',
        'Number of Constraints': 'Number Of Constraints',
        'Number of Variables': 'Number Of Variables',
        'Number of Iterations': 'Number Of Iterations',
        'Number of Solver Nodes': 'Number Of Solver Nodes',
        'Solver Gap': 'Solver Gap',
        'Objective Bound': 'Objective Bound',
        'Work': 'Work',
        'Max Runtime (min)': 'Max Runtime (min)'
        }

    
   

    df_test.rename(columns=rename_dict, inplace=True)
    
    return df_test