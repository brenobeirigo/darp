

import tempfile
import pytest
from pprint import pprint
from src.data.scenario import Scenario
from src.data.scenario import ScenarioConfig
from src.solution.build import build_run


def test_read_scenario_file():

    

    scenarios:dict[Scenario] = Scenario.read("tests/data/test_scenario1.json")
    pprint(scenarios)
    assert set(scenarios.keys()).issubset(set(["m6"]))
    all_scenarios = list(scenarios["m6"].generate_scenario_values())
    pprint(all_scenarios)
    
    scenario0 = ScenarioConfig(scenario_id='M6',
                instance_filepath='tests/data/vrppd_7-3-2.txt',
                is_flex_depot=True,
                max_driving_time_h=None,
                allow_rejection=True,
                obj=('Max. Profit', None),
                cost_per_min=0.333333333,
                cost_per_km=0.01,
                speed_km_h=50,
                revenue_per_load_unit=50,
                instance_label='vrppd_7-3-2')
    
    scenario1 = ScenarioConfig(scenario_id='M6',
                instance_filepath='tests/data/vrppd_7-3-2.txt',
                is_flex_depot=True,
                max_driving_time_h=None,
                allow_rejection=True,
                obj=('Multi-Objective Profit & Costs',
                     {'weight_costs': 0.75, 'weight_profit': 0.25}),
                cost_per_min=0.333333333,
                cost_per_km=0.01,
                speed_km_h=50,
                revenue_per_load_unit=50,
                instance_label='vrppd_7-3-2')
    
    print("Scenario 0:", scenario0.get_test_label(max_time=10))
    print("Scenario 1:", scenario1.get_test_label(max_time=10))
    
    scenario2 = ScenarioConfig(scenario_id='M6',
                instance_filepath='tests/data/vrppd_7-3-2.txt',
                is_flex_depot=True,
                max_driving_time_h=None,
                allow_rejection=True,
                obj=('Multi-Objective Profit & Costs',
                     {'weight_costs': 0.5, 'weight_profit': 0.5}),
                cost_per_min=0.333333333,
                cost_per_km=0.01,
                speed_km_h=50,
                revenue_per_load_unit=50,
                instance_label='vrppd_7-3-2')

    scenario3 = ScenarioConfig(scenario_id='M6',
                instance_filepath='tests/data/vrppd_7-3-2.txt',
                is_flex_depot=True,
                max_driving_time_h=None,
                allow_rejection=True,
                obj=('Multi-Objective Profit & Costs',
                     {'weight_costs': 0.25, 'weight_profit': 0.75}),
                cost_per_min=0.333333333,
                cost_per_km=0.01,
                speed_km_h=50,
                revenue_per_load_unit=50,
                instance_label='vrppd_7-3-2')
    
    scenario4 = ScenarioConfig(scenario_id='M6',
                instance_filepath='tests/data/vrppd_7-3-2.txt',
                is_flex_depot=True,
                max_driving_time_h=None,
                allow_rejection=True,
                obj=('Multi-Objective Profit & Costs',
                     {'weight_costs': 0.24, 'weight_profit': 0.75}), # change weight
                cost_per_min=0.333333333,
                cost_per_km=0.01,
                speed_km_h=50,
                revenue_per_load_unit=50,
                instance_label='vrppd_7-3-2')
    
    sol = build_run(config=scenario1, routes_folder="tests/data/routes/")
    pprint(sol.columns)
    pprint(sol.values)

    assert scenario1 in all_scenarios
    assert scenario2 in all_scenarios
    assert scenario3 in all_scenarios
    assert scenario4 not in all_scenarios
    