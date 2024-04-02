# from typing import list, dict
import pytest
from pprint import pprint
import math

from src.data.parser import vrppd_dict_to_instance_obj
from src import Darp
import src.solver.darp as darp

@pytest.mark.parametrize(
    'one_depot_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 10}, darp.OBJ_MAX_PROFIT, None, None), # Fails capacity
        ({"speed": 5, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 2, 3, 4)]), 40),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MAX_PROFIT, set([(1, 2, 3, 4)]),
         (50 * 20 - 0.05 * 0.2 * (10 + 10 + 20) - 20 / 60 * (320 - 190))),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_COST, set([(1, 2, 3, 4)]),
         (0.05 * 0.2 * (10 + 10 + 20))),
    ],
    indirect=["one_depot_one_customer_one_vehicle"]
)
def test_pudo_order(one_depot_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = one_depot_one_customer_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    solution_obj = model.solve()

    if solution_obj:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None



@pytest.mark.parametrize(
    'two_depots_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 10}, darp.OBJ_MAX_PROFIT, None, None), # Fails capacity
        ({"speed": 5, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 3, 4, 5)]), 40),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MAX_PROFIT, set([(1, 3, 4, 5)]),
         (50 * 20 - 0.05 * 0.2 * (10 + 10 + 20) - 20 / 60 * (320 - 190))),
    ],
    indirect=["two_depots_one_customer_one_vehicle"]
)
def test_if_flexible_depot_then_vehicle_departs_and_returns_from_closest(two_depots_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = two_depots_one_customer_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    
    solution_obj = model.solve()

    if solution_obj:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None
        
    
@pytest.mark.parametrize(
    'two_diff_depots_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 10}, darp.OBJ_MAX_PROFIT, None, None), # Fails capacity
        ({"speed": 1, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 3, 4, 6)]), 30),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MAX_PROFIT, set([(1, 3, 4, 6)]),
         (50 * 20 - 0.05 * 0.2 * (10 + 10 + 10) - 20 / 60 * (310 - 190))),
    ],
    indirect=["two_diff_depots_one_customer_one_vehicle"]
)

        
def test_if_flexible_depot_then_vehicle_starts_and_ends_at_different_depots(two_diff_depots_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = two_diff_depots_one_customer_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    
    solution_obj = model.solve()

    if solution_obj:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None


 
@pytest.mark.parametrize(
    'two_diff_depots_two_customers_one_vehicle, obj, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 10}, darp.OBJ_MAX_PROFIT, None, None), # Fails capacity
        ({"speed": 1, "capacity": 40}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        ({"speed": 60, "capacity": 40}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 3, 4, 5, 6, 8)]), 30),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 3, 5, 4, 6, 8)]), 40),
        ({"speed": 60, "capacity": 40}, darp.OBJ_MAX_PROFIT, set([(1, 3, 4, 5, 6, 8)]),
         (50 * 20 * 2 - 0.05 * 0.2 * (10 + 10 + 10) - 20 / 60 * (500 - 470))),
    ],
    indirect=["two_diff_depots_two_customers_one_vehicle"]
)

        
def test_picks_2_delivers_2_unrestricted_tws(two_diff_depots_two_customers_one_vehicle, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = two_diff_depots_two_customers_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    
    solution_obj = model.solve()

    if solution_obj:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None
        

    # depot1 = generate_node(1, 0, 0, 500, 10, 50)
    # depot2 = generate_node(2, 0, 0, 500, 40, 50)
    # pickup1 = generate_node(3, 20, 0, 500, 20, 50)
    # pickup2 = generate_node(4, 20, 0, 500, 25, 50)
    # delivery1 = generate_node(5, -20, 0, 500, 30, 50)
    # delivery2 = generate_node(6, -20, 0, 500, 35, 50)
    # trucks = generate_trucks_dict(1, capacity, 480, 50, speed_km_h=speed_km_h)
    
@pytest.mark.parametrize(
    'two_diff_depots_two_customers_one_vehicle, obj, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 10}, darp.OBJ_MAX_PROFIT, None, None), # Fails capacity
        ({"speed": 1, "capacity": 40}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        ({"speed": 60, "capacity": 40}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(2, 4, 3, 5, 6, 8)]), 40),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(2, 3, 5, 4, 6, 8)]), 50),
        ({"speed": 60, "capacity": 40}, darp.OBJ_MAX_PROFIT, set([(2, 4, 3, 5, 6, 8)]),
         (50 * 20 * 2 - 0.05 * 0.2 * (15 + 5 + 10 + 5 + 5) - 20 / 60 * (500 - 460))),
    ],
    indirect=["two_diff_depots_two_customers_one_vehicle"]
)

        
def test_picks_2_delivers_2_unrestricted_tws_fixed_depots(two_diff_depots_two_customers_one_vehicle, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = two_diff_depots_two_customers_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    model.set_flex_depot(False)
    
    solution_obj = model.solve()

    if solution_obj:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None
