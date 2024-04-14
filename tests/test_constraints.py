# from typing import list, dict
# pytest .\tests\test_constraints.py --log-cli-level=DEBUG -svv
import pytest
from pprint import pprint
import math

from src.data.parser import vrppd_dict_to_instance_obj
from src import Darp
import src.solver.darp as darp


@pytest.mark.parametrize(
    'one_depot_one_customer_one_vehicle, reject, obj, params, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 20},False, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0.5, weight_costs=0.5), set([(1, 2, 3, 4)]), 0.5), 
        ({"speed": 60, "capacity": 20},False, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0.25, weight_costs=0.75), set([(1, 2, 3, 4)]), 0.25), 
        ({"speed": 60, "capacity": 20},False, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=1, weight_costs=0), set([(1, 2, 3, 4)]), 1), 
        ({"speed": 60, "capacity": 20},False, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0, weight_costs=1), set([(1, 2, 3, 4)]), 0), # Users can't be rejected, but mo solution is 0
        ({"speed": 60, "capacity": 20},True, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0.5, weight_costs=0.5), set([(1, 4)]), 0.5*((0-(-166.666))/(956.266666-(-166.666666))) + 0.5 * 1), 
        ({"speed": 60, "capacity": 20},True, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0.25, weight_costs=0.75), set([(1, 4)]), 0.25*((0-(-166.666))/(956.26666-(-166.666666))) + 0.75 * 1), 
        ({"speed": 60, "capacity": 20},True, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0.75, weight_costs=0.25), set([(1, 2, 3, 4)]), 0.75*((956.26666-(-166.666))/(956.26666-(-166.666666))) + 0.25 * 0), 
        ({"speed": 60, "capacity": 20},True, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=1, weight_costs=0), set([(1, 2, 3, 4)]), 1), 
        ({"speed": 60, "capacity": 20},True, darp.MO_MIN_PROFIT_MIN_TRAVEL_COST, dict(weight_profit=0, weight_costs=1), set([(1, 4)]), 1), # 1 because costs is 0 
        # ({"speed": 5, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        # ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 2, 3, 4)]), 40),
        # ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_DRIVING_TIME, set([(1, 2, 3, 4)]), 130),
        # ({"speed": 60, "capacity": 20}, darp.OBJ_MAX_PROFIT, set([(1, 2, 3, 4)]),
        #  (50 * 20 - 0.05 * 0.2 * (10 + 10 + 20) - 20 / 60 * (320 - 190))),
        # ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_COST, set([(1, 2, 3, 4)]),
        #  (0.05 * 0.2 * (10 + 10 + 20))),
    ],
    indirect=["one_depot_one_customer_one_vehicle"]
)
def test_mo_solution(one_depot_one_customer_one_vehicle, reject, obj, params, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = one_depot_one_customer_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build(allow_rejection=reject)
    model.set_obj(obj, params=params)
    solution_obj = model.solve()

    if solution_obj.vehicle_routes:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.summary)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None
        

@pytest.mark.parametrize(
    'one_depot_one_customer_one_vehicle, obj, expected_sol_route, expected_sol_obj',
    [
        ({"speed": 60, "capacity": 10}, darp.OBJ_MAX_PROFIT, None, None), # Fails capacity
        ({"speed": 5, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, None, None), # Fails speed
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1, 2, 3, 4)]), 40),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_DRIVING_TIME, set([(1, 2, 3, 4)]), 130),
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

    if solution_obj.vehicle_routes:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.summary)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None
        
# 10km/60km/60min = 1/5*60 = 10
# Min. Duration
# 10[0,500]-------->20[100,200]----->30[300,400]----->50[0,500]
#      190  + 10 +              +100                +20 =  320

# Min. Distance
# 10[0,500]-------->20[100,200]----->30[300,400]----->50[0,500]
#         0 + 10    + 190(w)    + 10 + 90(w)    +20 =  320

@pytest.mark.parametrize(
    'one_depot_one_customer_one_vehicle, obj, expected_sol',
    [
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_DISTANCE, dict(
            route=set([(1, 2, 3, 4)]),
            obj=40,
            total_distance=40.0,
            total_duration=320.0,
            total_waiting=280.0,
            avg_waiting=280.0,
            total_transit=100.0,
            avg_transit=100.0,
            total_latency=500.0,
            final_makespan=320.0)),
        ({"speed": 60, "capacity": 20}, darp.OBJ_MIN_DRIVING_TIME, dict(
            route=set([(1, 2, 3, 4)]),
            obj=130,
            total_distance=40.0,
            total_duration=130.0,
            total_waiting=280.0,
            avg_waiting=280.0,
            total_transit=100.0,
            avg_transit=100.0,
            total_latency=500.0,
            final_makespan=320.0)),
        #({"speed": 60, "capacity": 20}, darp.OBJ_MIN_DRIVING_TIME, set([(1, 2, 3, 4)]), 130),
        #({"speed": 60, "capacity": 20}, darp.OBJ_MAX_PROFIT, set([(1, 2, 3, 4)]),
        # (50 * 20 - 0.05 * 0.2 * (10 + 10 + 20) - 20 / 60 * (320 - 190))),
        #({"speed": 60, "capacity": 20}, darp.OBJ_MIN_TRAVEL_COST, set([(1, 2, 3, 4)]),
        # (0.05 * 0.2 * (10 + 10 + 20))),
    ],
    indirect=["one_depot_one_customer_one_vehicle"]
)
def test_solution_summary(one_depot_one_customer_one_vehicle, obj, expected_sol):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = one_depot_one_customer_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    solution_obj = model.solve()

    if solution_obj.vehicle_routes:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.summary)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol["route"]
        assert math.isclose(expected_sol["obj"], actual_sol_obj, rel_tol=1e-6)
        assert math.isclose(expected_sol["final_makespan"], solution_obj.summary.final_makespan, rel_tol=1e-6)
        assert math.isclose(expected_sol["total_latency"], solution_obj.summary.total_latency, rel_tol=1e-6)
        assert math.isclose(expected_sol["total_duration"], solution_obj.summary.total_duration, rel_tol=1e-6)
        assert math.isclose(expected_sol["total_distance"], solution_obj.summary.total_distance, rel_tol=1e-6)
        assert math.isclose(expected_sol["total_waiting"], solution_obj.summary.total_waiting, rel_tol=1e-6)
        assert math.isclose(expected_sol["total_transit"], solution_obj.summary.total_transit, rel_tol=1e-6)
    else:
        assert expected_sol is None




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

    if solution_obj.vehicle_routes:
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
    model.save_lp(f"test_if_flexible_depot_then_vehicle_starts_and_ends_at_different_depot_{obj}_{str(expected_sol_route)}_{str(expected_sol_obj)}.lp")
    
    solution_obj = model.solve()

    if solution_obj.vehicle_routes:
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

        
def test_picks_2_delivers_2_unrestricted_tws_flex_true(two_diff_depots_two_customers_one_vehicle, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = two_diff_depots_two_customers_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build()
    model.set_obj(obj)
    
    solution_obj = model.solve()

    if solution_obj.vehicle_routes:
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
    'one_depot_one_customer_one_vehicle_no_tw,allow_rejection, max_driving_time_h, obj, expected_sol_route, expected_sol_obj',
    [
        #({"speed": 60, "capacity": 20}, False, 100/60, darp.OBJ_MIN_DRIVING_TIME, set([(2, 4, 6, 3, 5, 8)]), 100),
        ({"speed": 60, "capacity": 20}, False, 40/60, darp.OBJ_MIN_DRIVING_TIME, set([(1,2,3,4)]), 40),
        ({"speed": 60, "capacity": 20}, False, 39/60, darp.OBJ_MIN_DRIVING_TIME, None, None),
        ({"speed": 60, "capacity": 20}, True, 39/60, darp.OBJ_MIN_DRIVING_TIME, set([(1,4)]), 0),
        ({"speed": 60, "capacity": 20}, True, 40/60, darp.OBJ_MAX_PROFIT, set([(1,2,3,4)]), 986.2666666666665),
        ({"speed": 60, "capacity": 20}, False, 40/60, darp.OBJ_MIN_TRAVEL_DISTANCE, set([(1,2,3,4)]), 40),
    ],
    indirect=["one_depot_one_customer_one_vehicle_no_tw"]
)

        
def test_allow_rejection_if_max_driving_time_is_short(one_depot_one_customer_one_vehicle_no_tw, allow_rejection, max_driving_time_h, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = one_depot_one_customer_one_vehicle_no_tw
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build(allow_rejection=allow_rejection, max_driving_time_h=max_driving_time_h)
    model.set_obj(obj)
    model.set_flex_depot(False)
    model.save_lp(f"{allow_rejection} - {max_driving_time_h}.lp")
    solution_obj = model.solve()

    if solution_obj.vehicle_routes:
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
    'two_diff_depots_two_competing_customers_one_vehicle,allow_rejection, max_driving_time_h, obj, expected_sol_route, expected_sol_obj',
    [
        #({"speed": 60, "capacity": 20}, False, 100/60, darp.OBJ_MIN_DRIVING_TIME, set([(2, 4, 6, 3, 5, 8)]), 100),
        ({"speed": 60, "capacity": 100}, True, 40/60, darp.OBJ_MAX_PROFIT, set([(1,3,5,7)]), 986.2666666666665),
        ({"speed": 60, "capacity": 100}, True, 39/60, darp.OBJ_MAX_PROFIT, set([(1,7)]), 0),
        ({"speed": 60, "capacity": 100}, False, 39/60, darp.OBJ_MAX_PROFIT, None, None),
        
        #({"speed": 60, "capacity": 100}, False, 39/60, darp.OBJ_MIN_DRIVING_TIME, None, None),
        #({"speed": 60, "capacity": 100}, True, 39/60, darp.OBJ_MIN_DRIVING_TIME, set([(1,4)]), 0),
        #({"speed": 60, "capacity": 100}, True, 40/60, darp.OBJ_MAX_PROFIT, set([(1,2,3,4)]), 986.2666666666665),
    ],
    indirect=["two_diff_depots_two_competing_customers_one_vehicle"]
)

        
def test_allow_rejection_2_customers_flex_true(two_diff_depots_two_competing_customers_one_vehicle, allow_rejection, max_driving_time_h, obj, expected_sol_route, expected_sol_obj):
    """Test the order of pickup and delivery with a set of predefined conditions."""
    instance_config = two_diff_depots_two_competing_customers_one_vehicle
    pprint(instance_config)
    instance = vrppd_dict_to_instance_obj(instance_config)
    model = Darp(instance)
    model.build(allow_rejection=allow_rejection, max_driving_time_h=max_driving_time_h)
    model.set_obj(obj)
    model.set_flex_depot(False)
    model.save_lp(f"{allow_rejection} - {max_driving_time_h}.lp")
    solution_obj = model.solve()

    if solution_obj.vehicle_routes:
        df = solution_obj.route_df(fn_dist=model.dist)
        actual_sol_obj = solution_obj.solver_stats.sol_objvalue
        print(df)
        pprint(solution_obj.solver_stats)
        pprint(solution_obj.routes_dict())
        assert set(solution_obj.routes_dict().values()) == expected_sol_route
        assert math.isclose(expected_sol_obj, actual_sol_obj, rel_tol=1e-6)
    else:
        assert expected_sol_route is None
