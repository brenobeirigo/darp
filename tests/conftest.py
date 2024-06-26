import pytest
import src.solver.darp as darp


def generate_trucks_dict(total_number: int, capacity: int, max_working_hours: int,
                         revenue_per_load_unit: int, cost_per_min: float = 20 / 60,
                         cost_per_km: float = 0.01, speed_km_h: int = 50) -> dict[str, float]:
    """Generate a dictionary representing truck configurations."""
    return {
        'capacity': capacity,
        'max_working_hours': max_working_hours,
        'revenue_per_load_unit': revenue_per_load_unit,
        'total_number': total_number,
        'speed_km_h': speed_km_h,
        'cost_per_min': cost_per_min,
        'cost_per_km': cost_per_km
    }


def create_config_dict(depots: list[dict[str, int]], pickup_locations: list[dict[str, int]],
                       delivery_locations: list[dict[str, int]], trucks: dict[str, float]) -> dict[str, list]:
    """Create a configuration dictionary for VRPPD instances."""
    return {
        'depots': depots,
        'pickup_locations': pickup_locations,
        'delivery_locations': delivery_locations,
        'trucks': trucks
    }

def generate_node(node_id: int, demand: int, tw_start: int, tw_end: int,
                  x_coord: int, y_coord: int) -> dict[str, int]:
    """Generate a dictionary representing a node configuration."""
    return {
        'demand': demand,
        'node_ID': node_id,
        'tw_end': tw_end,
        'tw_start': tw_start,
        'x_coord': x_coord,
        'y_coord': y_coord
    }

@pytest.fixture
def obj_config(request) -> dict[str, str]:
    """Fixture for dynamic objective configuration based on test parameter."""
    objective_type = request.param
    match objective_type:
        case "min_dist_traveled":
            return {"obj": darp.OBJ_MIN_TRAVEL_DISTANCE}
        case "max_profit":
            return {"obj": darp.OBJ_MAX_PROFIT}
        case _:
            return {"obj": darp.OBJ_MIN_TRAVEL_DISTANCE}

@pytest.fixture
def depot_config(request) -> dict[str, str]:
    """Fixture for dynamic depot constraint configuration based on test parameter."""
    depot_type = request.param
    match depot_type:
        case "depot_flex":
            return {"constr_depot": darp.CONSTR_FLEXIBLE_DEPOT}
        case _:
            return {"constr_depot": darp.CONSTR_FIXED_DEPOT}

@pytest.fixture
def one_depot_one_customer_one_vehicle(request) -> dict[str, list]:
    """Fixture for creating a simple VRPPD instance configuration."""
    params = request.param
    speed_km_h, capacity = params["speed"], params["capacity"]
    # 10km/60km/60min = 1/5*60 = 10
    # 10[0,500]-------->20[100,200]----->30[300,400]----->50[0,500]
    #      190  + 10 +              +100                +20 =  320
    depot = generate_node(1, 0, 0, 500, 10, 50)
    pickup = generate_node(2, 20, 100, 200, 20, 50)
    delivery = generate_node(3, -20, 300, 400, 30, 50)
    trucks = generate_trucks_dict(1, capacity, 480, 50, speed_km_h=speed_km_h)

    return create_config_dict([depot], [pickup], [delivery], trucks)

@pytest.fixture
def one_depot_one_customer_one_vehicle_no_tw(request) -> dict[str, list]:
    """Fixture for creating a simple VRPPD instance configuration."""
    params = request.param
    speed_km_h, capacity = params["speed"], params["capacity"]
    depot = generate_node(1, 0, 0, 480, 10, 50)
    pickup = generate_node(2, 20, 0, 480, 20, 50)
    delivery = generate_node(3, -20, 0, 480, 30, 50)
    trucks = generate_trucks_dict(1, capacity, 480, 50, speed_km_h=speed_km_h)

    return create_config_dict([depot], [pickup], [delivery], trucks)


@pytest.fixture
def two_depots_one_customer_one_vehicle(request) -> dict[str, list]:
    """Fixture for creating a simple VRPPD instance configuration."""
    params = request.param
    speed_km_h, capacity = params["speed"], params["capacity"]

    depot1 = generate_node(1, 0, 0, 500, 10, 50)
    depot2 = generate_node(2, 0, 0, 500, 0, 0)
    pickup = generate_node(3, 20, 100, 200, 20, 50)
    delivery = generate_node(4, -20, 300, 400, 30, 50)
    trucks = generate_trucks_dict(1, capacity, 480, 50, speed_km_h=speed_km_h)

    return create_config_dict([depot1, depot2], [pickup], [delivery], trucks)


@pytest.fixture
def two_diff_depots_one_customer_one_vehicle(request) -> dict[str, list]:
    """Fixture for creating a simple VRPPD instance configuration."""
    params = request.param
    speed_km_h, capacity = params["speed"], params["capacity"]

    depot1 = generate_node(1, 0, 0, 500, 10, 50)
    depot2 = generate_node(2, 0, 0, 500, 40, 50)
    pickup = generate_node(3, 20, 100, 200, 20, 50)
    delivery = generate_node(4, -20, 300, 400, 30, 50)
    trucks = generate_trucks_dict(1, capacity, 480, 50, speed_km_h=speed_km_h)

    return create_config_dict([depot1, depot2], [pickup], [delivery], trucks)


@pytest.fixture
def two_diff_depots_two_customers_one_vehicle(request) -> dict[str, list]:
    """Fixture for creating a simple VRPPD instance configuration."""
    params = request.param
    speed_km_h, capacity, max_working_hours = params["speed"], params["capacity"], params.get("max_working_hours", None)
    # 1--(10)-->3--(5)-->4--(5)-->5--(5)-->6--(10)-->8 
    # 10       20        25       30       35       40
    
    # 2--(15)-->4--(5)-->3--(10)-->5--(5)-->6--(5)-->8 = 40
    # 40       25        20        30       35       40
    
    # 2--(20)-->3--(5)-->4--(5)-->5--(5)-->6--(5)-->8 = 40
    # 40       20        25       30       35       40
    depot1 = generate_node(1, 0, 0, 500, 10, 50)
    depot2 = generate_node(2, 0, 0, 500, 40, 50)
    pickup1 = generate_node(3, 20, 0, 500, 20, 50)
    pickup2 = generate_node(4, 20, 0, 500, 25, 50)
    delivery1 = generate_node(5, -20, 0, 500, 30, 50)
    delivery2 = generate_node(6, -20, 0, 500, 35, 50)
    trucks = generate_trucks_dict(1, capacity, max_working_hours, 50, speed_km_h=speed_km_h)

    return create_config_dict([depot1, depot2], [pickup1, pickup2], [delivery1, delivery2], trucks)

@pytest.fixture
def two_diff_depots_two_competing_customers_one_vehicle(request) -> dict[str, list]:
    """Fixture for creating a simple VRPPD instance configuration."""
    params = request.param
    speed_km_h, capacity, max_working_hours = params["speed"], params["capacity"], params.get("max_working_hours", None)
    
    # Total distance is 40 (10, 10, 20)
    depot1 = generate_node(1, 0, 0, 500, 10, 50)
    pickup1 = generate_node(3, 20, 0, 500, 20, 50)
    delivery1 = generate_node(5, -20, 0, 500, 30, 50)
    
    # Total distance is 50 (10, 20, 30)
    depot2 = generate_node(2, 0, 0, 500, 50, 50)
    pickup2 = generate_node(4, 20, 0, 500, 60, 50)
    delivery2 = generate_node(6, -20, 0, 500, 80, 50)
    
    # [2, 4, 6, 3, 5, 8] = 100
    #    10 + 20 + 50 + 10 + 10 = 100
    trucks = generate_trucks_dict(1, capacity, max_working_hours, 50, speed_km_h=speed_km_h)

    return create_config_dict([depot1, depot2], [pickup1, pickup2], [delivery1, delivery2], trucks)