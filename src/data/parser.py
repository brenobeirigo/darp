from ..model.Request import Request
from ..model.Vehicle import Vehicle, MDVRP, CVRP, create_vehicle
from .instance import Instance, InstanceConfig
import itertools
from ..model.node import NodeInfo, NodeType
from ..model.node import Node, OriginNode, DestinationNode


PARSER_TYPE_CORDEAU = "cordeau_2006"
PARSER_TYPE_ROPKE = "ropke_2007"
PARSER_TYPE_MVRPPDTW = "mvrppdtw"


def parse_node_line(
    line: str, node_type: NodeType, alias: str = None
) -> NodeInfo:
    id, x, y, service_duration, load, earliest, latest = line.split()
    return NodeInfo(
        int(id),
        float(x),
        float(y),
        int(service_duration),
        int(load),
        int(earliest),
        int(latest),
        node_type,
        alias,
    )


def parse_requests(
    request_lines: list[str], max_ride_time: int
) -> list[Request]:
    requests = []
    pickups = []
    dropoffs = []
    n_customers = len(request_lines) // 2

    for i in range(n_customers):
        pu_line = request_lines[i]
        pickup = parse_node_line(pu_line, node_type=NodeType.PU)
        pickups.append(pickup)

        do_line = request_lines[i + n_customers]
        dropoff = parse_node_line(
            do_line, node_type=NodeType.DO, alias=f"{pickup.id}*"
        )
        dropoffs.append(dropoff)

        request = Request(pickup, dropoff, max_ride_time=max_ride_time)
        requests.append(request)

    return pickups, dropoffs, requests


def get_node_list(
    requests: list[Request], vehicles: list[Vehicle]
) -> list[Node]:
    pickups, dropoffs = zip(
        *[(r.pickup_node, r.dropoff_node) for r in requests]
    )
    origins, destinations = zip(
        *[(v.origin_node, v.destination_node) for v in vehicles]
    )

    all_nodes = list(itertools.chain(origins, pickups, dropoffs, destinations))
    return all_nodes


def cordeau_parser(instance_path) -> Instance:
    with open(instance_path, "r") as file:
        lines = file.readlines()

        config = InstanceConfig(*map(int, lines[0].split()))

        # Create vehicles at the origin
        o_depot_line = lines[1]
        depot_o = parse_node_line(
            o_depot_line, node_type=NodeType.O_DEPOT, alias="Depot"
        )

        # Parse all request lines
        request_lines = lines[2 : 2 * config.n_customers + 2]
        pu_nodes, do_nodes, requests = parse_requests(
            request_lines, config.maximum_ride_time_min
        )

        # Some instances do not copy the depot to the last node.
        # In this case, we create a dummy node for the aux. destination depot.
        try:
            d_depot_line = lines[2 * config.n_customers + 2]
            depot_d = parse_node_line(
                d_depot_line, node_type=NodeType.D_DEPOT, alias="Depot"
            )
        except:
            depot_d = NodeInfo(
                2 * config.n_customers + 2,
                depot_o.x,
                depot_o.y,
                depot_o.service_duration,
                depot_o.load,
                depot_o.earliest,
                depot_o.latest,
                NodeType.D_DEPOT,
                "Depot",
            )

        vehicles = [
            create_vehicle(
                problem_type=CVRP,
                capacity=config.vehicle_capacity,
                node_o=depot_o,
                node_d=depot_d)
            
            for _ in range(config.n_vehicles)
        ]

        nodes = [depot_o] + pu_nodes + do_nodes + [depot_d]

        return Instance(
            vehicles, requests, nodes, config, instance_path, "cordeau"
        )


def vrppd_instance_to_dict(instance_path):

    with open(instance_path, "r") as file:
        lines = file.readlines()

        # Initialize dictionary structure
        instance_dict = {
            "grid_cardinality": int(lines[1]),
            "pickup_locations_count": int(lines[3]),
            "delivery_locations_count": int(lines[5]),
            "depots_count": int(lines[7]),
            "trucks": {
                "total_number": int(lines[9]),
                "capacity": int(lines[11]),
                "max_working_hours": int(lines[13])*60,
                "revenue_per_load_unit": int(lines[15]),  # Assume revenue is followed by "[e/kg]"
                "cost_per_min":20/60,
                "cost_per_km":0.05*0.2,
                "speed_km_h":50
            },
            "depots": [],
            "pickup_locations": [],
            "delivery_locations": [],
        }

        


        # Parsing depots
        line_index = 18  # Starting line index for depots
        for _ in range(instance_dict["depots_count"]):
            parts = lines[line_index].split()
            instance_dict["depots"].append({
                "node_ID": int(parts[0]),
                "x_coord": float(parts[1]),
                "y_coord": float(parts[2]),
                "demand": float(parts[3]),
                "tw_start": float(parts[4])*60,
                "tw_end": float(parts[5])*60,
            })
            line_index += 1

        # Skip to pickup locations
        line_index += 2  # Skipping the header of the next section
        for _ in range(instance_dict["pickup_locations_count"]):
            parts = lines[line_index].split()
            instance_dict["pickup_locations"].append({
                "node_ID": int(parts[0]),
                "x_coord": float(parts[1]),
                "y_coord": float(parts[2]),
                "demand": float(parts[3]),
                "tw_start": float(parts[4])*60,
                "tw_end": float(parts[5])*60,
            })
            line_index += 1

        
        # Skip to delivery locations
        line_index += 2  # Skipping the header of the next section
        for _ in range(instance_dict["delivery_locations_count"]):
            parts = lines[line_index].split()
            instance_dict["delivery_locations"].append({
                "node_ID": int(parts[0]),
                "x_coord": float(parts[1]),
                "y_coord": float(parts[2]),
                "demand": float(parts[3]),
                "tw_start": float(parts[4])*60,
                "tw_end": float(parts[5])*60,
            })
            line_index += 1


        return instance_dict

def vrppd_dict_to_instance_obj(instance_dict, instance_path=""):
    config = InstanceConfig(
        n_vehicles=instance_dict["trucks"]["total_number"],
        n_customers=len(instance_dict["pickup_locations"]),
        n_depots=len(instance_dict["depots"]),
        vehicle_capacity=instance_dict["trucks"]["capacity"],
        maximum_ride_time_min=instance_dict["trucks"]["max_working_hours"],
        time_horizon_min=instance_dict["trucks"]["max_working_hours"],
    )
    
    depots_o = []
    depots_d = []
    for depot_info in instance_dict["depots"]:
        depot_o = NodeInfo(
            id=depot_info["node_ID"],
            x=depot_info["x_coord"],
            y=depot_info["y_coord"],
            service_duration=0,
            load=depot_info["demand"],
            earliest=depot_info["tw_start"],
            latest=depot_info["tw_end"],
            type=NodeType.O_DEPOT,
        )
        depots_o.append(depot_o)

        depot_d = NodeInfo(
            id=depot_info["node_ID"] + 2*config.n_customers + config.n_depots,
            x=depot_info["x_coord"],
            y=depot_info["y_coord"],
            service_duration=0,
            load=depot_info["demand"],
            earliest=depot_info["tw_start"],
            latest=depot_info["tw_end"],
            type=NodeType.D_DEPOT,
        )

        depots_d.append(depot_d)
    
    pickup_nodes = []

    for pickup_info in instance_dict["pickup_locations"]:
        pickup_node = NodeInfo(
            id=pickup_info["node_ID"],
            x=pickup_info["x_coord"],
            y=pickup_info["y_coord"],
            service_duration=0,
            load=pickup_info["demand"],
            earliest=pickup_info["tw_start"],
            latest=pickup_info["tw_end"],
            type=NodeType.PU,
        )
        pickup_nodes.append(pickup_node)

    dropoff_nodes = []

    for delivery_info in instance_dict["delivery_locations"]:
        dropoff_node = NodeInfo(
            id=delivery_info["node_ID"],
            x=delivery_info["x_coord"],
            y=delivery_info["y_coord"],
            service_duration=0,
            load=delivery_info["demand"],
            earliest=delivery_info["tw_start"],
            latest=delivery_info["tw_end"],
            type=NodeType.DO,
            alias=f"{delivery_info['node_ID'] - config.n_customers}*",
        )
        dropoff_nodes.append(dropoff_node)

    requests = []
    for o, d in zip(pickup_nodes, dropoff_nodes):
        r = Request(o,d,max_ride_time=config.time_horizon_min)
        requests.append(r)
                
    vehicles = [
            create_vehicle(
                problem_type=MDVRP,
                capacity=config.vehicle_capacity,
                cost_per_min=instance_dict["trucks"]["cost_per_min"],
                cost_per_km=instance_dict["trucks"]["cost_per_km"],
                speed_km_h=instance_dict["trucks"]["speed_km_h"],
                revenue_per_load_unit=instance_dict["trucks"]["revenue_per_load_unit"],
    )
            for _ in range(config.n_vehicles)
        ]
    
    nodes = depots_o + pickup_nodes + dropoff_nodes + depots_d
    from pprint import pprint
    pprint(nodes)
    return Instance(
            vehicles,
            requests,
            nodes,
            config,
            instance_path,
            PARSER_TYPE_MVRPPDTW
        )
# Adapted `cordeau_parser` to match the provided instance structure
def vrppd_parser(instance_path):
    
    from pprint import pprint
    instance_dict = vrppd_instance_to_dict(instance_path)
    pprint(instance_dict)

    return vrppd_dict_to_instance_obj(
        instance_dict,
        instance_path=instance_path)

# def ropke_parser(instance_path):

#     with open(instance_path, "r") as file:
#         lines = file.readlines()

#         config_dict = get_config_dict(lines[0])

#         o_depot = lines[1]
#         d_depot = lines[1]
#         request_lines = lines[2:]

#         vehicles = [
#             parse_vehicle(o_depot, d_depot, config_dict.vehicle_capacity)
#             for _ in range(config_dict.n_vehicles)
#         ]


#         requests = parse_customer_requests(
#             request_lines, config_dict.maximum_ride_time_min
#         )

#         nodes = get_node_list(requests, vehicles)

#         dist_matrix = parse_dist_matrix_from_node_points(nodes)

#         return vehicles, requests, nodes, dist_matrix, config_dict


parsers = {
    PARSER_TYPE_CORDEAU: cordeau_parser,
    PARSER_TYPE_MVRPPDTW: vrppd_parser
}  # , PARSER_TYPE_ROPKE: ropke_parser}


def parse_instance_from_filepath(
    instance_filepath, instance_parser=PARSER_TYPE_CORDEAU
) -> Instance:
    p = parsers[instance_parser]
    i = p(instance_filepath)
    return i
