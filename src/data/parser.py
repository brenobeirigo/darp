from ..model.Request import Request
from ..model.Vehicle import Vehicle
from .instance import Instance, InstanceConfig
import itertools
from ..model.node import NodeInfo, NodeType
from ..model.node import Node


PARSER_TYPE_CORDEAU = "cordeau_2006"
PARSER_TYPE_ROPKE = "ropke_2007"
 
def parse_node_line(line:str, node_type:NodeType) -> NodeInfo:
    id, x, y, service_duration, load, earliest, latest = line.split()
    return NodeInfo(
        int(id),
        float(x),
        float(y),
        int(service_duration),
        int(load),
        int(earliest),
        int(latest),
        node_type
    )

def parse_request(pu_line:str, do_line:str, max_ride_time:int):
    pickup = parse_node_line(pu_line, node_type=NodeType.PU)
    dropoff = parse_node_line(do_line, node_type=NodeType.DO)
    return Request(pickup, dropoff, max_ride_time=max_ride_time)


def parse_requests(request_lines:list[str], max_ride_time:int):
    requests = []
    n_customers = len(request_lines) // 2

    for i in range(n_customers):
        pu_line = request_lines[i]
        do_line = request_lines[i + n_customers]
        request = parse_request(pu_line, do_line, max_ride_time)
        requests.append(request)

    return requests


def parse_vehicle(origin_line: str, destination_line:str, vehicle_capacity:int)->Vehicle:
    depot_o = parse_node_line(origin_line, node_type=NodeType.DEPOT_ORIGIN)
    depot_d = parse_node_line(destination_line, node_type=NodeType.DEPOT_DESTINATION)
    return Vehicle(depot_o, depot_d, vehicle_capacity)

def get_node_list(requests: list[Request], vehicles: list[Vehicle])->list[Node]:
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

        config = InstanceConfig(*map(int,lines[0].split()))

        o_depot = lines[1]
        d_depot = lines[-1]
        request_lines = lines[2:-1]

        vehicles = [
            parse_vehicle(o_depot, d_depot, config.vehicle_capacity)
            for _ in range(config.n_vehicles)
        ]


        requests = parse_requests(
            request_lines, config.maximum_ride_time_min
        )

        nodes = get_node_list(requests, vehicles)

        return Instance(
            vehicles,
            requests,
            nodes,
            config,
            instance_path,
            "cordeau")
    
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


parsers = {PARSER_TYPE_CORDEAU: cordeau_parser} #, PARSER_TYPE_ROPKE: ropke_parser}

def parse_instance_from_filepath(
        instance_filepath,
        instance_parser=PARSER_TYPE_CORDEAU
    ) -> Instance:
    
    p = parsers[instance_parser]
    i = p(instance_filepath)
    return i
