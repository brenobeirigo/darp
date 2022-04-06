from ..model.Request import Request
from ..model.Vehicle import Vehicle
from ..instance.Instance import Instance
import numpy as np
import itertools


PARSER_TYPE_CORDEAU = "cordeau_2006"
PARSER_TYPE_ROPKE = "ropke_2007"
    
def parse_node_line(line):
    (
        id,
        x,
        y,
        service_duration,
        load,
        earliest,
        latest,
    ) = line.split()
    return (
        int(id),
        float(x),
        float(y),
        int(service_duration),
        int(load),
        int(earliest),
        int(latest),
    )


def parse_request_line(pickup_node_line, dropoff_node_line, max_ride_time):

    (
        pickup_id,
        pickup_x,
        pickup_y,
        pickup_service_duration,
        pickup_load,
        pickup_earliest,
        pickup_latest,
    ) = parse_node_line(pickup_node_line)

    (
        dropoff_id,
        dropoff_x,
        dropoff_y,
        dropoff_service_duration,
        dropoff_load,
        dropoff_earliest,
        dropoff_latest,
    ) = parse_node_line(dropoff_node_line)

    pickup_point = (pickup_x, pickup_y)
    dropoff_point = (dropoff_x, dropoff_y)

    return Request(
        pickup_id,
        dropoff_id,
        pickup_earliest,
        pickup_latest,
        dropoff_earliest,
        dropoff_latest,
        load=pickup_load,
        pickup_point=pickup_point,
        dropoff_point=dropoff_point,
        pickup_delay=pickup_service_duration,
        dropoff_delay=dropoff_service_duration,
        max_ride_time=max_ride_time,
    )


def parse_request_lines(request_lines, max_ride_time):
    requests = list()
    n_customers = len(request_lines) // 2

    for i in range(n_customers):
        pickup_line = request_lines[i]
        dropoff_line = request_lines[i + n_customers]
        request = parse_request_line(pickup_line, dropoff_line, max_ride_time)
        requests.append(request)

    return requests


def parse_vehicle(origin_line, destination_line, vehicle_capacity):
    (
        o_id,
        o_x,
        o_y,
        _,
        _,
        o_earliest,
        o_latest,
    ) = parse_node_line(origin_line)

    o_point = (o_x, o_y)

    (
        d_id,
        d_x,
        d_y,
        _,
        _,
        d_earliest,
        d_latest,
    ) = parse_node_line(destination_line)

    d_point = (d_x, d_y)

    return Vehicle(
        o_id,
        vehicle_capacity,
        origin_point=o_point,
        destination_id=d_id,
        destination_point=d_point,
        origin_earliest_time=o_earliest,
        origin_latest_time=o_latest,
        destination_earliest_time=d_earliest,
        destination_latest_time=d_latest,
    )


def parse_dist_matrix_from_node_points(nodes):

    dist_matrix = {
        o.pos: {
            d.pos: o.point.distance(d.point)
            for d in nodes
        }
        for o in nodes
    }

    return dist_matrix


def get_node_list(requests, vehicles):
    pickups, dropoffs = zip(
        *[(r.pickup_node, r.dropoff_node) for r in requests]
    )
    origins, destinations = zip(
        *[(v.origin_node, v.destination_node) for v in vehicles]
    )

    all_nodes = list(itertools.chain(pickups, dropoffs, origins, destinations))
    return all_nodes


def cordeau_parser(instance_path):

    with open(instance_path, "r") as file:
        lines = file.readlines()

        config_dict = get_config_dict(lines[0])

        o_depot = lines[1]
        d_depot = lines[-1]
        request_lines = lines[2:-1]

        vehicles = [
            parse_vehicle(o_depot, d_depot, config_dict["vehicle_capacity"])
            for _ in range(config_dict["n_vehicles"])
        ]


        requests = parse_request_lines(
            request_lines, config_dict["maximum_ride_time_min"]
        )

        nodes = get_node_list(requests, vehicles)

        dist_matrix = parse_dist_matrix_from_node_points(nodes)

        return vehicles, requests, nodes, dist_matrix, config_dict
    
def ropke_parser(instance_path):

    with open(instance_path, "r") as file:
        lines = file.readlines()

        config_dict = get_config_dict(lines[0])

        o_depot = lines[1]
        d_depot = lines[1]
        request_lines = lines[2:]

        vehicles = [
            parse_vehicle(o_depot, d_depot, config_dict["vehicle_capacity"])
            for _ in range(config_dict["n_vehicles"])
        ]


        requests = parse_request_lines(
            request_lines, config_dict["maximum_ride_time_min"]
        )

        nodes = get_node_list(requests, vehicles)

        dist_matrix = parse_dist_matrix_from_node_points(nodes)

        return vehicles, requests, nodes, dist_matrix, config_dict


def get_config_dict(config_line):

    config = map(int, np.array(config_line.split()))
    (
        n_vehicles,
        n_customers,
        time_horizon_min,
        vehicle_capacity,
        maximum_ride_time_min,
    ) = config

    config_dict = dict(
        n_vehicles=n_vehicles,
        n_customers=n_customers,
        time_horizon_min=time_horizon_min,
        vehicle_capacity=vehicle_capacity,
        maximum_ride_time_min=maximum_ride_time_min,
    )
    return config_dict


parsers = {PARSER_TYPE_CORDEAU: cordeau_parser, PARSER_TYPE_ROPKE: ropke_parser}

def parse_instance_from_filepath(
    instance_filepath, instance_parser=PARSER_TYPE_CORDEAU
):
    vehicles, requests, nodes, dist_matrix, config_dict = parsers[
        instance_parser
    ](instance_filepath)
    config_dict["type"] = instance_parser
    config_dict["path"] = instance_filepath

    return Instance(vehicles, requests, nodes, dist_matrix, config_dict)


# dist = defaultdict(dict)
# dist[pickup_id][dropoff_id] = pickup_point.distance(dropoff_point)
# dist[dropoff_id][pickup_id] = dropoff_point.distance(pickup_point)

# pprint(dist)
